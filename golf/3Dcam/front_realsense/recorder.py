"""
recorder.py

RealsenseRecorder를 기존 예제 파일에서 재사용할 수 있도록 래퍼 제공.
이 모듈은 프로젝트 루트의 `realsense_pack_and_upload/realsense_pack_and_upload.py`에서
`RealsenseRecorder`를 임포트 시도합니다.
"""
from pathlib import Path
import threading
import time
import traceback
import json
from collections import deque

import numpy as np
import cv2
try:
    import pyrealsense2 as rs
except Exception:
    rs = None


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / 'realsense_pack_and_upload'

def _try_import_example():
    try:
        import sys
        if str(EXAMPLE_DIR) not in sys.path:
            sys.path.insert(0, str(EXAMPLE_DIR))
        import realsense_pack_and_upload as rp
        return getattr(rp, 'RealsenseRecorder', None)
    except Exception:
        return None


# If pyrealsense2 is available, provide a local RealsenseRecorder implementation
if rs is not None:
    class RealsenseRecorder:
        def __init__(self, out_dir: Path, width=640, height=480, fps=30):
            self.out_dir = Path(out_dir)
            self.width = width
            self.height = height
            self.fps = fps
            self._thread = None
            self._stop_event = threading.Event()
            self._running = threading.Event()
            self._idx = 0
            # last captured color frame (numpy array) for preview
            self._last_color = None
            self._frame_lock = threading.Lock()

        def start(self):
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._running.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            while not self._running.is_set():
                time.sleep(0.01)

        def stop(self):
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=5.0)

        def _run(self):
            color_dir = self.out_dir / "color"
            depth_dir = self.out_dir / "depth"
            color_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            align = rs.align(rs.stream.color)

            profile = pipeline.start(config)
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                intr = None
                self._running.set()
                self._idx = 0
                while not self._stop_event.is_set():
                    frames = pipeline.wait_for_frames()
                    frames = align.process(frames)
                    depth = frames.get_depth_frame()
                    color = frames.get_color_frame()
                    if not depth or not color:
                        continue

                    if intr is None:
                        try:
                            ci = color.profile.as_video_stream_profile().intrinsics
                            intr = {
                                "width": ci.width,
                                "height": ci.height,
                                "fx": float(ci.fx),
                                "fy": float(ci.fy),
                                "cx": float(ci.ppx),
                                "cy": float(ci.ppy),
                                "coeffs": [float(c) for c in getattr(ci, 'coeffs', [])]
                            }
                            meta = {"fps": self.fps, "width": self.width, "height": self.height, "depth_scale": float(depth_scale)}
                            intr_obj = {"color_intrinsics": intr, "meta": meta}
                            try:
                                (self.out_dir / "intrinsics.json").write_text(json.dumps(intr_obj, ensure_ascii=False, indent=2), encoding='utf-8')
                            except Exception:
                                pass
                        except Exception:
                            intr = None

                    color_img = np.asanyarray(color.get_data())
                    # update last frame for live preview (thread-safe)
                    try:
                        with self._frame_lock:
                            # store a copy to avoid referencing the underlying buffer
                            self._last_color = color_img.copy()
                    except Exception:
                        pass
                    depth_raw = np.asanyarray(depth.get_data())
                    depth_m = (depth_raw.astype(np.float32) * depth_scale)

                    color_path = color_dir / f"{self._idx:06d}.png"
                    depth_path = depth_dir / f"{self._idx:06d}.npy"
                    cv2.imwrite(str(color_path), color_img)
                    np.save(depth_path, depth_m)
                    self._idx += 1

            except Exception:
                traceback.print_exc()
            finally:
                pipeline.stop()
                self._running.clear()
                # clear last frame
                try:
                    with self._frame_lock:
                        self._last_color = None
                except Exception:
                    pass

else:
    # fallback: try to import example RealsenseRecorder
    _Example = _try_import_example()
    if _Example is not None:
        RealsenseRecorder = _Example
    else:
        RealsenseRecorder = None


class RecorderWrapper:
    def __init__(self, out_dir: str, width=640, height=480, fps=30):
        if RealsenseRecorder is None:
            raise RuntimeError('RealSense support not available (pyrealsense2 not installed and example class not found).')
        self._rec = RealsenseRecorder(Path(out_dir), width=width, height=height, fps=fps)

    def start(self):
        self._rec.start()

    def stop(self):
        self._rec.stop()

    def is_running(self):
        return getattr(self._rec, '_running', None) is not None

    def get_last_frame(self):
        """Return the most recent color frame (numpy array) or None."""
        try:
            lf = getattr(self._rec, '_last_color', None)
            lock = getattr(self._rec, '_frame_lock', None)
            if lock is not None:
                with lock:
                    return lf.copy() if lf is not None else None
            else:
                return lf.copy() if lf is not None else None
        except Exception:
            return None

    def frame_count(self):
        color_dir = Path(self._rec.out_dir) / 'color'
        if not color_dir.exists():
            return 0
        return len(list(color_dir.glob('*.png')))


class BufferedRealsenseRecorder:
    """Continuously capture frames into an in-memory ring buffer.
    Call `start_saving(out_dir)` to write buffered frames and continue writing live frames.
    """
    def __init__(self, buf_seconds=10, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self._buf_seconds = int(buf_seconds)
        self._maxlen = max(1, int(self._buf_seconds * self.fps))
        self._buffer = deque(maxlen=self._maxlen)  # store tuples (color, depth)
        self._thread = None
        self._stop_event = threading.Event()
        self._running = threading.Event()
        self._frame_lock = threading.Lock()
        self._last_color = None
        self._idx = 0
        self._saving = threading.Event()
        self._save_lock = threading.Lock()
        self._intrinsics_obj = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._running.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        while not self._running.is_set():
            time.sleep(0.01)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def is_running(self):
        return self._running.is_set()

    def get_last_frame(self):
        try:
            with self._frame_lock:
                return self._last_color.copy() if self._last_color is not None else None
        except Exception:
            return None

    def start_saving(self, out_dir: Path, drain: bool = True):
        """Start writing buffered frames and subsequent frames to disk under out_dir."""
        out_dir = Path(out_dir)
        color_dir = out_dir / 'color'
        depth_dir = out_dir / 'depth'
        color_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        # write intrinsics if available
        try:
            if self._intrinsics_obj is not None:
                (out_dir / "intrinsics.json").write_text(json.dumps(self._intrinsics_obj, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

        # store save dirs on the instance so _run() can write live frames
        self._save_color_dir = color_dir
        self._save_depth_dir = depth_dir

        def _drain_and_flag():
            # Write existing buffered frames first (if drain=True), otherwise discard buffer and start from current
            try:
                with self._save_lock:
                    # start each save session from index 0 so each recording has its own numbering
                    idx = 0
                    if drain:
                        # copy buffer snapshot
                        with self._frame_lock:
                            items = list(self._buffer)
                        for color_img, depth_m in items:
                            color_path = color_dir / f"{idx:06d}.png"
                            depth_path = depth_dir / f"{idx:06d}.npy"
                            try:
                                cv2.imwrite(str(color_path), color_img)
                            except Exception:
                                pass
                            try:
                                if depth_m is not None:
                                    np.save(depth_path, depth_m)
                            except Exception:
                                pass
                            idx += 1
                    else:
                        # discard buffered frames: reset index and clear buffer
                        with self._frame_lock:
                            try:
                                self._buffer.clear()
                            except Exception:
                                self._buffer = deque(maxlen=self._maxlen)
                    # set index to continue
                    self._idx = idx
                    # enable live saving
                    self._saving.set()
            except Exception:
                pass

        threading.Thread(target=_drain_and_flag, daemon=True).start()

    def stop_saving(self):
        self._saving.clear()

    def is_saving(self):
        return self._saving.is_set()

    def _run(self):
        # If pyrealsense2 is available, use RealSense pipeline, else fallback to OpenCV capture
        if rs is not None:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            align = rs.align(rs.stream.color)
            profile = pipeline.start(config)
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                intr = None
                self._running.set()
                while not self._stop_event.is_set():
                    frames = pipeline.wait_for_frames()
                    frames = align.process(frames)
                    depth = frames.get_depth_frame()
                    color = frames.get_color_frame()
                    if not depth or not color:
                        continue
                    if intr is None:
                        try:
                            ci = color.profile.as_video_stream_profile().intrinsics
                            intr = {
                                "width": ci.width,
                                "height": ci.height,
                                "fx": float(ci.fx),
                                "fy": float(ci.fy),
                                "cx": float(ci.ppx),
                                "cy": float(ci.ppy),
                                "coeffs": [float(c) for c in getattr(ci, 'coeffs', [])]
                            }
                            meta = {"fps": self.fps, "width": self.width, "height": self.height, "depth_scale": float(depth_scale)}
                            self._intrinsics_obj = {"color_intrinsics": intr, "meta": meta}
                        except Exception:
                            self._intrinsics_obj = None

                    color_img = np.asanyarray(color.get_data())
                    depth_raw = np.asanyarray(depth.get_data())
                    depth_m = (depth_raw.astype(np.float32) * depth_scale)

                    with self._frame_lock:
                        self._last_color = color_img.copy()
                        # append copy to buffer
                        try:
                            self._buffer.append((color_img.copy(), depth_m.copy()))
                        except Exception:
                            # fall back to appending color only
                            try:
                                self._buffer.append((color_img.copy(), None))
                            except Exception:
                                pass

                    # if saving enabled, write this frame to the active save dirs
                    if self._saving.is_set():
                        try:
                            with self._save_lock:
                                # ensure target dirs were set by start_saving
                                cdir = getattr(self, '_save_color_dir', None)
                                ddir = getattr(self, '_save_depth_dir', None)
                                if cdir is not None:
                                    try:
                                        color_path = cdir / f"{self._idx:06d}.png"
                                        cv2.imwrite(str(color_path), color_img)
                                    except Exception:
                                        pass
                                if ddir is not None:
                                    try:
                                        np.save(ddir / f"{self._idx:06d}.npy", depth_m)
                                    except Exception:
                                        pass
                                self._idx += 1
                        except Exception:
                            pass

            except Exception:
                traceback.print_exc()
            finally:
                try:
                    pipeline.stop()
                except Exception:
                    pass
                self._running.clear()
                with self._frame_lock:
                    self._last_color = None
        else:
            # OpenCV fallback (no depth)
            try:
                import cv2 as _cv2
                try_backends = [getattr(_cv2, 'CAP_DSHOW', 700), None]
                cap = None
                for b in try_backends:
                    try:
                        if b is None:
                            cap = _cv2.VideoCapture(0)
                        else:
                            cap = _cv2.VideoCapture(0, b)
                        time.sleep(0.2)
                        if cap is not None and cap.isOpened():
                            break
                    except Exception:
                        cap = None
                if cap is None or not cap.isOpened():
                    return
                self._running.set()
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue
                    with self._frame_lock:
                        self._last_color = frame.copy()
                        try:
                            self._buffer.append((frame.copy(), None))
                        except Exception:
                            pass
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception:
                pass


class ContinuousRecorderWrapper:
    """Facade used by UI: continuously buffers frames and can start/stop saving."""
    def __init__(self, buf_seconds=10, width=640, height=480, fps=30):
        # Choose buffered recorder implementation
        self._rec = BufferedRealsenseRecorder(buf_seconds=buf_seconds, width=width, height=height, fps=fps)

    def start(self):
        self._rec.start()

    def stop(self):
        self._rec.stop()

    def get_last_frame(self):
        return self._rec.get_last_frame()

    def start_saving(self, out_dir: str, drain: bool = True):
        # default behavior: drain existing buffer and save it
        self._rec.start_saving(Path(out_dir), drain=drain)

    def stop_saving(self):
        self._rec.stop_saving()

    def is_saving(self):
        return self._rec.is_saving()
