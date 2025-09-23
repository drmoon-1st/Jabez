import time
from pathlib import Path
import argparse

import numpy as np
import cv2

try:
    import yaml
except ImportError:
    yaml = None

import pyrealsense2 as rs

from utils_io import ensure_dir, save_json

# (선택) MediaPipe로 2D keypoints → depth로 3D 추출
USE_MEDIAPIPE = True
try:
    import mediapipe as mp
except Exception:
    USE_MEDIAPIPE = False

def load_cfg(p: Path):
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pip install pyyaml")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise ValueError("Use YAML for capture config.")

def init_pipeline(cfg):
    pipeline = rs.pipeline()
    cfg_rs = rs.config()
    c = cfg["color"]; d = cfg["depth"]
    cfg_rs.enable_stream(rs.stream.color, c["width"], c["height"], rs.format.bgr8, c["fps"])
    cfg_rs.enable_stream(rs.stream.depth, d["width"], d["height"], rs.format.z16, d["fps"])
    profile = pipeline.start(cfg_rs)
    align = rs.align(rs.stream.color)
    return pipeline, profile, align

def write_intrinsics(profile, out_path: Path):
    vs = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = vs.get_intrinsics()
    data = dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy, coeffs=list(intr.coeffs))
    save_json(out_path, data)

def deproject(intr_dict, u, v, depth_mm):
    if depth_mm <= 0: return None
    intr = rs.intrinsics()
    intr.width = intr_dict["width"]; intr.height = intr_dict["height"]
    intr.fx = intr_dict["fx"]; intr.fy = intr_dict["fy"]
    intr.ppx = intr_dict["ppx"]; intr.ppy = intr_dict["ppy"]
    intr.model = rs.distortion.none
    intr.coeffs = intr_dict["coeffs"]
    xyz_m = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_mm)/1000.0)
    return np.array(xyz_m, dtype=np.float32) * 1000.0  # mm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "capture.yaml"))
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    out_dir = Path(cfg["out_dir"])
    sub = cfg.get("subdirs", {})
    color_dir = out_dir / sub.get("color","color")
    depth_dir = out_dir / sub.get("depth","depth")
    ensure_dir(color_dir); ensure_dir(depth_dir)

    pipeline, profile, align = init_pipeline(cfg)
    intr_path = out_dir / cfg["files"]["intrinsics"]
    write_intrinsics(profile, intr_path)
    intr = (intr_path.read_text())
    intr = eval(intr) if isinstance(intr, str) else intr  # robust load (or use json.loads)

    # MediaPipe 준비
    backend = cfg["pose"]["backend"]
    use_pose = (backend == "mediapipe") and USE_MEDIAPIPE
    if use_pose:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    csv_out = out_dir / cfg["files"]["csv_2d3d"]
    write_csv = bool(cfg["files"]["csv_2d3d"])
    if write_csv:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fcsv = open(csv_out, "w", encoding="utf-8")
        # 헤더(예시): frame, Nose_x, Nose_y, Nose_z, ..., (원하면 Body25/MP index에 맞춰 확장)
        headers = ["frame","Nose_x","Nose_y","Nose_z"]
        fcsv.write(",".join(headers)+"\n")
    else:
        fcsv = None

    mode = cfg.get("mode","interactive")
    duration = int(cfg.get("duration_sec",5))
    t0 = time.time()
    idx = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color: continue
            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())  # uint16 mm

            # 저장
            cv2.imwrite(str(color_dir / f"{idx:06d}.png"), color_img)
            np.save(depth_dir / f"{idx:06d}.npy", depth_img)

            # (선택) 2D→3D CSV 기록 (Nose만 예시)
            if write_csv and use_pose:
                rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    h, w = color_img.shape[:2]
                    nose = res.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                    u, v = int(nose.x*w), int(nose.y*h)
                    u = np.clip(u,0,w-1); v = np.clip(v,0,h-1)
                    d_mm = int(depth_img[v,u])
                    xyz = deproject(eval(intr) if isinstance(intr,str) else intr, u, v, d_mm)
                    if xyz is not None:
                        fcsv.write(f"{idx},{xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}\n")
                    else:
                        fcsv.write(f"{idx},,,\n")
                else:
                    fcsv.write(f"{idx},,,\n")

            idx += 1

            if mode == "duration" and (time.time()-t0) >= duration:
                break
            if mode == "interactive":
                cv2.imshow("D455 Preview", color_img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
    finally:
        pipeline.stop()
        if fcsv: fcsv.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
