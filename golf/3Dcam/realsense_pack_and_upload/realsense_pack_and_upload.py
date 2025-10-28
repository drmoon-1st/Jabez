"""
realsense_pack_and_upload.py

Small Tkinter GUI program that records RGB (PNG) + depth (npy) from an Intel RealSense camera,
packages the recorded files into a ZIP and uploads it to a provided pre-signed S3 URL.

Usage:
  - Run the script (or build into an exe with PyInstaller) on a Windows PC with RealSense.
  - Provide an output folder name and the pre-signed URL (PUT) you received from your server.
  - Press Start Recording, then Stop Recording when finished. Then press Package & Upload.

Notes:
  - This script depends on pyrealsense2 which must be installed on the machine (and the
    Intel RealSense SDK runtime must be available). See requirements.txt.
  - Upload is performed with the requests library when available. If requests is missing,
    urllib will be used as a fallback.
  - The script intentionally uses a simple UI and stores frames as PNG + .npy depth arrays.

"""
import os
import threading
import time
import tempfile
import zipfile
import traceback
import mimetypes
import concurrent.futures
import json
from datetime import datetime
import uuid
from pathlib import Path
from tkinter import Tk, Button, Label, Entry, StringVar, filedialog, messagebox

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

import numpy as np
import pyrealsense2 as rs
import cv2

# Native Windows message box helper (falls back to tkinter.messagebox)
def show_native_error(title: str, message: str):
    try:
        # Windows native MessageBoxW for more visible system dialog
        import ctypes
        MB_OK = 0x00000000
        MB_ICONERROR = 0x00000010
        MB_TOPMOST = 0x00040000
        ctypes.windll.user32.MessageBoxW(None, str(message), str(title), MB_OK | MB_ICONERROR | MB_TOPMOST)
        return
    except Exception:
        try:
            messagebox.showerror(title, message)
        except Exception:
            print(f"ERROR: {title}: {message}")


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

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._running.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # wait a small moment until pipeline is ready
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
            self._running.set()
            self._idx = 0
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue

                color_img = np.asanyarray(color.get_data())
                depth_raw = np.asanyarray(depth.get_data())
                depth_m = (depth_raw.astype(np.float32) * depth_scale)

                # Save files
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


def zip_dir(src_dir: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(src_dir)
                zf.write(full, arcname=str(rel))


CONFIG_PATH = Path(__file__).with_suffix('.config.json')


def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_config(cfg: dict):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def get_presigned_url(server_base: str, object_name: str, timeout=30):
    """POST to {server_base}/get_presigned_url with JSON {'object_name': object_name}.
    Returns (success, url_or_error_message).
    """
    server_base = server_base.rstrip('/')
    endpoint = f"{server_base}/get_presigned_url"
    payload = {"object_name": object_name}
    headers = {"Content-Type": "application/json"}
    try:
        if _HAS_REQUESTS:
            res = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            text = res.text
            try:
                j = res.json()
            except Exception:
                j = None
            if res.status_code >= 400:
                return False, f"HTTP {res.status_code}: {text}"
            # Try to extract URL from JSON
            if isinstance(j, dict):
                url = None
                for k in ("url", "presigned_url", "put_url", "upload_url"):
                    if k in j and isinstance(j[k], str):
                        url = j[k]
                        break
                # if only raw string under other key
                if not url:
                    for v in j.values():
                        if isinstance(v, str) and v.startswith('http'):
                            url = v
                            break
                # optional required headers
                req_hdrs = None
                for hk in ("required_headers", "requiredHeaders", "headers", "signed_headers"):
                    if hk in j and isinstance(j[hk], dict):
                        req_hdrs = j[hk]
                        break
                if url:
                    if req_hdrs:
                        return True, {"url": url, "required_headers": req_hdrs}
                    else:
                        return True, url
            # fallback to raw text
            if text and text.strip().startswith('http'):
                return True, text.strip()
            return False, f"Cannot parse response: {text}"
        else:
            # urllib fallback
            from urllib.request import Request, urlopen
            data = json.dumps(payload).encode('utf-8')
            req = Request(endpoint, data=data, headers=headers, method='POST')
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode('utf-8')
                try:
                    j = json.loads(raw)
                except Exception:
                    j = None
                if isinstance(j, dict):
                    url = None
                    for k in ("url", "presigned_url", "put_url", "upload_url"):
                        if k in j and isinstance(j[k], str):
                            url = j[k]
                            break
                    req_hdrs = None
                    for hk in ("required_headers", "requiredHeaders", "headers", "signed_headers"):
                        if hk in j and isinstance(j[hk], dict):
                            req_hdrs = j[hk]
                            break
                    if url:
                        if req_hdrs:
                            return True, {"url": url, "required_headers": req_hdrs}
                        else:
                            return True, url
                if raw.strip().startswith('http'):
                    return True, raw.strip()
                return False, f"Cannot parse response: {raw}"
    except Exception as e:
        return False, str(e)


def get_presigned_url_for_file(server_base: str, s3_key: str, timeout=30):
    # wrapper naming clarity
    return get_presigned_url(server_base, s3_key, timeout=timeout)


def upload_files_individually(server_base: str, base_object_name: str, folder: Path, files: list, workers: int = 4, status_callback=None, abort_on_png_failure: bool = True):
    """Upload a list of files under `folder` to S3 by requesting a presigned URL per file from server_base.

    This implementation uploads files sequentially (deterministic order). If `abort_on_png_failure` is True,
    the function will stop and return immediately when any PNG file fails to upload (or fails presign).

    base_object_name is treated as the prefix/key under which files are uploaded. For each file, the S3 key will be
    base_object_name + '/' + relative_path.
    """
    results = []
    folder = Path(folder)

    # If workers > 1, upload in parallel but allow abort on PNG failure via abort_event
    if workers and workers > 1:
        abort_event = threading.Event()

        def worker_task(fp: Path, key: str):
            if abort_event.is_set():
                return (False, str(fp), 'aborted', {'aborted': True})
            # presign
            ok, url_or_msg = get_presigned_url_for_file(server_base, key)
            if not ok:
                msg = f"presign_failed: {url_or_msg}"
                status_callback and status_callback(f"Presign failed for {fp.name}: {url_or_msg}")
                # if PNG and abort requested, set abort and write log
                if abort_on_png_failure and fp.suffix.lower() == '.png':
                    abort_event.set()
                    try:
                        ts = int(time.time())
                        log_name = f"presign_error_{ts}.log"
                        log_path = folder / log_name
                        with open(log_path, 'w', encoding='utf-8') as lf:
                            lf.write(f"Presign failure for file: {fp}\n")
                            lf.write(f"S3 key attempted: {key}\n")
                            lf.write(f"Presign message: {url_or_msg}\n\n")
                            try:
                                lf.write("Raw presign response:\n")
                                lf.write(str(url_or_msg))
                            except Exception:
                                pass
                        return (False, str(fp), msg, {"log_path": str(log_path)})
                    except Exception:
                        return (False, str(fp), msg, {})
                return (False, str(fp), msg, {})

            # upload
            content_type, _ = mimetypes.guess_type(fp.name)
            if not content_type:
                content_type = 'application/octet-stream'
            ok2, msg2, details = upload_file_presigned(url_or_msg, fp, status_callback=status_callback, content_type=content_type)
            if not ok2 and abort_on_png_failure and fp.suffix.lower() == '.png':
                # write log
                try:
                    ts = int(time.time())
                    log_name = f"upload_error_{ts}.log"
                    log_path = folder / log_name
                    with open(log_path, 'w', encoding='utf-8') as lf:
                        lf.write(f"Upload failure for file: {fp}\n")
                        lf.write(f"S3 key attempted: {key}\n")
                        lf.write(f"Message: {msg2}\n\n")
                        lf.write("Details:\n")
                        lf.write(json.dumps(details, ensure_ascii=False, indent=2))
                    # request abort for others
                    abort_event.set()
                    return (False, str(fp), msg2, {"log_path": str(log_path)})
                except Exception:
                    abort_event.set()
                    return (False, str(fp), msg2, details or {})
            return (ok2, str(fp), msg2, details)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for f in files:
                fp = Path(f)
                rel = fp.relative_to(folder).as_posix()
                if base_object_name.endswith('/'):
                    key = base_object_name + rel
                else:
                    key = base_object_name + '/' + rel
                fut = ex.submit(worker_task, fp, key)
                futs[fut] = fp

            for fut in concurrent.futures.as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    # unexpected exception in worker
                    fp = futs.get(fut)
                    res = (False, str(fp), f'worker_exception: {e}', {'exception': str(e)})
                results.append(res)
                # if abort_event set, attempt to cancel remaining futures
                if abort_event.is_set():
                    for pending in futs:
                        if not pending.done():
                            try:
                                pending.cancel()
                            except Exception:
                                pass
                    break
        return results

    # sequential fallback (workers == 1 or unspecified)
    for f in files:
        fp = Path(f)
        rel = fp.relative_to(folder).as_posix()
        # construct s3 key
        if base_object_name.endswith('/'):
            key = base_object_name + rel
        else:
            key = base_object_name + '/' + rel
        # request presigned url
        ok, url_or_msg = get_presigned_url_for_file(server_base, key)
        if not ok:
            msg = f"presign_failed: {url_or_msg}"
            results.append((False, str(fp), msg))
            status_callback and status_callback(f"Presign failed for {fp.name}: {url_or_msg}")
            # If this is a PNG and abort_on_png_failure is requested, write a log and abort
            if abort_on_png_failure and fp.suffix.lower() == '.png':
                try:
                    ts = int(time.time())
                    log_name = f"presign_error_{ts}.log"
                    log_path = folder / log_name
                    with open(log_path, 'w', encoding='utf-8') as lf:
                        lf.write(f"Presign failure for file: {fp}\n")
                        lf.write(f"S3 key attempted: {key}\n")
                        lf.write(f"Presign message: {url_or_msg}\n\n")
                        try:
                            lf.write("Raw presign response:\n")
                            lf.write(str(url_or_msg))
                        except Exception:
                            pass
                    results.append((False, str(fp), msg, {"log_path": str(log_path)}))
                except Exception:
                    pass
                return results
            else:
                continue

        content_type, _ = mimetypes.guess_type(fp.name)
        if not content_type:
            content_type = 'application/octet-stream'
        ok2, msg2, details = upload_file_presigned(url_or_msg, fp, status_callback=status_callback, content_type=content_type)
        results.append((ok2, str(fp), msg2, details))
        # if upload failed and it's a PNG, abort further uploads
        if not ok2 and abort_on_png_failure and fp.suffix.lower() == '.png':
            status_callback and status_callback(f"PNG upload failed for {fp.name}: {msg2}")
            # write error log
            try:
                ts = int(time.time())
                log_name = f"upload_error_{ts}.log"
                log_path = folder / log_name
                with open(log_path, 'w', encoding='utf-8') as lf:
                    lf.write(f"Upload failure for file: {fp}\n")
                    lf.write(f"S3 key attempted: {key}\n")
                    lf.write(f"Message: {msg2}\n\n")
                    lf.write("Details:\n")
                    lf.write(json.dumps(details, ensure_ascii=False, indent=2))
                # include log_path in the returned details tuple by appending to results
                results.append((False, str(fp), msg2, {"log_path": str(log_path)}))
            except Exception:
                pass
            return results

    return results


def write_manifest(out_dir: Path, object_name: str, frame_count: int):
    manifest = {
        "object_name": object_name,
        "created_at": datetime.utcnow().isoformat() + 'Z',
        "frame_count": int(frame_count),
        "uuid": str(uuid.uuid4())
    }
    try:
        p = out_dir / 'manifest.json'
        p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def upload_file_presigned(url_or_obj, file_path: Path, status_callback=None, content_type: str = "application/octet-stream", max_retries: int = 3) -> tuple[bool, str, dict]:
    """Upload file_path to a pre-signed PUT URL with retries and timeouts.

    Returns (success, message). Uses requests if available, otherwise urllib.
    """
    # timeouts: (connect, read)
    connect_timeout = 10
    read_timeout = 120
    last_err = None
    attempts = 0
    for attempt in range(1, max_retries + 1):
        try:
            # url_or_obj can be either a string URL or a dict {'url':..., 'required_headers':{...}}
            if isinstance(url_or_obj, dict):
                url = url_or_obj.get('url')
                required_headers = url_or_obj.get('required_headers') or {}
            else:
                url = url_or_obj
                required_headers = {}

            attempts = attempt
            if _HAS_REQUESTS:
                with open(file_path, "rb") as fh:
                    # Build headers: if presign provided required headers, use them exactly; otherwise send Content-Type
                    headers = {}
                    # copy provided required headers (server may have signed on specific headers)
                    for k, v in (required_headers or {}).items():
                        headers[str(k)] = str(v)
                    # Only add Content-Type if the presign explicitly required it.
                    # Sending extra headers that were not included in the signature can cause SignatureDoesNotMatch.
                    hdr_keys_lower = {k.lower() for k in headers.keys()}
                    if any(k.lower() == 'content-type' for k in (required_headers or {}).keys()):
                        # use the required header value if provided, otherwise use guessed content_type
                        ct = None
                        for k, v in (required_headers or {}).items():
                            if k.lower() == 'content-type':
                                ct = v
                                break
                        headers['Content-Type'] = ct or content_type
                    status_callback and status_callback(f"Uploading {file_path.name} (attempt {attempt})...")
                    # stream upload with a reasonable timeout
                    res = requests.put(url, data=fh, headers=headers, timeout=(connect_timeout, read_timeout))
                    if 200 <= res.status_code < 300:
                        return True, f"OK {res.status_code}", {"status_code": res.status_code, "response_text": res.text, "sent_headers": headers, "attempts": attempt}
                    else:
                        # Non-2xx response
                        msg = f"Failed {res.status_code}: {res.text}"
                        status_callback and status_callback(msg)
                        last_err = {"status_code": res.status_code, "response_text": res.text, "sent_headers": headers, "attempts": attempt}
            else:
                # fallback to urllib
                from urllib.request import Request, urlopen
                with open(file_path, "rb") as fh:
                    data = fh.read()
                status_callback and status_callback(f"Uploading {file_path.name} (urllib, attempt {attempt})...")
                req = Request(url, data=data, method="PUT")
                # apply required headers if present
                for k, v in (required_headers or {}).items():
                    req.add_header(k, v)
                # Only add Content-Type if presign explicitly required it.
                if any(k.lower() == 'content-type' for k in (required_headers or {}).keys()):
                    ct = None
                    for k, v in (required_headers or {}).items():
                        if k.lower() == 'content-type':
                            ct = v
                            break
                    req.add_header("Content-Type", ct or content_type)
                with urlopen(req, timeout=read_timeout) as resp:
                    try:
                        body = resp.read().decode('utf-8', errors='replace')
                    except Exception:
                        body = '<unreadable>'
                    return True, f"OK {resp.status}", {"status_code": resp.status, "response_text": body, "sent_headers": dict((k, v) for k, v in (required_headers or {}).items()), "attempts": attempt}
        except Exception as e:
            status_callback and status_callback(f"Upload error (attempt {attempt}): {e}")
            last_err = {"exception": str(e), "attempt": attempt}
            # short backoff
            time.sleep(min(2 ** attempt, 8))
            continue
    # final failure
    details = last_err if isinstance(last_err, dict) else {"error": str(last_err), "attempts": attempts}
    return False, str(details), details


def build_gui():
    # --- new UI using ttk and frames for responsive layout ---
    try:
        from tkinter import ttk
    except Exception:
        import tkinter.ttk as ttk

    root = Tk()
    root.title("RealSense Recorder - Upload to S3")
    root.geometry('880x320')
    root.minsize(700, 240)

    cfg = load_config()

    out_var = StringVar(value=str(cfg.get('last_output', str(Path("output") / f"realsense_record_{int(time.time())}"))))
    url_var = StringVar(value=cfg.get('last_presigned_url', ''))
    server_var = StringVar(value=cfg.get('server_base', 'http://127.0.0.1:5000'))
    object_var = StringVar(value=cfg.get('last_object_name', 'realsense_package'))
    status_var = StringVar(value="Idle")
    from tkinter import BooleanVar
    zip_var = BooleanVar(value=True)

    recorder = {"obj": None}

    # Top frame: output folder
    top = ttk.Frame(root, padding=(8,6))
    top.grid(row=0, column=0, sticky='ew')
    top.columnconfigure(1, weight=1)
    ttk.Label(top, text="Output folder:").grid(row=0, column=0, sticky='w')
    out_entry = ttk.Entry(top, textvariable=out_var)
    out_entry.grid(row=0, column=1, sticky='ew', padx=6)
    def browse():
        d = filedialog.askdirectory(initialdir=str(Path.cwd()))
        if d:
            out_var.set(d)
    ttk.Button(top, text="Browse", command=browse).grid(row=0, column=2)

    # Middle frame: presigned URL and controls
    mid = ttk.Frame(root, padding=(8,6))
    mid.grid(row=1, column=0, sticky='ew')
    mid.columnconfigure(1, weight=1)
    ttk.Label(mid, text="Presigned PUT URL:").grid(row=0, column=0, sticky='w')
    url_entry = ttk.Entry(mid, textvariable=url_var)
    url_entry.grid(row=0, column=1, sticky='ew', padx=6)
    # example hint for presigned URL
    Label(mid, text="예: https://bucket.s3.amazonaws.com/path/key?X-Amz-... (Presigned PUT URL)", fg='gray').grid(row=1, column=1, sticky='w', padx=6, pady=(2,0))
    ttk.Label(mid, text="Object name:").grid(row=2, column=0, sticky='w', pady=(8,0))
    obj_entry = ttk.Entry(mid, textvariable=object_var)
    obj_entry.grid(row=2, column=1, sticky='ew', padx=6, pady=(8,0))
    Label(mid, text="예: videos/session1 (업로드될 S3 키의 prefix)", fg='gray').grid(row=3, column=1, sticky='w', padx=6, pady=(2,0))
    # Option: package everything into a single ZIP and upload (recommended for session atomicity)
    ttk.Checkbutton(mid, text="Package as ZIP (recommended)", variable=zip_var).grid(row=4, column=1, sticky='w', padx=6, pady=(6,0))
    # Server field and Get Presigned button to the right
    right = ttk.Frame(mid)
    right.grid(row=0, column=2, rowspan=2, sticky='ne', padx=(8,0))
    ttk.Label(right, text="Server base:").grid(row=0, column=0, sticky='w')
    server_entry = ttk.Entry(right, textvariable=server_var, width=28)
    server_entry.grid(row=1, column=0, sticky='ew', pady=(4,6))
    Label(right, text="예: http://98.84.179.212:5000", fg='gray').grid(row=2, column=0, sticky='w')
    def get_presigned_button():
        sb = server_var.get().strip()
        obj = object_var.get().strip()
        if not sb or not obj:
            status_var.set('서버 주소와 객체명을 입력하세요')
            return
        status_var.set('Requesting presigned URL...')
        root.update_idletasks()
        success, data = get_presigned_url(sb, obj)
        if success:
            # data may be a string URL or a dict {url, required_headers}
            if isinstance(data, dict):
                url_var.set(data.get('url'))
                # Do NOT persist the full presigned URL to disk. It is temporary and may
                # become invalid; storing it causes stale uploads when server IPs or
                # presigns change. Only keep server_base and object name.
                # cfg['last_presigned_url_details'] = data  # intentionally not saved
                # cfg['last_presigned_url'] = data.get('url')  # intentionally not saved
            else:
                url_var.set(data)
                # do not persist presigned URL
            status_var.set('Presigned URL received')
            # persist only server and object convenience values
            cfg['server_base'] = sb
            cfg['last_object_name'] = obj
            save_config(cfg)
        else:
            status_var.set('Presigned request failed: ' + str(data))
    ttk.Button(right, text="Get Presigned URL", command=get_presigned_button).grid(row=2, column=0, pady=(0,4))

    # Control frame: start/stop/upload
    ctrl = ttk.Frame(root, padding=(8,6))
    ctrl.grid(row=2, column=0, sticky='ew')
    ctrl.columnconfigure((0,1,2), weight=1)

    def start_recording():
        try:
            import pyrealsense2 as _rs
        except Exception:
            msg = (
                "RealSense runtime (pyrealsense2)가 설치되어 있지 않습니다.\n"
                "설치 링크: https://github.com/IntelRealSense/librealsense/releases\n"
                "설치 후 재시작하세요."
            )
            messagebox.showerror("RealSense runtime 필요", msg)
            return
        outp = Path(out_var.get())
        outp.mkdir(parents=True, exist_ok=True)
        rec = RealsenseRecorder(outp)
        recorder["obj"] = rec
        status_var.set("Starting RealSense and recording...")
        start_btn.config(state="disabled")
        stop_btn.config(state="normal")
        threading.Thread(target=_start_thread, args=(rec,), daemon=True).start()

    def _start_thread(rec):
        try:
            rec.start()
            status_var.set("Recording... (press Stop when finished)")
        except Exception as e:
            status_var.set(f"Start error: {e}")
            start_btn.config(state="normal")
            stop_btn.config(state="disabled")

    def stop_recording():
        rec = recorder.get("obj")
        if rec:
            status_var.set("Stopping...")
            rec.stop()
            status_var.set("Stopped")
            cfg['last_output'] = out_entry.get()
            save_config(cfg)
        start_btn.config(state="normal")
        stop_btn.config(state="disabled")

    def package_and_upload():
        # Run upload in background to avoid UI freeze
        def _worker():
            outp = Path(out_var.get())
            if not outp.exists():
                status_var.set("Output folder does not exist")
                # re-enable buttons
                start_btn.config(state='normal')
                stop_btn.config(state='disabled')
                upload_btn.config(state='normal')
                return
            color_dir = outp / 'color'
            frame_count = 0
            if color_dir.exists():
                frame_count = len(list(color_dir.glob('*.png')))
            write_manifest(outp, object_var.get().strip() or 'realsense_package', frame_count)
            # collect files
            files = []
            m = outp / 'manifest.json'
            if m.exists():
                files.append(m)
            for p in outp.glob('**/*'):
                if p.is_file() and p != m:
                    rel = p.relative_to(outp)
                    if rel.parts and rel.parts[0] in ('color', 'depth'):
                        files.append(p)
            if not files:
                status_var.set('업로드할 파일이 없습니다')
                # re-enable buttons
                start_btn.config(state='normal')
                stop_btn.config(state='disabled')
                upload_btn.config(state='normal')
                return
            sb = server_var.get().strip()
            if not sb:
                status_var.set('Server base를 입력하세요 (http://host:port)')
                # re-enable buttons
                start_btn.config(state='normal')
                stop_btn.config(state='disabled')
                upload_btn.config(state='normal')
                return
            base_obj = object_var.get().strip() or f"realsense_package_{int(time.time())}"
            # If zip_var is set, package into a single ZIP and upload that instead (recommended)
            if zip_var.get():
                try:
                    status_var.set('Creating ZIP package...')
                    root.update_idletasks()
                    zip_name = f"{base_obj.replace('/', '_')}.zip"
                    zip_path = outp.parent / zip_name
                    # create zip in parent of output to avoid including parent path in archive
                    zip_dir(outp, zip_path)
                    status_var.set('Requesting presigned URL for ZIP...')
                    root.update_idletasks()
                    ok, pres = get_presigned_url_for_file(sb, f"{base_obj}.zip")
                    if not ok:
                        # write log
                        ts = int(time.time())
                        log_path = outp / f"presign_error_{ts}.log"
                        with open(log_path, 'w', encoding='utf-8') as lf:
                            lf.write(f"Presign failure for ZIP: {zip_path}\n")
                            lf.write(f"Message: {pres}\n")
                        show_native_error('Presign failed', f'Presign for ZIP failed\n\n{pres}\n\nLog: {log_path}')
                        status_var.set('Presign failed for ZIP')
                        start_btn.config(state='normal')
                        stop_btn.config(state='disabled')
                        upload_btn.config(state='normal')
                        return
                    # upload zip
                    status_var.set('Uploading ZIP...')
                    root.update_idletasks()
                    # pres may be dict or url
                    url_obj = pres
                    ok2, msg2, details = upload_file_presigned(url_obj, zip_path, status_callback=lambda s: status_var.set(s), content_type='application/zip')
                    if not ok2:
                        ts = int(time.time())
                        log_path = outp / f"upload_error_{ts}.log"
                        with open(log_path, 'w', encoding='utf-8') as lf:
                            lf.write(f"ZIP upload failed: {zip_path}\n")
                            lf.write(f"Message: {msg2}\n\n")
                            lf.write(json.dumps(details, ensure_ascii=False, indent=2))
                        show_native_error('ZIP upload failed', f'ZIP upload failed\n\n{msg2}\n\nLog: {log_path}')
                        status_var.set('ZIP upload failed')
                        start_btn.config(state='normal')
                        stop_btn.config(state='disabled')
                        upload_btn.config(state='normal')
                        return
                    else:
                        status_var.set('ZIP upload succeeded')
                        ok_count = 1
                        status_var.set(f'Upload finished: {ok_count}/1 success')
                        start_btn.config(state='normal')
                        stop_btn.config(state='disabled')
                        upload_btn.config(state='normal')
                        return
                except Exception as e:
                    status_var.set(f'ZIP packaging/upload error: {e}')
                    start_btn.config(state='normal')
                    stop_btn.config(state='disabled')
                    upload_btn.config(state='normal')
                    return

            status_var.set(f'Uploading {len(files)} files...')
            results = upload_files_individually(sb, base_obj, outp, files, workers=4, status_callback=lambda s: status_var.set(s))

            # If any PNG failed, show an error dialog and abort further processing.
            png_failure = None
            for item in results:
                # item is (ok, fp, msg[, details_dict])
                if len(item) >= 3:
                    ok = item[0]
                    fp = item[1]
                    msg = item[2]
                    details = item[3] if len(item) > 3 else None
                else:
                    continue
                if not ok and fp.lower().endswith('.png'):
                    png_failure = (fp, msg, details)
                    break
            if png_failure:
                fp, msg, details = png_failure
                # schedule dialog on main thread
                def _show_err():
                    # show a native Windows error box so the user can't miss it
                    longmsg = f"PNG upload failed for {fp}\n\n{msg}"
                    if details and isinstance(details, dict) and 'log_path' in details:
                        longmsg += f"\n\nError log saved to: {details['log_path']}"
                    show_native_error("Upload failed", longmsg)
                    status_var.set(f"Upload aborted: PNG failure {Path(fp).name}")
                    # re-enable buttons
                    start_btn.config(state='normal')
                    stop_btn.config(state='disabled')
                    upload_btn.config(state='normal')
                root.after(0, _show_err)
                return

            ok_count = sum(1 for r in results if r[0])
            status_var.set(f'Upload finished: {ok_count}/{len(results)} success')
            # re-enable buttons
            start_btn.config(state='normal')
            stop_btn.config(state='disabled')
            upload_btn.config(state='normal')

        # disable controls while uploading
        start_btn.config(state='disabled')
        stop_btn.config(state='disabled')
        upload_btn.config(state='disabled')
        threading.Thread(target=_worker, daemon=True).start()
        # persist server/object info
        cfg['server_base'] = server_var.get().strip()
        cfg['last_object_name'] = object_var.get().strip()
        save_config(cfg)

    start_btn = ttk.Button(ctrl, text="Start Recording", command=start_recording)
    start_btn.grid(row=0, column=0, sticky='ew', padx=6, pady=6)
    stop_btn = ttk.Button(ctrl, text="Stop Recording", command=stop_recording, state='disabled')
    stop_btn.grid(row=0, column=1, sticky='ew', padx=6, pady=6)
    upload_btn = ttk.Button(ctrl, text="Upload Files", command=package_and_upload)
    upload_btn.grid(row=0, column=2, sticky='ew', padx=6, pady=6)

    # Status frame
    status_frame = ttk.Frame(root, padding=(8,6))
    status_frame.grid(row=3, column=0, sticky='ew')
    status_frame.columnconfigure(0, weight=1)
    status_label = ttk.Label(status_frame, textvariable=status_var, anchor='w')
    status_label.grid(row=0, column=0, sticky='ew')

    def on_close():
        rec = recorder.get("obj")
        if rec:
            rec.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    build_gui()
