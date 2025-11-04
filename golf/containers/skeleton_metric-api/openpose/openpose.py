import os
import base64
import traceback
import tempfile
import subprocess
import json
from pathlib import Path
from typing import List

import numpy as np


def run_openpose_on_image(image_path, output_json_path, output_img_path=None):
    """Run the OpenPose binary on a single image (tries common flags).

    This mirrors the original implementation in `api_server.py`.
    Raises RuntimeError on failure.
    """
    openpose_bin = "/opt/openpose/build/examples/openpose/openpose.bin"
    num_gpu = os.environ.get('OPENPOSE_NUM_GPU', os.environ.get('NUM_GPU', '1'))
    num_gpu_start = os.environ.get('OPENPOSE_NUM_GPU_START', os.environ.get('NUM_GPU_START', '0'))
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    base_args = [
        openpose_bin,
        "--model_folder", "/opt/openpose/models",
        "--write_json", str(output_json_path),
        "--display", "0",
        "--render_pose", "0",
        "--net_resolution", "-1x208",
        "--output_resolution", "-1x-1",
        "--number_people_max", "1",
        "--model_pose", "COCO",
        "--disable_blending", "false",
        "--scale_number", "1",
        "--scale_gap", "0.4",
        "--render_threshold", "0.3",
        "--num_gpu", str(num_gpu),
        "--num_gpu_start", str(num_gpu_start)
    ]

    if output_img_path and os.environ.get('OPENPOSE_WRITE_IMAGES', '0') == '1':
        base_args += ["--write_images", str(os.path.dirname(output_img_path)), "--render_pose", "1"]

    cmds_to_try = [base_args + ["--image_path", str(image_path)], base_args + ["--image_dir", str(os.path.dirname(image_path))]]
    last_err = None
    for cmd in cmds_to_try:
        env = os.environ.copy()
        if cuda_visible is not None:
            env['CUDA_VISIBLE_DEVICES'] = cuda_visible
        import time
        t0 = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        dt = time.time() - t0
        print(f"[DEBUG] OpenPose command took {dt:.3f}s: {' '.join(cmd)}")
        if result.returncode == 0:
            return
        last_err = result.stderr.decode('utf-8', errors='replace')

    err_msg = f"OpenPose failed after trying flags. Attempts:\n"
    for i, c in enumerate(cmds_to_try):
        err_msg += f"Attempt {i+1}: {' '.join(c)}\n"
    err_msg += f"Last stderr:\n{last_err}\n"
    raise RuntimeError(err_msg)


def run_openpose_on_dir(image_dir, output_json_path, output_img_path=None):
    """Run OpenPose on a directory of images using --image_dir."""
    openpose_bin = "/opt/openpose/build/examples/openpose/openpose.bin"
    cmd = [
        openpose_bin,
        "--image_dir", str(image_dir),
        "--model_folder", "/opt/openpose/models",
        "--write_json", str(output_json_path),
        "--display", "0",
        "--render_pose", "1",
        "--net_resolution", "-1x208",
        "--output_resolution", "-1x-1",
        "--number_people_max", "1",
        "--model_pose", "COCO",
        "--disable_blending", "false",
        "--scale_number", "1",
        "--scale_gap", "0.4",
        "--render_threshold", "0.3"
    ]
    if output_img_path:
        cmd += ["--write_images", str(os.path.dirname(output_img_path))]
    env = os.environ.copy()
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    import time
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    dt = time.time() - t0
    try:
        count = len(list(Path(image_dir).glob('*')))
    except Exception:
        count = 0
    print(f"[DEBUG] run_openpose_on_dir took {dt:.3f}s for {count} images")
    if result.returncode != 0:
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f"OpenPose dir invocation failed. Command: {' '.join(cmd)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n")


def _sanitize_person_list(person):
    for idx, kp in enumerate(person):
        try:
            x = kp[0] if kp[0] is not None else 0.0
            y = kp[1] if kp[1] is not None else 0.0
            c = kp[2] if kp[2] is not None else 0.0
            if not np.isfinite(x):
                x = 0.0
            if not np.isfinite(y):
                y = 0.0
            if not np.isfinite(c):
                c = 0.0
        except Exception:
            x, y, c = 0.0, 0.0, 0.0
        person[idx] = [float(x), float(y), float(c)]
    return person



def run_openpose_on_video(video_path, output_json_path, output_img_path=None):
    """Run OpenPose on a video file using --video."""
    openpose_bin = "/opt/openpose/build/examples/openpose/openpose.bin"
    cmd = [
        openpose_bin,
        "--video", str(video_path),
        "--model_folder", "/opt/openpose/models",
        "--write_json", str(output_json_path),
        "--display", "0",
        "--render_pose", "1",
        "--net_resolution", "-1x208",
        "--output_resolution", "-1x-1",
        "--number_people_max", "1",
        "--model_pose", "COCO",
        "--disable_blending", "false",
        "--scale_number", "1",
        "--scale_gap", "0.4",
        "--render_threshold", "0.3"
    ]
    if output_img_path:
        cmd += ["--write_images", str(os.path.dirname(output_img_path))]
    env = os.environ.copy()
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    import time
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    dt = time.time() - t0
    print(f"[DEBUG] run_openpose_on_video took {dt:.3f}s")
    if result.returncode != 0:
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f"OpenPose video invocation failed. Command: {' '.join(cmd)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n")


def OpenPoseProcessVideo(base64_video: str) -> List[List[List[List[float]]]]:
    """Process a base64-encoded MP4 video. Returns people-per-frame.

    Tries to run OpenPose with --video. If that fails, extracts frames and falls back to image-sequence processing.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, 'input.mp4')
            with open(video_path, 'wb') as vf:
                vf.write(base64.b64decode(base64_video))

            output_json_dir = os.path.join(tmpdir, 'json')
            os.makedirs(output_json_dir, exist_ok=True)
            output_img_dir = os.path.join(tmpdir, 'img')
            os.makedirs(output_img_dir, exist_ok=True)

            try:
                run_openpose_on_video(video_path, output_json_dir, output_img_dir)
            except Exception:
                traceback.print_exc()
                # Fallback: extract frames and use existing sequence processor
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    frame_idx = 0
                    frames_dir = os.path.join(tmpdir, 'fallback_frames')
                    os.makedirs(frames_dir, exist_ok=True)
                    success, frame = cap.read()
                    while success:
                        out_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(out_path, frame)
                        frame_idx += 1
                        success, frame = cap.read()
                    cap.release()
                    # Now run OpenPose on the frames directory
                    run_openpose_on_dir(frames_dir, output_json_dir, output_img_dir)
                except Exception:
                    traceback.print_exc()
                    # As a last resort, return empty list
                    return []

            # Parse JSON outputs (sorted by filename)
            json_files = sorted([f for f in os.listdir(output_json_dir) if f.endswith('.json')])
            result_by_frame = []
            for jf in json_files:
                path = os.path.join(output_json_dir, jf)
                with open(path, 'r', encoding='utf-8') as f:
                    jdata = json.load(f)
                raw_people = jdata.get('people', [])
                ppl = []
                for p in raw_people:
                    if 'pose_keypoints_2d' in p:
                        kps = p['pose_keypoints_2d']
                        person = [kps[idx:idx+3] for idx in range(0, len(kps), 3)]
                        person = _sanitize_person_list(person)
                        ppl.append(person)
                result_by_frame.append(ppl)
            return result_by_frame
    except Exception:
        traceback.print_exc()
        return []
