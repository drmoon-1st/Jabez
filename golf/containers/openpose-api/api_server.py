import os
import base64
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import cv2
from pathlib import Path
import tempfile
import subprocess
import json
from skeleton_interpolate import interpolate_sequence

app = FastAPI()

class OpenPoseRequest(BaseModel):
    img: Optional[str] = None
    imgs: Optional[List[str]] = None
    turbo_without_skeleton: Optional[bool] = True

# Helper: Run openpose binary and parse output
def run_openpose_on_image(image_path, output_json_path, output_img_path=None):
    openpose_bin = "/opt/openpose/build/examples/openpose/openpose.bin"
    # GPU control via env vars (optional)
    num_gpu = os.environ.get('OPENPOSE_NUM_GPU', os.environ.get('NUM_GPU', '1'))
    num_gpu_start = os.environ.get('OPENPOSE_NUM_GPU_START', os.environ.get('NUM_GPU_START', '0'))
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    base_args = [
        openpose_bin,
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
        "--render_threshold", "0.3",
        "--num_gpu", str(num_gpu),
        "--num_gpu_start", str(num_gpu_start)
    ]
    if output_img_path:
        base_args += ["--write_images", str(os.path.dirname(output_img_path))]

    # Try common image flags for different OpenPose builds: --image_path or --image_dir
    attempts = []
    cmds_to_try = [base_args + ["--image_path", str(image_path)], base_args + ["--image_dir", str(os.path.dirname(image_path))]]
    last_err = None
    for cmd in cmds_to_try:
        attempts.append(cmd)
        # run with optional CUDA_VISIBLE_DEVICES forwarded
        env = os.environ.copy()
        if cuda_visible is not None:
            env['CUDA_VISIBLE_DEVICES'] = cuda_visible
        import time
        t0 = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        dt = time.time() - t0
        # log duration for diagnostics
        print(f"[DEBUG] OpenPose command took {dt:.3f}s: {' '.join(cmd)}")
        if result.returncode == 0:
            return
        # collect stderr for diagnostics and try next
        last_err = result.stderr.decode('utf-8', errors='replace')
    # If we reach here, all attempts failed
    err_msg = f"OpenPose failed after trying flags. Attempts:\n"
    for i, c in enumerate(attempts):
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
    print(f"[DEBUG] run_openpose_on_dir took {dt:.3f}s for {len(list(Path(image_dir).glob('*')))} images")
    if result.returncode != 0:
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f"OpenPose dir invocation failed. Command: {' '.join(cmd)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n")

# Helper: Process base64 image with openpose binary
def OpenPoseImageProcessing(base64_image_string):
    try:
        # 1. Decode base64 to image file
        image_data = base64.b64decode(base64_image_string)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "input.jpg")
            with open(img_path, "wb") as f:
                f.write(image_data)
            # 2. Run openpose
            output_json_dir = os.path.join(tmpdir, "json")
            os.makedirs(output_json_dir, exist_ok=True)
            output_img_dir = os.path.join(tmpdir, "img")
            os.makedirs(output_img_dir, exist_ok=True)
            run_openpose_on_image(img_path, output_json_dir, output_img_dir)
            # 3. Parse output JSON (first file)
            json_files = [f for f in os.listdir(output_json_dir) if f.endswith('.json')]
            people = []
            if json_files:
                with open(os.path.join(output_json_dir, json_files[0]), "r", encoding="utf-8") as jf:
                    jdata = json.load(jf)
                # OpenPose JSON: {people: [{pose_keypoints_2d: [...]}]}
                raw_people = jdata.get("people", [])
                for p in raw_people:
                    if "pose_keypoints_2d" in p:
                        kps = p["pose_keypoints_2d"]
                        person = [kps[i:i+3] for i in range(0, len(kps), 3)]
                        # sanitize NaN/inf/None -> 0.0
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
                        people.append(person)
            # 4. Read rendered image and encode to base64
            img_files = [f for f in os.listdir(output_img_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if img_files:
                with open(os.path.join(output_img_dir, img_files[0]), "rb") as imf:
                    img_bytes = imf.read()
                output_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            else:
                output_image_base64 = ""
            return people, output_image_base64
    except Exception:
        traceback.print_exc()
        return [], ""


def OpenPoseProcessImageSequence(base64_images):
    """Process a list of base64 images by running OpenPose once on a temporary directory.

    Returns list of people per frame (each item is a list for a person or empty list).
    Falls back to per-image processing if the OpenPose binary doesn't support --image_dir.
    """
    # Write all images and attempt one-shot processing with run_openpose_on_dir
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = os.path.join(tmpdir, 'imgs')
            os.makedirs(img_dir, exist_ok=True)
            names = []
            for i, b64 in enumerate(base64_images):
                name = f"frame_{i:06d}.jpg"
                path = os.path.join(img_dir, name)
                with open(path, 'wb') as f:
                    f.write(base64.b64decode(b64))
                names.append(name)

            output_json_dir = os.path.join(tmpdir, 'json')
            os.makedirs(output_json_dir, exist_ok=True)
            output_img_dir = os.path.join(tmpdir, 'img')
            os.makedirs(output_img_dir, exist_ok=True)

            # Prefer directory-based invocation (single OpenPose run for all frames)
            try:
                run_openpose_on_dir(img_dir, output_json_dir, output_img_dir)
            except Exception:
                # If that fails (binary doesn't support --image_dir or other error),
                # fallback to per-image processing
                traceback.print_exc()
                out = []
                for b64 in base64_images:
                    ppl, _ = OpenPoseImageProcessing(b64)
                    out.append(ppl)
                return out

            # collect jsons and map to frames
            json_files = [f for f in os.listdir(output_json_dir) if f.endswith('.json')]
            result_by_frame = []
            for i, name in enumerate(names):
                base = os.path.splitext(name)[0]
                match = None
                for jf in json_files:
                    if jf.startswith(base):
                        match = jf
                        break
                if match:
                    with open(os.path.join(output_json_dir, match), 'r', encoding='utf-8') as jf:
                        jdata = json.load(jf)
                    raw_people = jdata.get('people', [])
                    ppl = []
                    for p in raw_people:
                        if 'pose_keypoints_2d' in p:
                            kps = p['pose_keypoints_2d']
                            person = [kps[idx:idx+3] for idx in range(0, len(kps), 3)]
                            # sanitize
                            for pi, kp in enumerate(person):
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
                                person[pi] = [float(x), float(y), float(c)]
                            ppl.append(person)
                    result_by_frame.append(ppl)
                else:
                    result_by_frame.append([])
            return result_by_frame
    except Exception:
        traceback.print_exc()
        # last resort fallback per-image
        out = []
        for b64 in base64_images:
            ppl, _ = OpenPoseImageProcessing(b64)
            out.append(ppl)
        return out

# Clustering / normalization code removed per request.
# This server now returns raw OpenPose keypoints only (people list and rendered image).

# --- FastAPI endpoint ---
@app.post("/openpose_predict")
async def openpose_predict(req: OpenPoseRequest):
    try:
        turbo_without_skeleton = req.turbo_without_skeleton
        # single image mode
        if req.img and not req.imgs:
            img_base64 = req.img
            pose_json, img_base64_rendered = OpenPoseImageProcessing(img_base64)
            people_list = pose_json or []
            response_payload = {
                'message': 'OK',
                'pose_id': None,
                'people': people_list,
                'pose_keypoints_2d': people_list[0] if people_list else [],
                'openposeimg': img_base64_rendered
            }
            if turbo_without_skeleton:
                response_payload['openposeimg'] = ""
        else:
            # sequence mode: batch-process frames by running OpenPose once on the image directory
            frames = req.imgs or []
            # batch process (writes images -> runs openpose on dir -> reads results)
            batch_results = OpenPoseProcessImageSequence(frames)
            # pick first person per frame (or empty list)
            sequence = [ (ppl[0] if (ppl and len(ppl) > 0) else []) for ppl in batch_results ]
            interpolated = interpolate_sequence(sequence, conf_thresh=0.0, method='linear', fill_method='none')
            response_payload = {
                'message': 'OK',
                'pose_id': None,
                'people_sequence': interpolated,
                'frame_count': len(interpolated)
            }
        return JSONResponse(content=response_payload)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 19030))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
