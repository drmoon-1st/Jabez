# rs_realsense_openpose_record2csv.py
"""
Intel RealSense -> RGB/Depth 녹화 -> OpenPose(1회) -> 2D+3D CSV 생성

사용 예)
1) 시간 지정 녹화 (예: 5초)
   python rs_realsense_openpose_record2csv.py --output output --duration 5

2) 인터랙티브 녹화 (RGB 미리보기 창에서 'q'로 시작/종료)
   python rs_realsense_openpose_record2csv.py --output output --interactive
"""

import os
import json
import time
import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import pyrealsense2 as rs

# ================= 사용자 환경 설정 =================
OPENPOSE_EXE  = Path(r"C:/openpose/openpose/bin/OpenPoseDemo.exe")  # OpenPoseDemo.exe 경로
OPENPOSE_ROOT = OPENPOSE_EXE.parent.parent
MODEL_FOLDER  = OPENPOSE_ROOT / "models"                             # .../openpose/models
COCO_MODEL    = MODEL_FOLDER / "pose/coco/pose_iter_440000.caffemodel"
# RealSense 캡처 해상도/FPS
IMG_W, IMG_H, FPS = 640, 480, 30

# COCO17 타깃 컬럼 순서 (2D/3D 모두 이 순서를 따름)
KP_17 = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]
COLS_2D = [f"{n}_{a}" for n in KP_17 for a in ("x","y","c")]
COLS_3D = [f"{n}_{a}" for n in KP_17 for a in ("X3D","Y3D","Z3D")]

# OpenPose COCO 출력(18개, Neck 포함 가능) -> 위 KP_17 순서로 재배열
_IDX_MAP_18_TO_17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

# ================= 유틸 =================
def assert_openpose():
    assert OPENPOSE_EXE.exists(), f"OpenPoseDemo.exe 없음: {OPENPOSE_EXE}"
    assert COCO_MODEL.exists(), f"COCO 모델 없음: {COCO_MODEL}"

def robust_depth_from_patch(depth_m: np.ndarray, x: int, y: int, r: int = 2) -> float:
    H, W = depth_m.shape
    x0, x1 = max(0, x-r), min(W, x+r+1)
    y0, y1 = max(0, y-r), min(H, y+r+1)
    patch = depth_m[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(vals)) if vals.size else np.nan

def deproject_xyZ_to_XYZ(x: float, y: float, Z: float, intr: dict):
    if not (Z > 0 and np.isfinite(Z)):
        return (np.nan, np.nan, np.nan)
    X = (x - intr["cx"]) * Z / intr["fx"]
    Y = (y - intr["cy"]) * Z / intr["fy"]
    return (float(X), float(Y), float(Z))

# ================= 1) RealSense 캡처 =================
def capture_rgbd_to_dir(output_dir: Path, duration_sec: float = None, interactive: bool = False):
    """
    RealSense에서 RGB/Depth 캡처.
    - color/*.png, depth/*.npy 저장 (depth는 미터 float32)
    - intrinsics.json 저장
    - duration_sec 지정 or interactive 모드('q'로 시작/종료, ESC로 즉시 종료)
    """
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, FPS)
    align = rs.align(rs.stream.color)

    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        intr = None

        t0 = time.time()
        idx = 0  # 저장된 프레임 인덱스(녹화 시에만 증가)

        win = None
        is_recording = False
        if interactive:
            win = "RealSense RGB (press 'q' to START/STOP, ESC to exit)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        pbar = tqdm(desc="Capturing RGB-D", unit="fr")
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            if intr is None:
                ci = color.profile.as_video_stream_profile().intrinsics
                intr = {
                    "width": ci.width, "height": ci.height,
                    "fx": ci.fx, "fy": ci.fy, "cx": ci.ppx, "cy": ci.ppy,
                    "coeffs": list(ci.coeffs),
                }
                meta = {"fps": FPS, "width": IMG_W, "height": IMG_H, "depth_scale": float(depth_scale)}
                with open(output_dir / "intrinsics.json", "w", encoding="utf-8") as f:
                    json.dump({"color_intrinsics": intr, "meta": meta}, f, indent=2)

            color_img = np.asanyarray(color.get_data())
            depth_raw = np.asanyarray(depth.get_data())
            depth_m = (depth_raw.astype(np.float32) * depth_scale)

            # 인터랙티브: 미리보기 + 키 처리
            if interactive and win is not None:
                disp = color_img.copy()
                if is_recording:
                    cv2.putText(disp, f"REC [{idx}]", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(disp, "Press 'q' to STOP", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2)
                else:
                    cv2.putText(disp, "Press 'q' to START", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2)
                    cv2.putText(disp, "ESC to exit", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(win, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    # 즉시 종료 (녹화 중이든 아니든)
                    break
                elif key == ord('q'):
                    # 토글
                    is_recording = not is_recording
                    if not is_recording:
                        # STOP → 루프 종료하고 후처리 진행
                        break

            # 저장은 "녹화 중"에만 수행 (duration 모드는 항상 저장)
            if (interactive and is_recording) or ((not interactive) and (duration_sec is not None)):
                cv2.imwrite(str(color_dir / f"{idx:06d}.png"), color_img)
                np.save(depth_dir / f"{idx:06d}.npy", depth_m)
                idx += 1
                pbar.update(1)

            # duration 모드: 시간이 다 되면 종료
            if (not interactive) and (duration_sec is not None) and ((time.time() - t0) >= duration_sec):
                break

        pbar.close()

    finally:
        pipeline.stop()
        if interactive:
            cv2.destroyAllWindows()

    return color_dir, depth_dir, output_dir / "intrinsics.json"

# ================= 2) OpenPose 1회 실행 =================
def run_openpose_on_image_dir(image_dir: Path, json_out_dir: Path):
    img_dir_abs  = Path(image_dir).resolve()
    json_dir_abs = Path(json_out_dir).resolve()
    json_dir_abs.mkdir(parents=True, exist_ok=True)

    if not img_dir_abs.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir_abs}")

    cmd = [
        str(OPENPOSE_EXE),
        "--image_dir", str(img_dir_abs),
        "--write_json", str(json_dir_abs),
        "--display", "0", "--render_pose", "0",
        "--number_people_max", "1",
        "--model_folder", str(MODEL_FOLDER.resolve()),
        "--model_pose", "COCO",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=OPENPOSE_ROOT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"OpenPose 실패\nstdout:\n{res.stdout}\n\nstderr:\n{res.stderr}")

# ================= 3) JSON+Depth -> 2D+3D CSV =================
def json_depth_to_2d3d_csv(json_dir: Path, depth_dir: Path, intrinsics_json: Path, out_csv: Path):
    with open(intrinsics_json, "r", encoding="utf-8") as f:
        intr_full = json.load(f)
    intr = intr_full["color_intrinsics"]

    rows = []
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No OpenPose JSON in {json_dir}")

    for idx, jf in enumerate(tqdm(json_files, desc="2D+Depth -> 3D", unit="fr")):
        depth_path = depth_dir / f"{idx:06d}.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth not found for frame {idx:06d}.npy in {depth_dir}")
        depth_m = np.load(depth_path)  # (H,W) meters

        data = json.load(open(jf, "r"))
        people = data.get("people", [])
        if not people:
            rows.append([np.nan] * (len(COLS_2D) + len(COLS_3D)))
            continue

        kps = np.array(people[0]["pose_keypoints_2d"]).reshape(-1, 3)
        kps_17 = kps[_IDX_MAP_18_TO_17, :] if kps.shape[0] >= 18 else kps

        row_2d, row_3d = [], []
        for (x, y, c) in kps_17:
            row_2d.extend([float(x), float(y), float(c)])
            xi, yi = int(round(x)), int(round(y))
            Z = robust_depth_from_patch(depth_m, xi, yi, r=2)
            X, Y, Zm = deproject_xyZ_to_XYZ(x, y, Z, intr)
            row_3d.extend([X, Y, Zm])

        rows.append(row_2d + row_3d)

    pd.DataFrame(rows, columns=(COLS_2D + COLS_3D)).to_csv(out_csv, index=False)
    print(f"[SAVE] 2D+3D CSV -> {out_csv}")

# ================= CLI =================
def main():
    parser = argparse.ArgumentParser(description="RealSense 녹화 → OpenPose(1회) → 2D+3D CSV")
    parser.add_argument("--output", required=True, type=str, help="출력 폴더")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--duration", type=float, help="녹화 시간(초)")
    group.add_argument("--interactive", action="store_true", help="미리보기 창에서 'q'로 시작/종료, ESC로 즉시 종료")

    args = parser.parse_args()
    assert_openpose()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 녹화
    color_dir, depth_dir, intr_json = capture_rgbd_to_dir(
        out_dir, duration_sec=args.duration, interactive=args.interactive
    )

    # 저장된 프레임이 하나도 없으면 바로 종료
    if not any(color_dir.glob("*.png")):
        print("[WARN] 녹화된 프레임이 없습니다. 종료합니다.")
        return

    # 2) OpenPose(1회)
    json_dir = out_dir / "openpose_json"
    run_openpose_on_image_dir(color_dir, json_dir)

    # 3) 2D+3D CSV
    out_csv = out_dir / "skeleton2d3d.csv"
    json_depth_to_2d3d_csv(json_dir, depth_dir, intr_json, out_csv)

if __name__ == "__main__":
    main()
