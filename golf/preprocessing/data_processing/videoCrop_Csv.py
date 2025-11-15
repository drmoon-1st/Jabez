# ===================== USER PARAMETERS =====================
# 경로 및 병렬 처리 설정 (여기만 수정하면 됨)
DATASET_BASE_PATH = r"D:/golfDataset/dataset/test"  # mp4, jpg, json, crop 등 모든 결과가 저장될 루트
OPENPOSE_EXE = r"C:/openpose/bin/OpenPoseDemo.exe"  # OpenPose 실행파일 경로
MAX_WORKERS = 2  # 동시 CPU 작업 수
USE_GPU = True   # OpenPose GPU 사용 여부
# ==========================================================


import os
import subprocess
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
# 로그 파일 경로 (Path import 이후 정의)
ERROR_LOG_PATH = Path().resolve() / "videoCrop_Csv_error.log"

import os
import subprocess
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# OpenPose 설정
OPENPOSE_EXE = Path(OPENPOSE_EXE)
OPENPOSE_ROOT = OPENPOSE_EXE.parent.parent
OPENPOSE_MODEL = OPENPOSE_ROOT / "models" / "pose" / "coco" / "pose_iter_440000.caffemodel"
PAD_RATIO = 0.10
KP = [
    "Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist",
    "MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye",
    "REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"
]
COLS = [f"{n}_{a}" for n in KP for a in ("x","y","c")]

def run_openpose(video: Path, out_dir: Path, gpu_id=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(OPENPOSE_EXE),
        "--video", str(video),
        "--write_json", str(out_dir),
        "--display", "0", "--render_pose", "0",
        "--number_people_max", "1",
        "--model_folder", str(OPENPOSE_ROOT / "models"),
        "--net_resolution", "-1x368",
        "--model_pose", "COCO",
        "--disable_blending"
    ]
    subprocess.run(cmd, check=True, cwd=OPENPOSE_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main_person_boxes(json_dir: Path):
    centers, boxes = [], []
    for jf in sorted(json_dir.glob("*.json")):
        data = json.load(open(jf))
        people = data.get("people")
        if not people:
            continue
        kps = np.array(people[0]["pose_keypoints_2d"]).reshape(-1, 3)
        if kps[8, 2] < 0.10:  # MidHip confidence
            continue
        cx, cy = kps[8, :2]
        valid = kps[:, 2] > 0.05
        xs, ys = kps[valid, 0], kps[valid, 1]
        centers.append([cx, cy])
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    if not centers:
        return []
    centers = np.array(centers)
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=100, min_samples=5).fit_predict(centers)
    if (labels != -1).any():
        main_label = np.bincount(labels[labels != -1]).argmax()
    else:
        main_label = 0
    return [boxes[i] for i, lb in enumerate(labels) if lb == main_label]

def union_box(box_list):
    arr = np.array(box_list)
    x1, y1 = arr[:, :2].min(0)
    x2, y2 = arr[:, 2:].max(0)
    w, h = x2 - x1, y2 - y1
    pad_w = w * PAD_RATIO
    pad_h = h * PAD_RATIO
    return int(x1 - pad_w), int(y1 - pad_h), int(w + 2 * pad_w), int(h + 2 * pad_h)

def crop_video(src: Path, dst: Path, bbox):
    x, y, w, h = bbox
    import cv2
    cap = cv2.VideoCapture(str(src))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > orig_w: w = orig_w - x
    if y + h > orig_h: h = orig_h - y
    if w <= 0 or h <= 0:
        raise ValueError(f"❌ Invalid crop size: {(w, h)} for video {src.name}")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter:v", f"crop={w}:{h}:{x}:{y}",
        "-pix_fmt", "yuv420p", str(dst)
    ]
    subprocess.run(cmd, check=True)

def normalize_pose(kps):
    if np.isnan(kps).all():
        return kps
    center = kps[8, :2]  # MidHip
    kps[:, :2] -= center
    std = np.std(kps[:, :2])
    if std > 1e-5:
        kps[:, :2] /= std
    return kps

def json_dir_to_csv(json_dir: Path, csv_path: Path):
    rows = []
    for jf in sorted(json_dir.glob("*.json")):
        data = json.load(open(jf))
        people = data.get("people")
        if not people:
            rows.append([np.nan] * len(COLS))
        else:
            kps = np.array(people[0]["pose_keypoints_2d"]).reshape(-1, 3)
            kps = normalize_pose(kps)
            rows.append(kps.flatten())
    pd.DataFrame(rows, columns=COLS).to_csv(csv_path, index=False)

def process_one_mp4(mp4_path: Path, gpu_id=0):
    root = mp4_path.parent.parent.parent  # .../video/label/source/xxx.mp4
    name = mp4_path.stem
    crop_dir = root / "crop_keypoint"
    crop_dir.mkdir(exist_ok=True)
    crop_csv = crop_dir / f"{name}_crop.csv"
    crop_mp4 = root / "crop_video" / f"{name}_crop.mp4"
    tmp_json_dir = root / "_tmp_json" / f"raw_{name}"
    crop_json_dir = root / "_tmp_json" / f"crop_{name}"
    for d in [crop_mp4.parent, crop_dir, tmp_json_dir.parent]:
        d.mkdir(parents=True, exist_ok=True)
    # 1차 OpenPose
    run_openpose(mp4_path, tmp_json_dir, gpu_id)
    boxes = main_person_boxes(tmp_json_dir)
    if not boxes:
        raise RuntimeError(f"No valid person in {mp4_path.name}")
    bbox = union_box(boxes)
    crop_video(mp4_path, crop_mp4, bbox)
    # 2차 OpenPose (crop 영상)
    run_openpose(crop_mp4, crop_json_dir, gpu_id)
    json_dir_to_csv(crop_json_dir, crop_csv)
    # 모든 임시 json 디렉토리(_tmp_json) 삭제
    if tmp_json_dir.parent.exists():
        shutil.rmtree(tmp_json_dir.parent, ignore_errors=True)

def find_all_mp4s(base_path: Path):
    mp4s = []
    for tf in ["true", "false"]:
        tf_dir = base_path / tf
        if not tf_dir.exists():
            continue
        for eval_dir in tf_dir.iterdir():
            if not eval_dir.is_dir():
                continue
            video_root = eval_dir / "video"
            if not video_root.exists():
                continue
            for full_name_dir in video_root.iterdir():
                if not full_name_dir.is_dir():
                    continue
                for fname in os.listdir(full_name_dir):
                    if fname.lower().endswith('.mp4'):
                        mp4s.append(full_name_dir / fname)
    return mp4s

def main(dataset_base_path, max_workers=8, use_gpu=True):
    from tqdm import tqdm
    mp4s = find_all_mp4s(Path(dataset_base_path))
    print(f"MP4 found: {len(mp4s)}")
    if len(mp4s) > 0:
        print(mp4s[:5])
    # GPU 분배: CUDA_VISIBLE_DEVICES 환경변수에서 실제 GPU id 추출
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if use_gpu and cuda_visible:
        gpu_ids = [int(x) for x in cuda_visible.split(',') if x.strip().isdigit()]
        if not gpu_ids:
            gpu_ids = [0]
    else:
        gpu_ids = [0]
    def get_gpu_id(idx):
        return gpu_ids[idx % len(gpu_ids)] if use_gpu and gpu_ids else 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, mp4 in enumerate(mp4s):
            gpu_id = get_gpu_id(idx)
            futures.append(executor.submit(process_one_mp4, mp4, gpu_id))
        try:
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing mp4s"):
                _.result()
        except Exception as e:
            with open(ERROR_LOG_PATH, "a", encoding="utf-8") as logf:
                logf.write(f"{e}\n")
            print(f"Error occurred. See log: {ERROR_LOG_PATH}")
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_base_path', type=str, default=DATASET_BASE_PATH, help='Root path of the dataset')
    parser.add_argument('--max_workers', type=int, default=MAX_WORKERS, help='Number of parallel workers')
    parser.add_argument('--use_gpu', action='store_true', default=USE_GPU, help='Use GPU for OpenPose if available')
    args = parser.parse_args()
    main(args.dataset_base_path, args.max_workers, args.use_gpu)