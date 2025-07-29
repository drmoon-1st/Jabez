import os
import subprocess
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor

# OpenPose 및 기타 설정
OPENPOSE_EXE  = Path(r"C:/openpose/openpose/bin/OpenPoseDemo.exe")
OPENPOSE_ROOT = OPENPOSE_EXE.parent.parent
PAD_RATIO     = 0.10

KP = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist",
      "MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye",
      "REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"]
COLS = [f"{n}_{a}" for n in KP for a in ("x","y","c")]

def run_openpose(video: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(OPENPOSE_EXE),
           "--video", str(video),
           "--write_json", str(out_dir),
           "--display", "0", "--render_pose", "0",
           "--number_people_max", "1",
           "--model_folder", str(OPENPOSE_ROOT / "models")]
    subprocess.run(cmd, check=True, cwd=OPENPOSE_ROOT)

def main_person_boxes(json_dir: Path):
    centers, boxes = [], []
    for jf in sorted(json_dir.glob("*.json")):
        data = json.load(open(jf))
        people = data.get("people")
        if not people:
            continue
        kps = np.array(people[0]["pose_keypoints_2d"]).reshape(-1, 3)
        if kps[8, 2] < 0.10:
            continue
        cx, cy = kps[8, :2]
        valid = kps[:, 2] > 0.05
        xs, ys = kps[valid, 0], kps[valid, 1]
        centers.append([cx, cy])
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    if not centers:
        return []
    centers = np.array(centers)
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
    w, h    = x2 - x1, y2 - y1
    pad_w   = w * PAD_RATIO
    pad_h   = h * PAD_RATIO
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
    cmd = ["ffmpeg", "-y", "-i", str(src),
           "-filter:v", f"crop={w}:{h}:{x}:{y}",
           "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)

def json_dir_to_csv(json_dir: Path, csv_path: Path):
    rows = []
    for jf in sorted(json_dir.glob("*.json")):
        data = json.load(open(jf))
        people = data.get("people")
        if not people: 
            rows.append([np.nan] * len(COLS))
        else:
            kps = np.array(people[0]["pose_keypoints_2d"]).reshape(-1, 3)
            # 정규화 없이 원본 좌표 저장
            rows.append(kps.flatten())
    pd.DataFrame(rows, columns=COLS).to_csv(csv_path, index=False)

def preprocess_all(root_dir: Path):
    VIDEO_DIR      = root_dir / "video"
    CROP_VIDEO_DIR = root_dir / "crop_video"
    CROP_KP_DIR    = root_dir / "crop_keypoint"
    TMP_JSON_DIR   = root_dir / "_tmp_json"
    for d in [CROP_VIDEO_DIR, CROP_KP_DIR, TMP_JSON_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for vid in tqdm(sorted(VIDEO_DIR.glob("*.mp4")), desc=f"{root_dir.parent.name}/{root_dir.name}"):
        name = vid.stem
        crop_csv  = CROP_KP_DIR / f"{name}_crop.csv"
        crop_mp4  = CROP_VIDEO_DIR / f"{name}_crop.mp4"
        try:
            raw_dir = TMP_JSON_DIR / f"raw_{name}"
            run_openpose(vid, raw_dir)
            boxes = main_person_boxes(raw_dir)
            if not boxes:
                print(f"⚠️  No valid person in {vid.name}")
                continue
            bbox = union_box(boxes)
            crop_video(vid, crop_mp4, bbox)
            crop_dir = TMP_JSON_DIR / f"crop_{name}"
            run_openpose(crop_mp4, crop_dir)
            json_dir_to_csv(crop_dir, crop_csv)
        except Exception as e:
            print(f"❌ Error processing {vid.name}: {e}")
    shutil.rmtree(TMP_JSON_DIR)

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    other_folders = [
        Path(r"D:\golfDataset\dataset\train\false"),
        Path(r"D:\golfDataset\dataset\test\balanced_true"),
        Path(r"D:\golfDataset\dataset\test\false"),
    ]
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(preprocess_all, other_folders))