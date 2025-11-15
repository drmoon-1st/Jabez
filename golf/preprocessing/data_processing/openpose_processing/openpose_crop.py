import argparse
import json
import os
import sqlite3
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from skeleton_interpolate import interpolate_sequence

# Defaults copied from rs_realsense_openpose_record2csv.py
OPENPOSE_EXE_DEFAULT = Path(r"C:/openpose/bin/OpenPoseDemo.exe")
OPENPOSE_ROOT_DEFAULT = OPENPOSE_EXE_DEFAULT.parent.parent
MODEL_FOLDER_DEFAULT = OPENPOSE_ROOT_DEFAULT / "models"

# COCO17 target order (same as rs script)
KP_17 = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]
COLS_2D = [f"{n}_{a}" for n in KP_17 for a in ("x", "y", "c")]

# Map OpenPose COCO 18 -> 17 ordering used by rs script
_IDX_MAP_18_TO_17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]


DB_DEFAULT = Path(r"E:\data_processing\openpose_processing\processed.db")


def _open_index_db(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute(
        "CREATE TABLE IF NOT EXISTS openpose_videos (video_path TEXT PRIMARY KEY, status TEXT, output_csv TEXT, ts TEXT, error TEXT);"
    )
    return conn


def _is_video_done(conn: sqlite3.Connection, video_path: str) -> bool:
    cur = conn.cursor()
    cur.execute('SELECT status FROM openpose_videos WHERE video_path = ?', (video_path,))
    row = cur.fetchone()
    return bool(row and row[0] == 'done')


def _mark_video_done(conn: sqlite3.Connection, video_path: str, output_csv: str):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute('INSERT OR REPLACE INTO openpose_videos (video_path, status, output_csv, ts, error) VALUES (?, ?, ?, ?, ?)',
                (video_path, 'done', output_csv, ts, None))


def _mark_video_failed(conn: sqlite3.Connection, video_path: str, error_msg: str = None):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute('INSERT OR REPLACE INTO openpose_videos (video_path, status, output_csv, ts, error) VALUES (?, ?, ?, ?, ?)',
                (video_path, 'failed', None, ts, error_msg))


def find_video_crop_dirs(root: Path) -> List[Path]:
    out = []
    if not root.exists():
        return out
    for p in root.rglob('video_crop'):
        if p.is_dir():
            out.append(p)
    return out


def run_openpose_on_image_dir(image_dir: Path, json_out_dir: Path, openpose_exe: str, model_folder: Optional[str],
                              num_gpu: Optional[int] = None, num_gpu_start: Optional[int] = None, cuda_devices: Optional[str] = None):
    """Run OpenPose on an image directory and write JSON outputs.

    This is adapted from rs_realsense_openpose_record2csv.py's run_openpose_on_image_dir.
    """
    img_dir_abs = Path(image_dir).resolve()
    json_dir_abs = Path(json_out_dir).resolve()
    json_dir_abs.mkdir(parents=True, exist_ok=True)

    if not img_dir_abs.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir_abs}")

    openpose_exe = openpose_exe or OPENPOSE_EXE_DEFAULT
    model_folder = model_folder or MODEL_FOLDER_DEFAULT

    cmd = [
        str(openpose_exe),
        "--image_dir", str(img_dir_abs),
        "--write_json", str(json_dir_abs),
        "--display", "0", "--render_pose", "0",
        "--number_people_max", "1",
        "--model_folder", str(model_folder),
        "--model_pose", "COCO",
    ]
    if num_gpu is not None:
        cmd += ["--num_gpu", str(num_gpu)]
    if num_gpu_start is not None:
        cmd += ["--num_gpu_start", str(num_gpu_start)]

    # run with cwd at the OpenPose root
    cwd = OPENPOSE_ROOT_DEFAULT if OPENPOSE_ROOT_DEFAULT.exists() else None
    env = os.environ.copy()
    if cuda_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(cuda_devices)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"OpenPose failed\nstdout:\n{res.stdout}\n\nstderr:\n{res.stderr}")


def process_one_crop_dir(crop_dir: Path, conn: sqlite3.Connection = None, force: bool = False,
                         openpose_exe: str = None, model_folder: Optional[str] = None,
                         conf_thresh: float = 0.0, interp_fill: str = 'zero', interp_limit: Optional[int] = None,
                         num_gpu: Optional[int] = None, num_gpu_start: Optional[int] = None, cuda_devices: Optional[str] = None,
                         progress_bar: Optional[tqdm] = None):
    mp4s = sorted([p for p in crop_dir.glob('*.mp4')])
    if not mp4s:
        return []

    out_dir = crop_dir.parent / 'skeleton_crop'
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = []

    openpose_exe = openpose_exe or os.environ.get('OPENPOSE_BIN', 'OpenPoseDemo.exe')
    model_folder = model_folder or os.environ.get('OPENPOSE_MODEL_FOLDER', None)

    for mp4 in mp4s:
        target_csv = out_dir / (mp4.stem + '.csv')
        mp4_str = str(mp4.resolve())

        if conn is not None and not force and _is_video_done(conn, mp4_str):
            print(f"Skipping (done): {mp4.name}")
            produced.append(str(target_csv))
            if progress_bar is not None:
                progress_bar.update(1)
            continue

        try:
            print(f"Processing: {mp4}")
            with tempfile.TemporaryDirectory() as tmpdir:
                frames_dir = Path(tmpdir) / 'frames'
                frames_dir.mkdir(parents=True, exist_ok=True)

                # Extract frames
                cap = cv2.VideoCapture(str(mp4))
                idx = 0
                success, frame = cap.read()
                while success:
                    out_path = frames_dir / f"frame_{idx:06d}.png"
                    cv2.imwrite(str(out_path), frame)
                    idx += 1
                    success, frame = cap.read()
                cap.release()

                if idx == 0:
                    raise RuntimeError('No frames extracted')

                json_out = Path(tmpdir) / 'openpose_json'
                run_openpose_on_image_dir(frames_dir, json_out, openpose_exe, model_folder,
                                          num_gpu=num_gpu, num_gpu_start=num_gpu_start, cuda_devices=cuda_devices)

                # Parse JSON outputs into frames_keypoints (first person)
                json_files = sorted([p for p in json_out.glob('*.json')])
                frames_keypoints = []
                for jf in json_files:
                    data = json.load(open(jf, 'r', encoding='utf-8'))
                    people = data.get('people', [])
                    if not people:
                        frames_keypoints.append([])
                    else:
                        kps = np.array(people[0].get('pose_keypoints_2d', [])).reshape(-1, 3)
                        # remap 18->17 if needed
                        if kps.shape[0] >= 18:
                            kps_17 = kps[_IDX_MAP_18_TO_17, :]
                        else:
                            kps_17 = kps
                        person = [[float(x), float(y), float(c)] for (x, y, c) in kps_17]
                        frames_keypoints.append(person)

                # Interpolate per skeleton (frames_keypoints is list[frame][joint][x,y,c])
                interp = interpolate_sequence(frames_keypoints, conf_thresh=conf_thresh, method='linear', fill_method=interp_fill, limit=interp_limit)

                if not interp:
                    raise RuntimeError('Interpolated sequence empty')

                # Build DataFrame with columns COLS_2D
                n_joints = len(interp[0])
                cols = COLS_2D if len(COLS_2D) == n_joints * 3 else [f'x_{j}' if k%3==0 else ('y_'+str(j//3) if k%3==1 else 'c_'+str(j//3)) for j,k in enumerate(range(n_joints*3))]

                rows = []
                for frame in interp:
                    flat = []
                    for kp in frame:
                        flat.extend([kp[0], kp[1], kp[2]])
                    if len(flat) < n_joints * 3:
                        flat.extend([0.0] * (n_joints * 3 - len(flat)))
                    rows.append(flat)

                df = pd.DataFrame(rows, columns=cols[:n_joints*3])
                df.to_csv(target_csv, index=False)

            if conn is not None:
                _mark_video_done(conn, mp4_str, str(target_csv))

            produced.append(str(target_csv))
            print(f"Wrote CSV: {target_csv}")
            if progress_bar is not None:
                progress_bar.update(1)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Failed processing {mp4.name}: {e}\n{tb}")
            if conn is not None:
                _mark_video_failed(conn, mp4_str, str(e))
            if progress_bar is not None:
                progress_bar.update(1)

    return produced


def main(root: str, db_path: str = None, openpose_exe: str = None, model_folder: Optional[str] = None, force: bool = False,
         conf_thresh: float = 0.0, interp_fill: str = 'zero', interp_limit: Optional[int] = None,
         num_gpu: Optional[int] = None, num_gpu_start: Optional[int] = None, cuda_devices: Optional[str] = None,
         progress_bar: Optional[tqdm] = None):
    rootp = Path(root)
    conn = None
    if db_path:
        conn = _open_index_db(Path(db_path))
    else:
        conn = _open_index_db(DB_DEFAULT)

    crop_dirs = find_video_crop_dirs(rootp)
    print(f"Found {len(crop_dirs)} video_crop dirs under {root}")

    all_outputs = []
    for cd in crop_dirs:
        outs = process_one_crop_dir(cd, conn=conn, force=force, openpose_exe=openpose_exe, model_folder=model_folder,
                                    conf_thresh=conf_thresh, interp_fill=interp_fill, interp_limit=interp_limit,
                                    num_gpu=num_gpu, num_gpu_start=num_gpu_start, cuda_devices=cuda_devices,
                                    progress_bar=progress_bar)
        all_outputs.extend(outs)

    print(f"Completed. Wrote {len(all_outputs)} CSV files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=r'E:\golfDataset_dlc\dataset\train', help='Top-level dataset root to search for video_crop directories')
    parser.add_argument('--class', dest='cls', choices=['good','best','normal','worst','bad','test'], default=None, help='Optional class to process. One of: good,best,normal,worst,bad,test. good/best/normal/test -> <root>/true/<class>, worst/bad -> <root>/false/<class>')
    parser.add_argument('--db', type=str, default=str(DB_DEFAULT), help='Path to processed.db sqlite file')
    parser.add_argument('--openpose-exe', type=str, default=os.environ.get('OPENPOSE_BIN', str(OPENPOSE_EXE_DEFAULT)), help='OpenPose executable path')
    parser.add_argument('--model-folder', type=str, default=os.environ.get('OPENPOSE_MODEL_FOLDER', str(MODEL_FOLDER_DEFAULT)), help='OpenPose model folder')
    parser.add_argument('--force', action='store_true', help='Reprocess even if DB marks done')
    parser.add_argument('--conf-thresh', type=float, default=0.0, help='Confidence threshold for masking before interpolation')
    parser.add_argument('--interp-fill', type=str, default='zero', choices=['none','ffill','bfill','nearest','zero'], help='How to fill remaining NaNs after interpolation')
    parser.add_argument('--interp-limit', type=int, default=None, help='Max consecutive NaNs to interpolate (None = unlimited)')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f'Root does not exist: {root}')

    target_roots = []
    if args.cls:
        cand_direct = root / args.cls
        cand_under_true = root / 'true' / args.cls
        cand_under_false = root / 'false' / args.cls
        true_classes = {'good', 'best', 'normal', 'test'}
        false_classes = {'worst', 'bad'}

        if args.cls in false_classes:
            if cand_under_false.exists() and cand_under_false.is_dir():
                target_roots.append(cand_under_false)
            elif cand_direct.exists() and cand_direct.is_dir():
                target_roots.append(cand_direct)
            else:
                raise SystemExit(f'Requested class folder not found. Tried: {cand_under_false} and {cand_direct}')
        else:
            if cand_under_true.exists() and cand_under_true.is_dir():
                target_roots.append(cand_under_true)
            elif cand_direct.exists() and cand_direct.is_dir():
                target_roots.append(cand_direct)
            else:
                raise SystemExit(f'Requested class folder not found. Tried: {cand_under_true} and {cand_direct}')
    else:
        target_roots.append(root)

    # Default to GPU usage: use GPU 0
    DEFAULT_NUM_GPU = 1
    DEFAULT_NUM_GPU_START = 0
    DEFAULT_CUDA_DEVICES = '0'

    # Compute total number of mp4 files across all target roots so we can show a single overall progress bar
    total_mp4s = 0
    for tr in target_roots:
        trp = Path(tr)
        for vc in trp.rglob('video_crop'):
            total_mp4s += len(list(vc.glob('*.mp4')))

    # If user requested debug class 'test', do not write to the index DB
    db_to_use = None if args.cls == 'test' else args.db

    with tqdm(total=total_mp4s, desc='OpenPose all videos') as pbar:
        for tr in target_roots:
            main(str(tr), db_path=db_to_use, openpose_exe=args.openpose_exe, model_folder=args.model_folder, force=args.force,
                 conf_thresh=args.conf_thresh, interp_fill=args.interp_fill, interp_limit=args.interp_limit,
                 num_gpu=DEFAULT_NUM_GPU, num_gpu_start=DEFAULT_NUM_GPU_START, cuda_devices=DEFAULT_CUDA_DEVICES,
                 progress_bar=pbar)
