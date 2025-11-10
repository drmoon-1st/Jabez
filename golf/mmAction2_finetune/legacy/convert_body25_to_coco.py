#!/usr/bin/env python3
"""
Convert OpenPose BODY_25 CSVs (x,y,c per joint) into COCO-17 (remapped 18->17) CSVs.

Reads CSVs with columns like 'Nose_x,Nose_y,Nose_c,...' (BODY_25 order or named joints),
maps joints into the COCO17 order used by other pipeline scripts, runs temporal interpolation
to fill short gaps (linear) and writes out a CSV with columns like 'Nose_x,Nose_y,Nose_c,LEye_x,...'.

This script is intended to live in mmAction2_finetune/legacy and be run against
the folder containing *_crop.csv files.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# Body_25 canonical joint names (OpenPose BODY_25)
BODY25 = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

# Target COCO-17 ordering used elsewhere in this repo
KP_17 = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]

# Map KP_17 -> BODY25 index
KP17_TO_BODY25_IDX = {
    "Nose": 0,
    "LEye": 16,
    "REye": 15,
    "LEar": 18,
    "REar": 17,
    "LShoulder": 5,
    "RShoulder": 2,
    "LElbow": 6,
    "RElbow": 3,
    "LWrist": 7,
    "RWrist": 4,
    "LHip": 12,
    "RHip": 9,
    "LKnee": 13,
    "RKnee": 10,
    "LAnkle": 14,
    "RAnkle": 11,
}


def interpolate_sequence(frames_keypoints: List[List[List[float]]], conf_thresh: float = 0.0,
                         method: str = 'linear', fill_method: str = 'none', limit: Optional[int] = None):
    """
    Small local copy of the interpolation helper used elsewhere in the repo.
    frames_keypoints: list of frames; each frame is a list of joints; each joint is [x,y,c]
    Returns interpolated_frames with same shape.
    """
    if not frames_keypoints:
        return []

    n_frames = len(frames_keypoints)
    n_joints = max(len(p) for p in frames_keypoints)

    arr = np.full((n_frames, n_joints * 3), np.nan, dtype=float)
    for t, person in enumerate(frames_keypoints):
        for j, kp in enumerate(person):
            try:
                x = float(kp[0])
                y = float(kp[1])
                c = float(kp[2])
            except Exception:
                x, y, c = np.nan, np.nan, np.nan
            arr[t, j*3 + 0] = x
            arr[t, j*3 + 1] = y
            arr[t, j*3 + 2] = c

    # sentinel (0,0,0) -> missing
    xcols = arr[:, 0::3]
    ycols = arr[:, 1::3]
    ccols = arr[:, 2::3]
    sentinel = (xcols == 0.0) & (ycols == 0.0) & (ccols == 0.0)
    for j in range(n_joints):
        mask = sentinel[:, j]
        arr[mask, j*3:(j+1)*3] = np.nan

    if conf_thresh and conf_thresh > 0.0:
        conf = arr[:, 2::3]
        low_conf = conf < float(conf_thresh)
        for j in range(n_joints):
            mask = low_conf[:, j]
            arr[mask, j*3:(j+1)*3] = np.nan

    cols = []
    for j in range(n_joints):
        cols += [f'x_{j}', f'y_{j}', f'c_{j}']
    df = pd.DataFrame(arr, columns=cols)

    df_interp = df.interpolate(method=method, axis=0, limit=limit, limit_direction='both')

    if fill_method != 'none':
        if fill_method == 'zero':
            df_interp = df_interp.fillna(0.0)
        else:
            df_interp = df_interp.fillna(method=fill_method, limit=None)

    out = []
    darr = df_interp.values
    for t in range(n_frames):
        person = []
        for j in range(n_joints):
            x = darr[t, j*3 + 0]
            y = darr[t, j*3 + 1]
            c = darr[t, j*3 + 2]
            if not np.isfinite(x):
                x = 0.0
            if not np.isfinite(y):
                y = 0.0
            if not np.isfinite(c):
                c = 0.0
            person.append([float(x), float(y), float(c)])
        out.append(person)

    return out


def detect_joint_columns(df: pd.DataFrame) -> List[str]:
    """Return list of unique joint root names found in DataFrame columns (without _x/_y/_c)."""
    cols = df.columns.tolist()
    joints = set()
    rx = re.compile(r'^(?P<name>.+)_(?P<axis>[xyc])$', re.IGNORECASE)
    for c in cols:
        m = rx.match(c)
        if m:
            joints.add(m.group('name'))
    return list(joints)


def read_body25_frames(df: pd.DataFrame) -> List[List[List[float]]]:
    """Read DataFrame with body25-style columns into frames_keypoints list[frame][joint_index][x,y,c].

    If columns are missing for a joint, zeros are used.
    """
    n_frames = len(df)
    frames = []
    # create lowercase->col mapping to be robust
    col_map = {c.lower(): c for c in df.columns}

    for i in range(n_frames):
        row = df.iloc[i]
        person = []
        for name in BODY25:
            xcol = f"{name}_x".lower()
            ycol = f"{name}_y".lower()
            ccol = f"{name}_c".lower()
            x = float(row[col_map[xcol]]) if xcol in col_map else 0.0
            y = float(row[col_map[ycol]]) if ycol in col_map else 0.0
            c = float(row[col_map[ccol]]) if ccol in col_map else 0.0
            person.append([x, y, c])
        frames.append(person)
    return frames


def map_body25_to_coco17(frames_body25: List[List[List[float]]]) -> List[List[List[float]]]:
    """Given frames in BODY25 order, return frames in KP_17 order by selecting indices."""
    out = []
    for frame in frames_body25:
        person = []
        for k in KP_17:
            idx = KP17_TO_BODY25_IDX.get(k)
            if idx is None or idx >= len(frame):
                person.append([0.0, 0.0, 0.0])
            else:
                person.append(frame[idx])
        out.append(person)
    return out


def build_output_df(frames: List[List[List[float]]]) -> pd.DataFrame:
    cols = [f"{k}_{a}" for k in KP_17 for a in ("x", "y", "c")]
    rows = []
    for frame in frames:
        flat = []
        for kp in frame:
            flat.extend([kp[0], kp[1], kp[2]])
        rows.append(flat)
    return pd.DataFrame(rows, columns=cols)


def find_part_person_in_name(name: str) -> Optional[str]:
    # look for 'part##_person##_crop' or 'part##_person##'
    m = re.search(r'(part\d+_person\d+)', name, re.IGNORECASE)
    if m:
        return m.group(1) + '_crop.csv'
    return None


def process_file(path: Path, out_dir: Path, conf_thresh: float = 0.0, interp_fill: str = 'zero', interp_limit: Optional[int] = None, rename: bool = False) -> Path:
    df = pd.read_csv(path)

    # Read body25 frames
    body25_frames = read_body25_frames(df)

    # Map to coco17
    coco_frames = map_body25_to_coco17(body25_frames)

    # Interpolate temporally
    interp = interpolate_sequence(coco_frames, conf_thresh=float(conf_thresh), method='linear', fill_method=interp_fill, limit=interp_limit)

    out_df = build_output_df(interp)

    out_dir.mkdir(parents=True, exist_ok=True)

    if rename:
        candidate = find_part_person_in_name(path.name)
        if candidate:
            out_name = candidate
        else:
            out_name = path.name
    else:
        out_name = path.name

    out_path = out_dir / out_name
    out_df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=r'D:\golfDataset\dataset\train\balanced_true\crop_keypoint', help='Folder containing *_crop.csv files')
    parser.add_argument('--output-dir', type=str, default=str(Path(__file__).resolve().parent / 'converted'), help='Output folder')
    parser.add_argument('--conf-thresh', type=float, default=0.0, help='Confidence threshold to mask before interpolation')
    parser.add_argument('--interp-fill', type=str, default='zero', choices=['none','ffill','bfill','nearest','zero'], help='Fill method after interpolation')
    parser.add_argument('--interp-limit', type=int, default=None, help='Max consecutive NaNs to interpolate')
    parser.add_argument('--rename', action='store_true', help='Rename output files to keep only part##_person##_crop.csv if present')
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    csvs = sorted(inp.glob('*.csv'))
    if not csvs:
        print(f'No CSV files found in {inp}')
        return

    for p in csvs:
        try:
            print(f'Processing {p}')
            outp = process_file(p, out, conf_thresh=args.conf_thresh, interp_fill=args.interp_fill, interp_limit=args.interp_limit, rename=args.rename)
            print(f'Wrote {outp}')
        except Exception as e:
            print(f'Failed {p}: {e}')


if __name__ == '__main__':
    main()
