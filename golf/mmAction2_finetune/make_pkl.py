#!/usr/bin/env python
"""
make_pkl.py

Collect all skeleton CSV files under a directory (rglob), convert each CSV to a single
annotation (extracting x,y keypoints per frame) and write a single combined PKL
that contains:

    {
        'split': {'xsub_train': [frame_dir, ...]},
        'annotations': [ { 'frame_dir': str, 'label': 0|1, 'keypoint': list(T,V,2), 'img_shape': (H,W) }, ... ]
    }

This output is compatible with the downstream `finetune_stgcn.py` in this folder.

Usage:
    python make_pkl.py --csv-root D:/golfDataset/dataset/train --out combined.pkl

By default all samples are placed into the xsub_train split. Optionally a validation
ratio can be provided to hold out a subset for xsub_val.
"""

from pathlib import Path
import argparse
import pickle
import pandas as pd
import numpy as np
import random
import sys


def csv_to_annotation(csv_path: Path, frame_dir: str, label: int, img_shape=(1080, 1920),
                      normalize_method='none', confidence_threshold=0.1):
    """Read a single CSV and produce one annotation dict.

    CSV rows are expected to contain OpenPose pose_keypoints_2d flat arrays:
    [x1,y1,c1, x2,y2,c2, ...]. We extract (x,y) for each keypoint and form a
    (T, V, 2) array (T frames, V keypoints).
    """
    df = pd.read_csv(csv_path, header=None)
    if df.size == 0:
        return None

    frames = []
    for _, row in df.iterrows():
        arr = np.array(row.dropna(), dtype=float)
        if arr.size % 3 != 0:
            # try to handle rows saved as only x,y pairs
            if arr.size % 2 == 0:
                pts = arr.reshape(-1, 2)
            else:
                # unexpected format
                pts = np.zeros((25, 2), dtype=float)
        else:
            pts3 = arr.reshape(-1, 3)
            pts = pts3[:, :2]

        # optional normalization
        if normalize_method == '0to1':
            h, w = img_shape
            pts = pts.copy()
            pts[:, 0] = pts[:, 0] / float(w)
            pts[:, 1] = pts[:, 1] / float(h)

        frames.append(pts.tolist())

    keypoint = np.stack([np.array(f) for f in frames], axis=0)  # (T, V, 2)

    ann = {
        'frame_dir': frame_dir,
        'label': int(label),
        'keypoint': keypoint.tolist(),
        'img_shape': tuple(img_shape)
    }
    return ann


def detect_label_from_path(p: Path):
    # Support multi-class evaluation labels found in path components.
    # Expected evaluation folders: best, good, normal, bad, worst
    mapping = {
        'worst': 0,
        'bad': 1,
        'normal': 2,
        'good': 3,
        'best': 4,
    }
    parts = [s.lower() for s in p.parts]
    # prefer explicit evaluation labels if present anywhere in the path
    for name, lab in mapping.items():
        if name in parts:
            return lab

    # default: unknown -> assign 'normal' (middle class)
    return 2


def collect_and_make(csv_root: Path, out_pkl: Path, img_shape=(1080, 1920),
                     normalize_method='0to1', val_ratio: float = 0.0, seed: int = 42):
    csv_root = Path(csv_root)
    all_csvs = sorted([p for p in csv_root.rglob('*.csv')])
    print(f"Found {len(all_csvs)} CSV files under {csv_root}")

    annotations = []
    samples = []
    for csv in all_csvs:
        label = detect_label_from_path(csv)
        # use relative stem as frame_dir to be consistent with other tools
        frame_dir = csv.stem
        ann = csv_to_annotation(csv, frame_dir=frame_dir, label=label,
                                img_shape=img_shape, normalize_method=normalize_method)
        if ann is None:
            print(f"[WARN] skipping empty or unreadable CSV: {csv}")
            continue
        annotations.append(ann)
        samples.append(frame_dir)

    # optional train/val split
    random.seed(seed)
    indices = list(range(len(annotations)))
    if val_ratio > 0.0:
        random.shuffle(indices)
        k = int(len(indices) * val_ratio)
        val_idx = set(indices[:k])
        train_dirs = [annotations[i]['frame_dir'] for i in indices if i not in val_idx]
        val_dirs = [annotations[i]['frame_dir'] for i in indices if i in val_idx]
        split = {'xsub_train': train_dirs, 'xsub_val': val_dirs}
    else:
        split = {'xsub_train': [a['frame_dir'] for a in annotations]}

    combined = {
        'split': split,
        'annotations': annotations
    }

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(combined, f)

    print(f"[OK] Wrote combined PKL: {out_pkl} -> {len(annotations)} samples")


def cli():
    parser = argparse.ArgumentParser(description='Make combined PKL from skeleton CSVs')
    parser.add_argument('--csv-root', required=True, help='top directory to rglob for CSV files')
    parser.add_argument('--out', default='combined_ntu.pkl', help='output PKL path')
    parser.add_argument('--img-shape', default='1080,1920', help='H,W for normalization')
    parser.add_argument('--normalize', choices=['0to1', 'none'], default='none',
                        help='whether to normalize keypoints to 0..1 (default: none)')
    parser.add_argument('--val-ratio', type=float, default=0.0, help='holdout ratio for xsub_val')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    h, w = [int(x) for x in args.img_shape.split(',')]
    collect_and_make(Path(args.csv_root), Path(args.out), img_shape=(h, w),
                     normalize_method=args.normalize, val_ratio=args.val_ratio, seed=args.seed)


if __name__ == '__main__':
    cli()
