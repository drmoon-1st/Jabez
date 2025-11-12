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
from tqdm import tqdm
import random
import sys


def csv_to_annotation(csv_path: Path, frame_dir: str, label: int, img_shape=(1080, 1920),
                      normalize_method='none', confidence_threshold=0.1):
    """Read a single CSV and produce one annotation dict.

    CSV rows are expected to contain OpenPose pose_keypoints_2d flat arrays:
    [x1,y1,c1, x2,y2,c2, ...]. We extract (x,y) for each keypoint and form a
    (T, V, 2) array (T frames, V keypoints).
    """
    # Try to detect COCO-style columnar CSVs (with columns like 'Nose_x',
    # 'Nose_y', 'Nose_c'). If detected, parse into a (T, V, 2) array using
    # the column layout. Otherwise, fall back to the legacy per-row flat
    # array format handling.
    try:
        df_try = pd.read_csv(csv_path, encoding='utf-8-sig')
        cols = [c.strip().replace('\ufeff', '') for c in df_try.columns]
        COCO_NAMES = [
            'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow',
            'LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
        ]
        expected_cols = []
        for n in COCO_NAMES:
            for s in ['_x', '_y', '_c']:
                expected_cols.append(f"{n}{s}")

        if all(ec in cols for ec in expected_cols):
            # Columnar COCO-style CSV detected.
            df = df_try.copy()
            df.columns = cols
            arr = np.stack([
                df[[f"{name}_x" for name in COCO_NAMES]].values,
                df[[f"{name}_y" for name in COCO_NAMES]].values,
                df[[f"{name}_c" for name in COCO_NAMES]].values
            ], axis=2)
            # arr shape: (T, V, 3) -> we need (T, V, 2)
            keypoint_arr = arr[:, :, :2]
            frames = [row.tolist() for row in keypoint_arr]
        else:
            # fallback to legacy row-based parsing
            df = pd.read_csv(csv_path, header=None)
            if df.size == 0:
                return None

            frames = []
            for _, row in df.iterrows():
                vals = row.dropna()
                if vals.empty:
                    continue
                vals_num = pd.to_numeric(vals, errors='coerce').dropna()
                if vals_num.size == 0:
                    continue
                arr = np.array(vals_num, dtype=float)

                if arr.size % 3 != 0:
                    if arr.size % 2 == 0:
                        pts = arr.reshape(-1, 2)
                    else:
                        pts = np.zeros((25, 2), dtype=float)
                else:
                    pts3 = arr.reshape(-1, 3)
                    pts = pts3[:, :2]

                frames.append(pts.tolist())
    except Exception:
        # if CSV reading fails for any reason, try the legacy path
        df = pd.read_csv(csv_path, header=None)
        if df.size == 0:
            return None
        frames = []
        for _, row in df.iterrows():
            vals = row.dropna()
            if vals.empty:
                continue
            vals_num = pd.to_numeric(vals, errors='coerce').dropna()
            if vals_num.size == 0:
                continue
            arr = np.array(vals_num, dtype=float)

            if arr.size % 3 != 0:
                if arr.size % 2 == 0:
                    pts = arr.reshape(-1, 2)
                else:
                    pts = np.zeros((25, 2), dtype=float)
            else:
                pts3 = arr.reshape(-1, 3)
                pts = pts3[:, :2]

            frames.append(pts.tolist())

        # (removed stray per-file normalization/append which duplicated last frame)

    if len(frames) == 0:
        return None

    keypoint = np.stack([np.array(f) for f in frames], axis=0)  # (T, V, 2)

    # optional normalization applied on the whole array
    if normalize_method == '0to1':
        h, w = img_shape
        kp = keypoint.copy()
        kp[..., 0] = kp[..., 0] / float(w)
        kp[..., 1] = kp[..., 1] / float(h)
        keypoint = kp

    # use float32 arrays so downstream tooling can index with ellipsis
    keypoint = keypoint.astype(np.float32)

    ann = {
        'frame_dir': frame_dir,
        'label': int(label),
        'keypoint': keypoint,
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
    # Instead of scanning the entire tree, only look under the expected
    # evaluation folders (true/false) and the five labels under them. For
    # each of those label folders, look specifically in the `skeleton_crop`
    # subfolder which is expected to contain only the skeleton CSVs. This
    # avoids walking through many irrelevant files (videos, json, jpgs).
    target_labels = ['best', 'good', 'normal', 'bad', 'worst']
    candidates = []

    # top-level evaluation folders commonly used in this dataset
    eval_dirs = ['true', 'false']
    for ed in eval_dirs:
        base = csv_root / ed
        if not base.exists():
            continue
        # check two possible layouts: label dirs directly under true/false,
        # or csv_root/label/skeleton_crop (if labels are at top-level).
        for lab in target_labels:
            # prefer csv_root/ed/label/skeleton_crop
            sc = base / lab / 'skeleton_crop'
            if sc.exists():
                candidates.append(sc)
            else:
                # fallback: csv_root/label/skeleton_crop (no true/false)
                sc2 = csv_root / lab / 'skeleton_crop'
                if sc2.exists():
                    candidates.append(sc2)

    # As a fallback, if no candidates were found, try the legacy path under
    # csv_root/*/skeleton_crop to be permissive but still avoid scanning all files.
    if not candidates:
        for p in csv_root.iterdir():
            sc = p / 'skeleton_crop'
            if sc.exists():
                candidates.append(sc)

    # Collect CSV files from the selected skeleton_crop folders only.
    all_csvs = []
    for c in candidates:
        all_csvs.extend([p for p in c.iterdir() if p.suffix.lower() == '.csv'])
    all_csvs = sorted(all_csvs)
    print(f"Found {len(all_csvs)} CSV files under selected skeleton_crop folders of {csv_root}")

    annotations = []
    samples = []
    for csv in tqdm(all_csvs, desc="Processing CSVs", unit="file"):
        # Determine label from the path: expect one of the target labels to be
        # in the ancestry. If not found, fall back to detect_label_from_path.
        parts = [s.lower() for s in csv.parts]
        label = None
        for name, lab in {
            'worst': 0,
            'bad': 1,
            'normal': 2,
            'good': 3,
            'best': 4,
        }.items():
            if name in parts:
                label = lab
                break
        if label is None:
            label = detect_label_from_path(csv)

        # Derive a unique frame_dir relative to the csv_root. Prefer the
        # parent folder name (the sample folder inside skeleton_crop). If the
        # CSVs are named as <sample>.csv, use that stem but prefix with the
        # label and eval dir to ensure uniqueness across the dataset.
        try:
            rel = csv.relative_to(csv_root)
            # rel might be like 'false/bad/skeleton_crop/20201208_.../skeleton.csv'
            # Use parts from rel to construct a stable frame_dir
            rel_parts = list(rel.parts)
            # drop the trailing file name and 'skeleton_crop' if present
            if rel_parts[-2].lower() == 'skeleton_crop':
                parent_sample = rel_parts[-1]
                # parent_sample is the csv filename; use its stem
                frame_dir = f"{rel_parts[0]}/{rel_parts[1]}/{Path(parent_sample).stem}"
            else:
                frame_dir = Path(csv.stem).name
        except Exception:
            # fallback
            frame_dir = csv.stem
        ann = csv_to_annotation(csv, frame_dir=frame_dir, label=label,
                                img_shape=img_shape, normalize_method=normalize_method)
        if ann is None:
            print(f"[WARN] skipping empty or unreadable CSV: {csv}")
            continue
        annotations.append(ann)
        samples.append(frame_dir)

    # Diagnostic counts before normalization
    parsed_ann_count = len(annotations)
    print(f"Parsed annotations from CSVs: {parsed_ann_count}")

    # Normalize/validate collected annotations: ensure 'keypoint' is a numpy
    # ndarray of dtype float32 and shape (M, T, V, 2) where M is number of
    # persons (usually 1). This accepts the legacy (T, V, 2) format produced
    # by csv_to_annotation and wraps it into (1, T, V, 2). We also populate
    # helpful metadata fields expected by the MMAction2 pipeline.
    cleaned_annotations = []
    skipped_short = 0
    skipped_shape = 0
    skipped_conversion = 0
    for ann in tqdm(annotations, desc="Normalizing annotations", unit="sample"):
        kp = ann.get('keypoint')
        if isinstance(kp, list):
            try:
                kp = np.stack([np.array(f) for f in kp], axis=0).astype(np.float32)
            except Exception:
                print(f"[WARN] could not convert keypoint list for {ann.get('frame_dir')}, skipping sample")
                skipped_conversion += 1
                continue
        elif isinstance(kp, np.ndarray):
            kp = kp.astype(np.float32)
        else:
            try:
                kp = np.array(kp, dtype=np.float32)
            except Exception:
                print(f"[WARN] unknown keypoint type for {ann.get('frame_dir')}, skipping sample")
                skipped_conversion += 1
                continue

        # Accept legacy shape (T, V, 2) and wrap to (1, T, V, 2)
        if kp.ndim == 3 and kp.shape[2] == 2:
            kp = np.expand_dims(kp, axis=0)  # -> (1, T, V, 2)
        # If already in (M, T, V, 2), accept as-is
        elif kp.ndim == 4 and kp.shape[3] == 2:
            pass
        else:
            print(f"[WARN] keypoint for {ann.get('frame_dir')} has invalid shape {getattr(kp, 'shape', None)}, skipping")
            skipped_shape += 1
            continue

        # Ensure dtype
        kp = kp.astype(np.float32)

        # Populate/ensure metadata fields downstream components expect
        total_frames = kp.shape[1]
        # Exclude samples with 50 frames or fewer per user request
        if int(total_frames) <= 50:
            skipped_short += 1
            # skip this sample
            continue
        img_shape = ann.get('img_shape', (1080, 1920))
        # keypoint_score: default to ones if not provided (shape: M, T, V)
        if 'keypoint_score' in ann:
            kps = np.array(ann['keypoint_score'], dtype=np.float32)
            # if legacy (T, V) -> wrap
            if kps.ndim == 2:
                kps = np.expand_dims(kps, axis=0)
        else:
            kps = np.ones((kp.shape[0], kp.shape[1], kp.shape[2]), dtype=np.float32)

        ann['keypoint'] = kp
        ann['keypoint_score'] = kps
        ann['total_frames'] = int(total_frames)
        ann['original_shape'] = tuple(ann.get('original_shape', img_shape))
        ann['img_shape'] = tuple(img_shape)
        ann.setdefault('metainfo', {'frame_dir': ann.get('frame_dir')})

        # append this normalized/validated annotation
        cleaned_annotations.append(ann)

    annotations = cleaned_annotations

    # Diagnostic summary after normalization
    final_count = len(annotations)
    print(f"Normalization summary: parsed={parsed_ann_count}, final={final_count}, skipped_short={skipped_short}, skipped_shape={skipped_shape}, skipped_conversion={skipped_conversion}")

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

    print(f"[OK] Wrote combined PKL: {out_pkl} -> {len(annotations)} samples (skipped {skipped_short} samples with <=50 frames)")


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
