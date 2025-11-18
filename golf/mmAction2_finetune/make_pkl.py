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
# static-trim utilities (required)
try:
    from module import trim_static_frames
except Exception as e:
    raise ImportError("static_trim module not found or failed to import. Ensure 'module/static_trim.py' exists and is importable.\nOriginal error: {}".format(e))
import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import sys
from collections import Counter # Counter는 사용되지 않지만, 다른 진단 목적으로 유용할 수 있음


def stratified_split_annotations(annotations, val_ratio, seed=42):
    """Return train_indices, val_indices using a stratified split by 'label'.

    For each label, select approx. val_ratio fraction of its indices.
    If val_ratio>0 and a label has at least one sample, at least one sample
    will be selected for validation for that label to ensure coverage.
    """
    from collections import defaultdict
    import random

    label_to_indices = defaultdict(list)
    for idx, ann in enumerate(annotations):
        label = ann.get('label')
        label_to_indices[label].append(idx)

    val_indices = set()
    random.seed(seed)
    for label, idxs in label_to_indices.items():
        if not idxs:
            continue
        k = int(len(idxs) * val_ratio)
        # ensure at least one sample per class in val if val_ratio>0
        if k == 0 and val_ratio > 0:
            k = 1
        k = min(k, len(idxs))
        if k > 0:
            sampled = random.sample(idxs, k)
            val_indices.update(sampled)

    train_indices = [i for i in range(len(annotations)) if i not in val_indices]
    val_indices = sorted(list(val_indices))
    return train_indices, val_indices


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

    keypoint = np.stack([np.array(f) for f in frames], axis=0)    # (T, V, 2)

    # --- Static-trim: remove leading/trailing static frames conservatively ---
    # This trim is always executed; if it fails we log a warning and keep the
    # untrimmed sequence to avoid dropping data.
    try:
        start_idx, end_idx = trim_static_frames(keypoint, fps=30)
        # slice and ensure we still have frames
        if start_idx > 0 or end_idx < (keypoint.shape[0] - 1):
            keypoint = keypoint[start_idx:(end_idx + 1)]
    except Exception as e:
        print(f"[WARN] static trimming failed for {csv_path}: {e}")

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


def detect_label_from_path(p: Path, three_class_mode: bool = False):
    """
    경로에서 레이블을 감지하고, three_class_mode에 따라 레이블을 매핑합니다.
    """
    # 5-class mapping
    mapping_5c = {
        'worst': 0,
        'bad': 1,
        'normal': 2,
        'good': 3,
        'best': 4,
    }
    
    # 3-class mapping: 0: Bad (worst, bad), 1: Normal, 2: Good (good, best)
    mapping_3c_012 = { # 0, 1, 2 매핑 (수정됨)
        'worst': 0,
        'bad': 0,
        'normal': 1,
        'good': 2,
        'best': 2,
    }
    
    mapping = mapping_3c_012 if three_class_mode else mapping_5c
    
    parts = [s.lower() for s in p.parts]
    # prefer explicit evaluation labels if present anywhere in the path
    for name, lab in mapping.items():
        if name in parts:
            return lab

    # default: unknown -> assign 'normal' (middle class)
    # 3-class: 1, 5-class: 2
    return 1 if three_class_mode else 2


def collect_and_make(csv_root: Path, out_pkl: Path, img_shape=(1080, 1920),
                     normalize_method='0to1', val_ratio: float = 0.0, seed: int = 42,
                     test_mode: bool = False, three_class_mode: bool = False):
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

    annotations_5c = [] # 5-class label을 저장할 임시 리스트
    samples = []
    
    # 5-class 레이블 매핑 (파일 경로에서 감지된 원래 레이블)
    mapping_5c = {
        'worst': 0,
        'bad': 1,
        'normal': 2,
        'good': 3,
        'best': 4,
    }
    
    for csv in tqdm(all_csvs, desc="Processing CSVs", unit="file"):
        # Determine label from the path: expect one of the target labels to be
        # in the ancestry. If not found, fall back to detect_label_from_path.
        parts = [s.lower() for s in csv.parts]
        label_5c = None # 5-class 레이블
        for name, lab in mapping_5c.items():
            if name in parts:
                label_5c = lab
                break
        if label_5c is None:
            # detect_label_from_path를 5-class 모드로 호출하여 기본값(2)을 얻습니다.
            label_5c = detect_label_from_path(csv, three_class_mode=False)

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
            
        # csv_to_annotation은 5-class 레이블을 사용하여 어노테이션을 생성합니다.
        ann = csv_to_annotation(csv, frame_dir=frame_dir, label=label_5c,
                                img_shape=img_shape, normalize_method=normalize_method)
        if ann is None:
            print(f"[WARN] skipping empty or unreadable CSV: {csv}")
            continue
        
        # 5-class 레이블을 가진 어노테이션을 저장
        annotations_5c.append(ann) 
        samples.append(frame_dir)

    # Diagnostic counts before normalization
    parsed_ann_count = len(annotations_5c)
    print(f"Parsed annotations from CSVs: {parsed_ann_count}")

    # --- 3-Class Label Transformation (Modified: 0, 1, 2) ---
    # 5-class 레이블을 3-class 레이블 (0, 1, 2)로 변환합니다.
    if three_class_mode:
        print("[INFO] Applying 3-class label transformation: 0(worst,bad), 1(normal), 2(good,best)")
        
        def map_to_3c_012(label_5c):
            """5-class 레이블 (0~4)을 3-class 레이블 (0~2)로 변환합니다."""
            if label_5c in (0, 1):    # worst(0), bad(1) -> 0
                return 0
            elif label_5c == 2:      # normal(2) -> 1
                return 1
            elif label_5c in (3, 4):  # good(3), best(4) -> 2
                return 2
            return 1 # 기본값 (안전 장치: normal)
            
        for ann in annotations_5c:
            ann['label'] = map_to_3c_012(ann['label'])
    
    annotations = annotations_5c # 이후 처리는 변환된 (또는 변환되지 않은) 리스트를 사용

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
    if test_mode:
        # In test mode, follow makePkl_forFintuning convention: put all samples
        # into the validation split only (xsub_val).
        split = {'xsub_val': [a['frame_dir'] for a in annotations]}
    else:
        if val_ratio > 0.0:
            # Perform stratified split to preserve class ratios in val set
            train_idx, val_idx = stratified_split_annotations(annotations, val_ratio, seed=seed)
            train_dirs = [annotations[i]['frame_dir'] for i in train_idx]
            val_dirs = [annotations[i]['frame_dir'] for i in val_idx]
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
    parser.add_argument('--test', action='store_true', help='Write test PKL: place all samples into xsub_val only')
    # three_class 플래그는 여전히 존재하며, 이제 실제로 레이블을 0, 1, 2로 변환합니다.
    parser.add_argument('--three_class', action='store_true', help='Use 3-class labels: 0(worst,bad), 1(normal), 2(good,best)')
    args = parser.parse_args()

    h, w = [int(x) for x in args.img_shape.split(',')]
    collect_and_make(Path(args.csv_root), Path(args.out), img_shape=(h, w),
                     normalize_method=args.normalize, val_ratio=args.val_ratio, seed=args.seed,
                     test_mode=bool(args.test), three_class_mode=bool(args.__dict__.get('three_class', False)))


if __name__ == '__main__':
    cli()