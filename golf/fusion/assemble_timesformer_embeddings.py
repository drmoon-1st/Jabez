#!/usr/bin/env python3
"""
assemble_timesformer_embeddings.py

- per_video/{video_id}.npy, {video_id}.json 파일들을 id 리스트에 맞춰 조립
- embeddings.npy, labels.npy, ids.npy로 저장
- split, shuffle, stratify 등 자유롭게 가능
"""
import argparse
from pathlib import Path
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-video-dir', type=Path, required=True, help='per_video 폴더 경로')
    parser.add_argument('--id-list', type=Path, required=True, help='id 리스트(txt, 1줄 1id)')
    parser.add_argument('--out-dir', type=Path, required=True, help='조립 결과 저장 폴더')
    return parser.parse_args()

args = parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

ids = args.id_list.read_text().splitlines()
embs, labels, out_ids = [], [], []
for vid in ids:
    npy_path = args.per_video_dir / f"{vid}.npy"
    json_path = args.per_video_dir / f"{vid}.json"
    if not npy_path.exists():
        print(f"[WARN] {vid}.npy 없음"); continue
    emb = np.load(npy_path)
    embs.append(emb)
    # label/meta
    if json_path.exists():
        with open(json_path, encoding='utf-8') as f:
            meta = json.load(f)
        labels.append(meta.get('label', -1))
    else:
        labels.append(-1)
    out_ids.append(vid)
embs = np.stack(embs)
labels = np.array(labels)
ids = np.array(out_ids)
np.save(args.out_dir / 'embeddings.npy', embs)
np.save(args.out_dir / 'labels.npy', labels)
np.save(args.out_dir / 'ids.npy', ids)
print(f"Saved: {embs.shape}, {labels.shape}, {ids.shape}") 