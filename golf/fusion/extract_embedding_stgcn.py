#!/usr/bin/env python3
import os, sys
# (필요시) 중복 OpenMP 런타임 허용
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# MMAction2 모듈 경로
BASE_DIR = r"D:\mmaction2"
sys.path.insert(0, BASE_DIR)

import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

# Body25 → COCO17 인덱스 매핑
MAPPING_BODY25_TO_COCO17 = [
    0,16,15,18,17,
    5,2,6,3,7,
    4,12,9,13,10,
    14,11
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ST-GCN 임베딩 추출 (mmAction2, fixed train/valid split)"
    )
    parser.add_argument('--csv-root',   type=Path, required=True,
                        help='원본 CSV 폴더 (예: D:\\golfDataset\\dataset\\train)')
    parser.add_argument('--train-list', type=Path, required=True,
                        help='train_ids.txt')
    parser.add_argument('--valid-list', type=Path, required=True,
                        help='valid_ids.txt')
    parser.add_argument('--cfg',        type=str, required=True,
                        help='configs/skeleton/stgcnpp/my_stgcnpp.py')
    parser.add_argument('--ckpt',       type=str, required=True,
                        help='체크포인트 .pth 파일 경로')
    parser.add_argument('--device',     type=str, default='cuda:0',
                        help='실행 디바이스 (e.g. cuda:0)')
    parser.add_argument('--out-dir',    type=Path, required=True,
                        help='임베딩 & PKL 생성 폴더')
    return parser.parse_args()


def load_and_process(csv_path: Path,
                     img_shape=(1080,1920),
                     confidence_threshold=0.1,
                     normalize_method='0to1') -> dict:
    df = pd.read_csv(csv_path)
    T = len(df)
    V25 = 25
    kp25 = np.zeros((1, T, V25, 2), dtype=np.float32)
    score25 = np.zeros((1, T, V25), dtype=np.float32)
    for t, row in df.iterrows():
        vals = row.values.reshape(V25, 3)
        kp25[0, t] = vals[:, :2]
        score25[0, t] = vals[:, 2]
    mask = score25 < confidence_threshold
    kp25[mask] = 0
    score25[mask] = 0
    h, w = img_shape
    if normalize_method == '0to1':
        kp25[..., 0] /= w
        kp25[..., 1] /= h
    kp17 = kp25[:, :, MAPPING_BODY25_TO_COCO17, :]
    score17 = score25[:, :, MAPPING_BODY25_TO_COCO17]
    return {
        'total_frames': T,
        'img_shape': img_shape,
        'original_shape': img_shape,
        'keypoint': kp17,
        'keypoint_score': score17
    }


def make_pkls(csv_root, train_ids, valid_ids, out_dir):
    annotations_train, annotations_valid = [], []
    for csv in (csv_root / 'balanced_true' / 'crop_keypoint').glob('*.csv'):
        vid = csv.stem
        label = 1
        annot = load_and_process(csv)
        annot.update({
            'frame_dir': vid,
            'label': label,
            'metainfo': {'frame_dir': vid, 'img_shape': annot['img_shape']}
        })
        if vid in train_ids:
            annotations_train.append(annot)
        if vid in valid_ids:
            annotations_valid.append(annot)
    for csv in (csv_root / 'false' / 'crop_keypoint').glob('*.csv'):
        vid = csv.stem
        label = 0
        annot = load_and_process(csv)
        annot.update({
            'frame_dir': vid,
            'label': label,
            'metainfo': {'frame_dir': vid, 'img_shape': annot['img_shape']}
        })
        if vid in train_ids:
            annotations_train.append(annot)
        if vid in valid_ids:
            annotations_valid.append(annot)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pkl = out_dir / 'skeleton_dataset_train.pkl'
    valid_pkl = out_dir / 'skeleton_dataset_valid.pkl'
    with open(train_pkl, 'wb') as f:
        pickle.dump({'annotations': annotations_train, 'split': {'xsub_val': train_ids}}, f, protocol=4)
    with open(valid_pkl, 'wb') as f:
        pickle.dump({'annotations': annotations_valid, 'split': {'xsub_val': valid_ids}}, f, protocol=4)
    return train_pkl, valid_pkl


def extract(pkl_path: Path, args, split_tag: str):
    cfg = Config.fromfile(args.cfg)
    if hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.num_workers = 0
        cfg.test_dataloader.persistent_workers = False
        cfg.test_dataloader.dataset.ann_file = str(pkl_path)
    else:
        cfg.data.test.num_workers = 0
        if 'persistent_workers' in cfg.data.test:
            cfg.data.test.persistent_workers = False
        cfg.data.test.ann_file = str(pkl_path)
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, args.ckpt, map_location='cpu', strict=False)
    runner.model.to(args.device).eval()
    last_lin = next((m for m in runner.model.cls_head.modules() if isinstance(m, nn.Linear)), None)
    if last_lin is None:
        raise RuntimeError("cls_head 내부에 nn.Linear 레이어가 없습니다.")
    embs, labels = [], []
    with torch.no_grad():
        for batch in runner.test_dataloader:
            ds = batch['data_samples']
            label = int(ds[0].gt_label)
            clip_embs = []
            handle = last_lin.register_forward_hook(lambda m, inp, out: clip_embs.append(inp[0].cpu().squeeze(0)))
            raw = batch['inputs']
            if isinstance(raw, list):
                for clip in raw:
                    runner.model.forward(clip.unsqueeze(0).to(args.device), ds, mode='predict')
            else:
                inp = raw.unsqueeze(0).to(args.device) if torch.is_tensor(raw) else {k: v.unsqueeze(0).to(args.device) for k, v in raw.items()}
                runner.model.forward(inp, ds, mode='predict')
            handle.remove()
            video_emb = torch.stack(clip_embs, 0).mean(0)
            embs.append(video_emb.numpy())
            labels.append(label)
    emb_arr = np.stack(embs, 0)
    lbl_arr = np.array(labels, dtype=np.int64)
    prefix = args.out_dir / split_tag
    prefix.mkdir(parents=True, exist_ok=True)
    np.save(prefix / 'embeddings.npy', emb_arr)
    np.save(prefix / 'labels.npy', lbl_arr)
    print(f"✅ {split_tag} done: embeddings={emb_arr.shape}, labels={lbl_arr.shape}")


def main():
    args = parse_args()
    train_ids = args.train_list.read_text().splitlines()
    valid_ids = args.valid_list.read_text().splitlines()
    train_pkl, valid_pkl = make_pkls(args.csv_root, train_ids, valid_ids, args.out_dir)
    extract(train_pkl, args, 'train')
    extract(valid_pkl, args, 'valid')

if __name__ == '__main__':
    main()
