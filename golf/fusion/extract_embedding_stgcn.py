#!/usr/bin/env python3
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description="ST-GCN(mmAction2) embedding extraction with fixed train/valid split"
    )
    parser.add_argument('--csv-root',    type=Path, required=True,
                        help='CSV 폴더 (e.g. D:\\golfDataset\\dataset\\train)')
    parser.add_argument('--train-list',  type=Path, required=True,
                        help='train_ids.txt')
    parser.add_argument('--valid-list',  type=Path, required=True,
                        help='valid_ids.txt')
    parser.add_argument('--cfg',         type=str, required=True,
                        help='configs/stgcnpp/*.py')
    parser.add_argument('--ckpt',        type=str, required=True,
                        help='체크포인트 .pth')
    parser.add_argument('--device',      type=str, default='cuda:0')
    parser.add_argument('--out-dir',     type=Path, required=True,
                        help='임베딩 저장 디렉토리')
    return parser.parse_args()

args = parse_args()

# 1) CSV → annotations 리스트 로드 (사용자 구현)
def load_skeleton_annotations(csv_root: Path):
    anns = []
    for csv in csv_root.rglob('*.csv'):
        vid = csv.stem
        kp = np.loadtxt(csv, delimiter=',')  # 예시
        anns.append({'frame_dir': vid, 'keypoint': kp})
    return anns

# 2) train/valid ID 로드
train_ids = set(args.train_list.read_text().splitlines())
valid_ids = set(args.valid_list.read_text().splitlines())

# 3) ann 분할 & PKL 생성
anns = load_skeleton_annotations(args.csv_root)
train_anns = [a for a in anns if a['frame_dir'] in train_ids]
valid_anns = [a for a in anns if a['frame_dir'] in valid_ids]

args.out_dir.mkdir(parents=True, exist_ok=True)
train_pkl = args.out_dir/'skeleton_dataset_train.pkl'
valid_pkl = args.out_dir/'skeleton_dataset_valid.pkl'

with open(train_pkl, 'wb') as f:
    pickle.dump({'annotations': train_anns, 'split': {'xsub_train': list(train_ids)}}, f)
with open(valid_pkl, 'wb') as f:
    pickle.dump({'annotations': valid_anns, 'split': {'xsub_val': list(valid_ids)}}, f)

# 4) MMAction2 Config & Runner 준비
cfg = Config.fromfile(args.cfg)
# train / valid 각각 ann_file 덮어쓰기
for split_name, pkl in [('test_dataloader', valid_pkl), ('data.test', valid_pkl)]:
    if hasattr(cfg, split_name):
        getattr(cfg, split_name).dataset.ann_file = str(pkl)
    else:
        cfg.data.test.ann_file = str(pkl)

runner = Runner.from_cfg(cfg)
load_checkpoint(runner.model, args.ckpt, map_location='cpu', strict=False)
runner.model.to(args.device).eval()

# 5) hook 으로 embedding 추출
last_lin = None
for m in runner.model.cls_head.modules():
    if isinstance(m, nn.Linear):
        last_lin = m
if last_lin is None:
    raise RuntimeError("Linear head not found")

def extract(pkl_path, split_tag):
    final_embs, final_labels = [], []
    cfg_name = pkl_path.stem
    # 해당 split 설정
    if hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.dataset.ann_file = str(pkl_path)
    else:
        cfg.data.test.ann_file = str(pkl_path)

    runner.load_checkpoint(args.ckpt, map_location='cpu')  # reload to apply new ann_file
    runner.model.to(args.device)

    with torch.no_grad():
        for batch in runner.test_dataloader:
            clip_embs = []
            data_samples = batch['data_samples']
            label = int(data_samples[0].gt_label)
            handle = last_lin.register_forward_hook(
                lambda m, inp, out: clip_embs.append(inp[0].detach().cpu().squeeze(0))
            )
            inp = batch['inputs']
            # 일괄 처리
            if isinstance(inp, list):
                for clip in inp:
                    runner.model.forward(clip.unsqueeze(0).to(args.device),
                                         data_samples, mode='predict')
            else:
                inp = {k:v.to(args.device) for k,v in inp.items()} if isinstance(inp, dict) else inp.unsqueeze(0).to(args.device)
                runner.model.forward(inp, data_samples, mode='predict')
            handle.remove()

            vs = torch.stack(clip_embs,0).mean(0)
            final_embs.append(vs.cpu().numpy())
            final_labels.append(label)

    emb_arr = np.stack(final_embs,0)
    lbl_arr = np.array(final_labels, dtype=np.int64)
    np.save(args.out_dir/f"{split_tag}_embeddings.npy", emb_arr)
    np.save(args.out_dir/f"{split_tag}_labels.npy",     lbl_arr)
    print(f"✅ {split_tag} done: {emb_arr.shape}")

# 6) 실제 추출 (train/valid)
extract(train_pkl, 'train')
extract(valid_pkl, 'valid')
