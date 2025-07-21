#!/usr/bin/env python3
import os, sys
# (필요시) 중복 OpenMP 런타임 허용
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# MMAction2 루트로 이동 & 모듈 경로 등록
BASE_DIR = r"D:\mmaction2"
sys.path.insert(0, BASE_DIR)

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

args = parse_args()

# 1) CSV → Keypoint annotations 로드
def load_skeleton_annotations(csv_root: Path):
    anns = []
    for csv in csv_root.rglob('*.csv'):
        vid = csv.stem
        kp = np.loadtxt(csv, delimiter=',')  # (T×V, 3) 형태라고 가정
        # 사용하실 형태로 가공해 주세요
        anns.append({'frame_dir': vid, 'keypoint': kp})
    return anns

# 2) train/valid ID 목록
train_ids = set(args.train_list.read_text().splitlines())
valid_ids = set(args.valid_list.read_text().splitlines())

# 3) ann 분리 & PKL 생성 (둘 다 xsub_val 키로!)
anns = load_skeleton_annotations(args.csv_root)
train_anns = [a for a in anns if a['frame_dir'] in train_ids]
valid_anns = [a for a in anns if a['frame_dir'] in valid_ids]

args.out_dir.mkdir(parents=True, exist_ok=True)
train_pkl = args.out_dir / 'skeleton_dataset_train.pkl'
valid_pkl = args.out_dir / 'skeleton_dataset_valid.pkl'

with open(train_pkl, 'wb') as f:
    pickle.dump({
        'annotations': train_anns,
        'split': {'xsub_val': list(train_ids)}
    }, f, protocol=4)

with open(valid_pkl, 'wb') as f:
    pickle.dump({
        'annotations': valid_anns,
        'split': {'xsub_val': list(valid_ids)}
    }, f, protocol=4)

# 4) Config 로드 & ann_file 덮어쓰기 (초기값은 valid_pkl로)
cfg = Config.fromfile(args.cfg)
if hasattr(cfg, 'test_dataloader'):
    cfg.test_dataloader.dataset.ann_file = str(valid_pkl)
else:
    cfg.data.test.ann_file = str(valid_pkl)

# 5) Runner 생성 & 체크포인트 로드
runner = Runner.from_cfg(cfg)
load_checkpoint(runner.model, args.ckpt, map_location='cpu', strict=False)
runner.model.to(args.device).eval()

# 6) 마지막 Linear 레이어 hook 준비
last_lin = None
for m in runner.model.cls_head.modules():
    if isinstance(m, nn.Linear):
        last_lin = m
if last_lin is None:
    raise RuntimeError("cls_head 내부에 nn.Linear 레이어가 없습니다.")

# 7) 임베딩 추출 함수 (train/valid 공용)
def extract(pkl_path: Path, split_tag: str):
    # ann_file 변경
    if hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.dataset.ann_file = str(pkl_path)
    else:
        cfg.data.test.ann_file = str(pkl_path)
    # 체크포인트 재로딩(메타파일 덮어쓰기 위해)
    runner.load_checkpoint(args.ckpt, map_location='cpu')
    runner.model.to(args.device).eval()

    final_embs = []
    final_labels = []

    with torch.no_grad():
        for batch in runner.test_dataloader:
            data_samples = batch['data_samples']
            label = int(data_samples[0].gt_label)

            clip_embs = []
            # hook 등록
            handle = last_lin.register_forward_hook(
                lambda m, inp, out: clip_embs.append(inp[0].cpu().squeeze(0))
            )
            # 입력 분기
            raw = batch['inputs']
            if isinstance(raw, list):
                for clip in raw:
                    runner.model.forward(clip.unsqueeze(0).to(args.device),
                                         data_samples, mode='predict')
            else:
                inp = (raw.unsqueeze(0).to(args.device)
                       if torch.is_tensor(raw)
                       else {k: v.unsqueeze(0).to(args.device) for k,v in raw.items()})
                runner.model.forward(inp, data_samples, mode='predict')
            handle.remove()

            # 클립 임베딩 평균
            video_emb = torch.stack(clip_embs, 0).mean(0)
            final_embs.append(video_emb.cpu().numpy())
            final_labels.append(label)

    emb_arr = np.stack(final_embs, 0)
    lbl_arr = np.array(final_labels, dtype=np.int64)
    np.save(args.out_dir/f"{split_tag}_embeddings.npy", emb_arr)
    np.save(args.out_dir/f"{split_tag}_labels.npy",     lbl_arr)
    print(f"✅ {split_tag} done: embeddings={emb_arr.shape}, labels={lbl_arr.shape}")

# 8) 실제 실행
extract(train_pkl, 'train')
extract(valid_pkl, 'valid')
