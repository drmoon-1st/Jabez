#!/usr/bin/env python3
import os, sys
# (필요시) 중복 OpenMP 런타임 허용
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# MMAction2 모듈 경로 등록
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
        description="ST-GCN 임베딩 추출 (mmAction2, 기존 PKL 사용)"
    )
    parser.add_argument('--cfg',      type=str,   required=True, help='configs/skeleton/stgcnpp/my_stgcnpp.py')
    parser.add_argument('--ckpt',     type=str,   required=True, help='체크포인트 .pth 파일 경로')
    parser.add_argument('--device',   type=str,   default='cuda:0', help='실행 디바이스 (e.g. cuda:0)')
    parser.add_argument('--out-dir',  type=Path, required=True, help='임베딩 저장 디렉토리')
    parser.add_argument('--train-pkl',type=Path, required=True, help='생성된 train PKL 파일 경로')
    parser.add_argument('--valid-pkl',type=Path, required=True, help='생성된 valid PKL 파일 경로')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers')
    return parser.parse_args()


def extract(pkl_path: Path, split_tag: str, cfg_file: str, ckpt: str, device: str, out_dir: Path, num_workers: int):
    # Config 로드 및 테스트 데이터로더 설정
    cfg = Config.fromfile(cfg_file)
    # 항상 test loader 사용, split은 'xsub_val'
    if hasattr(cfg, 'test_dataloader'):
        dl_cfg = cfg.test_dataloader
        dl_cfg.num_workers = num_workers
        dl_cfg.persistent_workers = False if 'persistent_workers' in dl_cfg else False
        dl_cfg.dataset.ann_file = str(pkl_path)
    else:
        data_cfg = cfg.data.test
        data_cfg.num_workers = num_workers
        if 'persistent_workers' in data_cfg:
            data_cfg.persistent_workers = False
        data_cfg.dataset.ann_file = str(pkl_path)

    # Runner 준비
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, ckpt, map_location='cpu', strict=False)
    runner.model.to(device).eval()

    # cls_head 마지막 Linear 모듈 찾기
    last_lin = next((m for m in runner.model.cls_head.modules() if isinstance(m, nn.Linear)), None)
    if last_lin is None:
        raise RuntimeError("cls_head 내부에 nn.Linear 레이어가 없습니다.")

    embs, labels = [], []
    with torch.no_grad():
        for batch in runner.test_dataloader:
            data_samples = batch['data_samples']
            label = int(data_samples[0].gt_label)

            clip_embs = []
            # hook으로 입력 특징 저장
            handle = last_lin.register_forward_hook(lambda m, inp, out: clip_embs.append(inp[0].cpu().squeeze(0)))

            inputs = batch['inputs']
            if isinstance(inputs, list):
                for clip in inputs:
                    runner.model.forward(clip.unsqueeze(0).to(device), data_samples, mode='predict')
            else:
                inp = inputs.unsqueeze(0).to(device) if torch.is_tensor(inputs) else {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
                runner.model.forward(inp, data_samples, mode='predict')

            handle.remove()
            # 클립별 임베딩 평균
            video_emb = torch.stack(clip_embs, 0).mean(0).cpu().numpy()
            embs.append(video_emb)
            labels.append(label)

    em_arr = np.stack(embs, 0)
    lbl_arr = np.array(labels, dtype=np.int64)
    split_dir = out_dir / split_tag
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / 'embeddings.npy', em_arr)
    np.save(split_dir / 'labels.npy', lbl_arr)
    print(f"✅ {split_tag} done: embeddings={em_arr.shape}, labels={lbl_arr.shape}")


def main():
    args = parse_args()
    # train/valid PKL에서 바로 임베딩 추출
    extract(args.train_pkl, 'train', args.cfg, args.ckpt, args.device, args.out_dir, args.num_workers)
    extract(args.valid_pkl, 'valid', args.cfg, args.ckpt, args.device, args.out_dir, args.num_workers)

if __name__ == '__main__':
    main()
