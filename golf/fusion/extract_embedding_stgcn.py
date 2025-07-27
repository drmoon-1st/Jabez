import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
BASE_DIR = r"D:\mmaction2"
sys.path.insert(0, BASE_DIR)

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="ST-GCN 임베딩 추출")
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--train-pkl', type=Path)
    parser.add_argument('--valid-pkl', type=Path)
    parser.add_argument('--test-pkl', type=Path)
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()

def get_id_list_from_pkl(pkl_path):
    """PKL에서 frame_dir을 순서대로 뽑아냄."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return [anno['frame_dir'] for anno in data['annotations']]

def patch_pipeline(cfg, pkl_path, num_workers):
    # 동일
    if hasattr(cfg, 'test_dataloader'):
        dl_cfg = cfg.test_dataloader
        dl_cfg.num_workers = num_workers
        dl_cfg.persistent_workers = False
        dl_cfg.dataset.ann_file = str(pkl_path)
    else:
        data_cfg = cfg.data.test
        data_cfg.num_workers = num_workers
        if 'persistent_workers' in data_cfg:
            data_cfg.persistent_workers = False
        data_cfg.dataset.ann_file = str(pkl_path)

def extract(pkl_path: Path, split_tag: str, cfg_file: str, ckpt: str, device: str, out_dir: Path, num_workers: int):
    cfg = Config.fromfile(cfg_file)
    patch_pipeline(cfg, pkl_path, num_workers)
    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, ckpt, map_location='cpu', strict=False)
    runner.model.to(device).eval()

    last_lin = next((m for m in runner.model.cls_head.modules() if isinstance(m, nn.Linear)), None)
    if last_lin is None:
        raise RuntimeError("cls_head 내부에 nn.Linear 레이어가 없습니다.")

    # 🔥 PKL에서 직접 id 목록 추출
    gt_ids = get_id_list_from_pkl(pkl_path)

    embs, labels, ids = [], [], []
    feat_dim = last_lin.in_features
    idx = 0  # GT ID 인덱스
    with torch.no_grad():
        for batch in runner.test_dataloader:
            data_samples = batch['data_samples']
            # 여러 DataSample이 올 수 있음
            for i, ds in enumerate(data_samples):
                label = int(ds.gt_label)
                # 무조건 pkl에서 id를 순서대로 가져옴
                frame_dir = gt_ids[idx] if idx < len(gt_ids) else None
                idx += 1

                clip_embs = []
                handle = last_lin.register_forward_hook(
                    lambda m, inp, out: clip_embs.append(inp[0].cpu().squeeze(0))
                )
                inputs = batch['inputs']
                # single clip or list
                inp = None
                if isinstance(inputs, list):
                    inp = inputs[i].unsqueeze(0).to(device)
                elif isinstance(inputs, dict):
                    inp = {k: v[i].unsqueeze(0).to(device) for k, v in inputs.items()}
                elif torch.is_tensor(inputs):
                    inp = inputs[i].unsqueeze(0).to(device)

                runner.model.forward(inp, [ds], mode='predict')
                handle.remove()
                if not clip_embs:
                    clip_embs.append(torch.zeros(feat_dim))
                video_emb = torch.stack(clip_embs, 0).mean(0).cpu().numpy()
                video_emb = np.nan_to_num(video_emb, nan=0.0, posinf=0.0, neginf=0.0)
                embs.append(video_emb)
                labels.append(label)
                ids.append(frame_dir)
    em_arr = np.stack(embs, 0)
    lbl_arr = np.array(labels, dtype=np.int64).reshape(-1, 1)
    ids_arr = np.array(ids)
    split_dir = out_dir / split_tag
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / 'embeddings.npy', em_arr)
    np.save(split_dir / 'labels.npy',     lbl_arr)
    np.save(split_dir / 'ids.npy',        ids_arr)
    print(f"\u2705 {split_tag} done: embeddings={em_arr.shape}, labels={lbl_arr.shape}, ids={ids_arr.shape}")

def main():
    args = parse_args()
    if args.test_pkl:
        extract(args.test_pkl, 'test', args.cfg, args.ckpt, args.device, args.out_dir, args.num_workers)
    if args.train_pkl:
        extract(args.train_pkl, 'train', args.cfg, args.ckpt, args.device, args.out_dir, args.num_workers)
    if args.valid_pkl:
        extract(args.valid_pkl, 'valid', args.cfg, args.ckpt, args.device, args.out_dir, args.num_workers)

if __name__ == '__main__':
    main()
