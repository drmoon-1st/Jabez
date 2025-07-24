#!/usr/bin/env python3
"""
extract_embedding_timesformer.py

TimeSformer embedding extraction with train/valid/test split and video augmentation
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import random
from decord import VideoReader
from tqdm import tqdm

# 필요시 timesformer 모듈 경로 추가
sys.path.append(r"D:\timesformer")
from timesformer.models.vit import TimeSformer

CURRENT_SPLIT = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="TimeSformer embedding extraction with augmentation"
    )
    parser.add_argument('--root',        type=Path, required=True,
                        help='Dataset root (e.g. D:\\golfDataset\\dataset\\train)')
    parser.add_argument('--train-list',  type=Path,
                        help='newline-separated train video IDs')
    parser.add_argument('--valid-list',  type=Path,
                        help='newline-separated valid video IDs')
    parser.add_argument('--test-list',   type=Path,
                        help='newline-separated test video IDs (optional)')
    parser.add_argument('--num-frames',   type=int, default=32)
    parser.add_argument('--clips-per-vid', type=int, default=5)
    parser.add_argument('--img-size',     type=int, default=224)
    parser.add_argument('--batch-size',   type=int, default=1)
    parser.add_argument('--num-workers',  type=int, default=0)
    parser.add_argument('--pretrained',   type=Path, required=True,
                        help='사전학습 모델 경로 (.pyth)')
    parser.add_argument('--output-dir',   type=Path, required=True,
                        help='임베딩 저장 디렉토리')
    return parser.parse_args()

args = parse_args()

# ────────────── 파라미터 및 경로 검증 ──────────────
if not args.root.is_dir():
    sys.exit(f"[ERROR] --root 경로 없음: {args.root}")

if args.test_list:
    if not args.test_list.is_file():
        sys.exit(f"[ERROR] --test-list 파일 없음: {args.test_list}")
else:
    # 여기서 args.train_list / args.valid_list 로 검사해야 합니다.
    if args.train_list is None or not args.train_list.is_file():
        sys.exit(f"[ERROR] --train-list 경로가 유효하지 않습니다: {args.train_list}")
    if args.valid_list is None or not args.valid_list.is_file():
        sys.exit(f"[ERROR] --valid-list 경로가 유효하지 않습니다: {args.valid_list}")

if not args.pretrained.is_file():
    sys.exit(f"[ERROR] --pretrained 모델 파일 없음: {args.pretrained}")

try:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tp = args.output_dir / ".write_test"
    tp.write_text("test"); tp.unlink()
except Exception as e:
    sys.exit(f"[ERROR] --output-dir에 쓸 수 없습니다 ({args.output_dir}): {e}")

# ────────────── 재현성 ──────────────
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────── Augmentation 정의 ──────────────
def build_transforms(split: str):
    mean = [0.45,0.45,0.45]; std=[0.225,0.225,0.225]
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.4,0.4,0.4,0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

train_transform = build_transforms('train')
eval_transform  = build_transforms('eval')

# ────────────── 비디오 로드 & 샘플링 ──────────────
def uniform_sample(length, num):
    if length >= num:
        return np.linspace(0, length-1, num, dtype=int)
    return np.pad(np.arange(length), (0,num-length), mode='edge')

def load_clip(path: Path):
    vr = VideoReader(str(path))
    L  = len(vr)
    segs = np.linspace(0, L, args.clips_per_vid+1, dtype=int)
    clips = []
    for s,e in zip(segs[:-1], segs[1:]):
        idx = uniform_sample(e-s, args.num_frames) + s
        arr = vr.get_batch(idx).asnumpy()  # (T,H,W,3)
        proc = []
        for frame in arr:
            img = transforms.ToPILImage()(frame)
            if CURRENT_SPLIT == 'train':
                img_t = train_transform(img)
            else:
                img_t = eval_transform(img)
            proc.append(img_t)
        clip = torch.stack(proc, dim=1)  # (C,T,H,W)
        clips.append(clip)
    return clips

# ────────────── Dataset ──────────────
class SwingDataset(Dataset):
    def __init__(self, root: Path, id_list):
        mapping = {"balanced_true":1, "false":0}
        self.samples = []
        for cat, lbl in mapping.items():
            vd = root/ cat/ "crop_video"
            if not vd.exists(): continue
            for mp4 in vd.glob("*.mp4"):
                if mp4.stem in id_list:
                    self.samples.append((mp4, lbl))
        missing = set(id_list) - {p.stem for p,_ in self.samples}
        if missing:
            print(f"⚠️ {len(missing)} missing IDs: {list(missing)[:5]}")
        print(f"✅ [{CURRENT_SPLIT}] samples loaded: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p,l = self.samples[i]
        clips = load_clip(p)
        return torch.stack(clips,0), l

# ────────────── 모델 로드 & head 제거 ──────────────
model = TimeSformer(
    img_size=args.img_size,
    num_frames=args.num_frames,
    num_classes=2,
    attention_type='divided_space_time',
    pretrained_model=str(args.pretrained)
).to(device)

for attr in ('head','cls_head'):
    if hasattr(model, attr): setattr(model, attr, nn.Identity())
    if hasattr(model, 'model') and hasattr(model.model, attr):
        setattr(model.model, attr, nn.Identity())

model.eval()

# ────────────── 추출 함수 ──────────────
def extract(loader, split_name: str):
    global CURRENT_SPLIT
    CURRENT_SPLIT = 'train' if split_name=='train' else 'eval'
    embs, labels = [], []
    for clips, lbl in tqdm(loader, desc=f"{split_name} Embedd", ncols=80):
        clips = clips.squeeze(0)
        feats = []
        for clip in clips:
            c = clip.unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.model.forward_features(c)
            cls = out[:,0,:] if out.ndim==3 else out
            feats.append(cls.squeeze(0).cpu().numpy())
        embs.append(np.stack(feats,0).mean(0))
        labels.append(lbl)
    od = args.output_dir/ split_name
    od.mkdir(exist_ok=True)
    np.save(od/"embeddings.npy", np.stack(embs))
    np.save(od/"labels.npy", np.array(labels))
    print(f"✅ {split_name} → {len(embs)} saved in {od}")

# ────────────── main ──────────────
def main():
    if args.test_list:
        ids = args.test_list.read_text().splitlines()
        ds = SwingDataset(args.root, ids)
        ld = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
        extract(ld, 'test')
    else:
        # train
        ids_tr = args.train_list.read_text().splitlines()
        ds_tr = SwingDataset(args.root, ids_tr)
        ld_tr = DataLoader(ds_tr, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)
        extract(ld_tr, 'train')
        # valid
        ids_va = args.valid_list.read_text().splitlines()
        ds_va = SwingDataset(args.root, ids_va)
        ld_va = DataLoader(ds_va, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
        extract(ld_va, 'valid')

if __name__ == "__main__":
    main()
