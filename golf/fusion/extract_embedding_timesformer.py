#!/usr/bin/env python3
import sys
sys.path.append(r"D:\timesformer")  # 필요시 Timesformer 모듈 경로
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import InterpolationMode, functional as F
import numpy as np
import random
from decord import VideoReader
from tqdm import tqdm
from timesformer.models.vit import TimeSformer

def parse_args():
    parser = argparse.ArgumentParser(
        description="TimeSformer embedding extraction with fixed train/valid split"
    )
    parser.add_argument('--root',        type=Path, required=True,
                        help='Dataset root (e.g. D:\\golfDataset\\dataset\\train)')
    parser.add_argument('--train-list',  type=Path, required=True,
                        help='train_ids.txt')
    parser.add_argument('--valid-list',  type=Path, required=True,
                        help='valid_ids.txt')
    parser.add_argument('--num-frames',  type=int, default=32)
    parser.add_argument('--clips-per-vid', type=int, default=5)
    parser.add_argument('--img-size',    type=int, default=224)
    parser.add_argument('--batch-size',  type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pretrained',  type=Path, required=True,
                        help='사전학습 모델 경로 (.pyth)')
    parser.add_argument('--output-dir',  type=Path, required=True,
                        help='임베딩 저장 디렉토리')
    return parser.parse_args()

args = parse_args()

# 재현성
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# 전처리 함수
def preprocess_tensor(img_tensor):
    img = F.resize(img_tensor, 256, interpolation=InterpolationMode.BICUBIC)
    img = F.center_crop(img, args.img_size)
    img = F.normalize(img, [0.45]*3, [0.225]*3)
    return img

def uniform_sample(length, num):
    if length >= num:
        return np.linspace(0, length - 1, num).astype(int)
    return np.pad(np.arange(length), (0, num - length), mode='edge')

def load_clip(path: Path):
    vr = VideoReader(str(path))
    L = len(vr)
    seg_edges = np.linspace(0, L, args.clips_per_vid + 1, dtype=int)
    clips = []
    for start, end in zip(seg_edges[:-1], seg_edges[1:]):
        idx = uniform_sample(end - start, args.num_frames) + start
        arr = vr.get_batch(idx).asnumpy().astype(np.uint8)
        clip = torch.from_numpy(arr).permute(0,3,1,2).float()/255.0
        clip = torch.stack([preprocess_tensor(f) for f in clip])
        clips.append(clip.permute(1,0,2,3))
    return clips

class SwingDataset(Dataset):
    def __init__(self, root: Path, id_list):
        mapping = {"balanced_true": 1, "false": 0}
        samples = []
        for sub, label in mapping.items():
            for p in (root/sub/"crop_video").glob("*.mp4"):
                if p.stem in id_list:
                    samples.append((p, label))
        self.samples = samples
        print(f"✅ {len(self.samples)} samples loaded from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return torch.stack(load_clip(path)), label

# ID 리스트 로드
train_ids = args.train_list.read_text().splitlines()
valid_ids = args.valid_list.read_text().splitlines()

# Dataset & DataLoader
full_ds = SwingDataset(args.root, train_ids + valid_ids)
train_idx = [i for i,(p,_) in enumerate(full_ds.samples) if p.stem in train_ids]
valid_idx = [i for i,(p,_) in enumerate(full_ds.samples) if p.stem in valid_ids]

train_loader = DataLoader(
    Subset(full_ds, train_idx),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True
)
valid_loader = DataLoader(
    Subset(full_ds, valid_idx),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True
)

# 모델 로드 & 헤드 제거
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TimeSformer(
    img_size=args.img_size, num_frames=args.num_frames,
    num_classes=2, attention_type='divided_space_time',
    pretrained_model=str(args.pretrained)
).to(device)

model.head = nn.Identity()
if hasattr(model, 'cls_head'): model.cls_head = nn.Identity()
model.model.head = nn.Identity()
if hasattr(model.model, 'cls_head'): model.model.cls_head = nn.Identity()
model.eval()

# 임베딩 추출 함수
def extract(loader, split_name):
    embs, labels = [], []
    for clips, label in tqdm(loader, desc=f"{split_name} Embeddings"):
        clips = clips.squeeze(0).to(device)
        feats = model.model.forward_features(clips)
        if feats.ndim == 3:
            cls_feats = feats[:,0,:]
        else:
            cls_feats = feats
        emb = cls_feats.mean(dim=0).cpu().numpy()
        embs.append(emb)
        labels.append(label.item())
    out_dir = args.output_dir/split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir/'embeddings.npy', np.stack(embs))
    np.save(out_dir/'labels.npy',    np.array(labels))
    print(f"✅ {split_name} saved to {out_dir}")

# 실제 실행
extract(train_loader, 'train')
extract(valid_loader, 'valid')
