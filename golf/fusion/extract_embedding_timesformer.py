#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode, functional as F
import numpy as np
import random
from decord import VideoReader
from tqdm import tqdm

# 필요시 timesformer 모듈 경로 추가
sys.path.append(r"D:\timesformer")
from timesformer.models.vit import TimeSformer

def parse_args():
    parser = argparse.ArgumentParser(
        description="TimeSformer embedding extraction with fixed train/valid split"
    )
    parser.add_argument('--root',        type=Path, required=True,
                        help='Dataset root (e.g. D:\\golfDataset\\dataset\\train)')
    parser.add_argument('--train-list',  type=Path, required=True,
                        help='newline-separated train video IDs')
    parser.add_argument('--valid-list',  type=Path, required=True,
                        help='newline-separated valid video IDs')
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

# reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# preprocessing
def preprocess_tensor(img_tensor):
    img = F.resize(img_tensor, 256, interpolation=InterpolationMode.BICUBIC)
    img = F.center_crop(img, args.img_size)
    img = F.normalize(img, [0.45]*3, [0.225]*3)
    return img

def uniform_sample(length, num):
    if length >= num:
        return np.linspace(0, length - 1, num, dtype=int)
    return np.pad(np.arange(length), (0, num - length), mode='edge')

def load_clip(path: Path):
    vr = VideoReader(str(path))
    L = len(vr)
    seg_edges = np.linspace(0, L, args.clips_per_vid + 1, dtype=int)
    clips = []
    for start, end in zip(seg_edges[:-1], seg_edges[1:]):
        idx = uniform_sample(end - start, args.num_frames) + start
        arr = vr.get_batch(idx).asnumpy().astype(np.uint8)  # (T,H,W,3)
        clip = torch.from_numpy(arr).permute(0,3,1,2).float() / 255.0
        clip = torch.stack([preprocess_tensor(f) for f in clip])  # (T,3,H,W)
        clips.append(clip.permute(1,0,2,3))  # (3,T,H,W)
    return clips

class SwingDataset(Dataset):
    def __init__(self, root: Path, id_list):
        mapping = {"balanced_true": 1, "false": 0}
        self.samples = []
        for sub, label in mapping.items():
            video_dir = root / sub / "crop_video"
            if not video_dir.exists():
                continue
            for p in video_dir.glob("*.mp4"):
                stem = p.stem
                vid_id = stem[:-5] if stem.endswith("_crop") else stem
                if vid_id in id_list:
                    self.samples.append((p, label))
        print(f"✅ {len(self.samples)} samples loaded from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clips = load_clip(path)                 # list of (3,T,H,W)
        clips = torch.stack(clips, dim=0)       # (CLIPS_PER_VID,3,T,H,W)
        return clips, label

# load ID lists
train_ids = args.train_list.read_text().splitlines()
valid_ids = args.valid_list.read_text().splitlines()

# datasets & loaders
train_ds = SwingDataset(args.root, train_ids)
valid_ds = SwingDataset(args.root, valid_ids)
train_ld = DataLoader(train_ds, batch_size=1, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)
valid_ld = DataLoader(valid_ds, batch_size=1, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)

# model load & strip heads
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

# extraction
def extract(loader, split_name):
    embs, labels = [], []
    for clips, label in tqdm(loader, desc=f"{split_name} Embeddings", ncols=80):
        clips = clips.squeeze(0)  # (CLIPS_PER_VID,3,T,H,W)
        clip_embs = []
        for clip in clips:
            c = clip.unsqueeze(0).to(device)  # (1,3,T,H,W)
            with torch.no_grad():
                feats = model.model.forward_features(c)
            if feats.ndim == 3:
                cls = feats[:,0,:]  # (1,D)
            else:
                cls = feats        # (1,D)
            clip_embs.append(cls.squeeze(0).cpu().numpy())
            del c, feats, cls
            torch.cuda.empty_cache()
        emb = np.stack(clip_embs,0).mean(0)  # (D,)
        embs.append(emb)
        labels.append(label)
    out_dir = args.output_dir/split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir/'embeddings.npy', np.stack(embs))
    np.save(out_dir/'labels.npy',    np.array(labels))
    print(f"✅ {split_name} saved to {out_dir}")

# run
extract(train_ld, 'train')
extract(valid_ld, 'valid')
