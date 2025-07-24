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
        description="TimeSformer embedding extraction with fixed train/valid/test split"
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

# ---------------------------------------------------
# 인수로 받은 경로 검증
# ---------------------------------------------------
# root 디렉토리 확인
if not args.root.is_dir():
    sys.exit(f"[ERROR] --root 경로가 존재하지 않거나 디렉토리가 아닙니다: {args.root}")

# train/valid/test 리스트 파일 확인
if args.test_list:
    if not args.test_list.is_file():
        sys.exit(f"[ERROR] --test-list 파일이 존재하지 않습니다: {args.test_list}")
else:
    if args.train_list is None or not args.train_list.is_file():
        sys.exit(f"[ERROR] --train-list 파일을 지정하고, 경로가 유효한지 확인하세요: {args.train_list}")
    if args.valid_list is None or not args.valid_list.is_file():
        sys.exit(f"[ERROR] --valid-list 파일을 지정하고, 경로가 유효한지 확인하세요: {args.valid_list}")

# pretrained 모델 파일 확인
if not args.pretrained.is_file():
    sys.exit(f"[ERROR] --pretrained 모델 파일이 존재하지 않습니다: {args.pretrained}")

# output-dir 위치 생성 가능 여부 확인
try:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_file = args.output_dir / ".write_test"
    with open(test_file, "w") as f:
        f.write("test")
    test_file.unlink()
except Exception as e:
    sys.exit(f"[ERROR] --output-dir 경로에 쓸 수 없습니다 ({args.output_dir}): {e}")

# ---------------------------------------------------
# reproducibility 설정
# ---------------------------------------------------
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
        id2item = {}
        for sub, label in mapping.items():
            video_dir = root / sub / "crop_video"
            if not video_dir.exists():
                continue
            for p in video_dir.glob("*.mp4"):
                stem = p.stem
                vid_id = stem   # video ID는 파일 이름, ids_***.txt와 비교해서 찾음 
                # ids와 파일 이름 모두 _crop이 붙어 있기 때문에 그냥 stem을 받는다
                id2item[vid_id] = (p, label)
        self.samples = [id2item[i] for i in id_list if i in id2item]
        missing = set(id_list) - id2item.keys()
        if missing:
            print(f"⚠️ {len(missing)} ids not found: {list(missing)[:5]} ...")
        print(f"✅ {len(self.samples)} samples loaded from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clips = load_clip(path)                 # list of (3,T,H,W)
        clips = torch.stack(clips, dim=0)       # (CLIPS_PER_VID,3,T,H,W)
        return clips, label

# load ID lists and create loaders
if args.test_list:
    test_ids = args.test_list.read_text().splitlines()
    test_ds  = SwingDataset(args.root, test_ids)
    test_ld  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
else:
    train_ids = args.train_list.read_text().splitlines()
    valid_ids = args.valid_list.read_text().splitlines()
    train_ds  = SwingDataset(args.root, train_ids)
    valid_ds  = SwingDataset(args.root, valid_ids)
    train_ld  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    valid_ld  = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
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

# extraction function
def extract(loader, split_name):
    embs, labels = [], []
    for clips, label in tqdm(loader, desc=f"{split_name} Embeddings", ncols=80):
        clips = clips.squeeze(0)
        clip_embs = []
        for clip in clips:
            c = clip.unsqueeze(0).to(device)
            with torch.no_grad():
                feats = model.model.forward_features(c)
            cls = feats[:,0,:] if feats.ndim == 3 else feats
            clip_embs.append(cls.squeeze(0).cpu().numpy())
            del c, feats, cls
            torch.cuda.empty_cache()
        emb = np.stack(clip_embs,0).mean(0)
        embs.append(emb)
        labels.append(label)
    out_dir = args.output_dir / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', np.stack(embs))
    np.save(out_dir / 'labels.npy',    np.array(labels))
    print(f"✅ {split_name} saved to {out_dir}")

# run extraction
if args.test_list:
    extract(test_ld, 'test')
else:
    extract(train_ld, 'train')
    extract(valid_ld, 'valid')
