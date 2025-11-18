#!/usr/bin/env python
"""
finetune_stgcn.py

ST-GCN fine-tuning using an existing combined annotation PKL
and MMAction2 tools/train.py.
Adds --shuffle option to re-split the PKL data into train/val sets
based on a fixed ratio. The shuffled PKL is temporary and deleted after training.
"""

import argparse
import os
import sys
import pickle
import subprocess
import random
from collections import Counter
import uuid # ⭐️ [추가/수정] resplit_pkl 함수에서 사용되므로 import합니다.

# --- Global Configuration ---
# MMAction2 루트 경로 (tools/train.py 접근용)
MM_ROOT = r"D:\mmaction2"
sys.path.append(MM_ROOT)

# Frequently-used defaults (hardcoded for this task)
DEVICE = "cuda:0"
VAL_SPLIT = "xsub_val" # MMAction2의 split name은 그대로 사용
RESPLIT_RATIO = 0.1 

# Fixed paths for 5-Class (Default)
DEFAULT_INPUT_PKL = r"D:\golfDataset\crop_pkl\combined_5class.pkl"
DEFAULT_CFG = r"configs\skeleton\stgcnpp\my_stgcnpp.py"

# Fixed paths for 3-Class
THREE_CLASS_INPUT_PKL = r"D:\golfDataset\crop_pkl\combined_3class.pkl"
THREE_CLASS_CFG = r"configs\skeleton\stgcnpp\my_stgcnpp_3class.py"

DEFAULT_PRETRAINED = r"D:\mmaction2\checkpoints\stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth"
DEFAULT_WORK_DIR = r"D:\work_dirs\finetune_stgcn_shuffle"

# ----------------------------------------------------------------------
# PKL Manipulation Helper Functions (수정 없음)
# ----------------------------------------------------------------------

def _get_all_annotation_indices(data: dict) -> list:
    """PKL 파일에서 모든 annotation의 인덱스(0부터 N-1)를 가져옵니다."""
    anns = data.get('annotations', [])
    return list(range(len(anns)))

def _update_splits(data: dict, train_indices: list, val_indices: list):
    """PKL 데이터의 split 정보를 새로운 train/val 인덱스로 덮어씁니다."""
    data['split'] = {
        'xsub_train': train_indices,
        'xsub_val': val_indices
        # Test split은 건드리지 않습니다.
    }

def resplit_pkl(input_pkl_path: str, ratio: float, seed: int = 42) -> str:
    """
    기존 PKL 파일의 모든 annotation을 ratio에 따라 train/val로 재분할하고,
    임시 PKL 파일을 생성하여 그 경로를 반환합니다.
    """
    print(f"\n[SHUFFLE] Re-splitting data with validation ratio {ratio}...")
    
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Use stratified split based on annotation labels to preserve class ratios
    anns = data.get('annotations', [])
    # Build label -> indices map and sample per-label
    def _stratified_split(annotations, val_ratio, seed=42):
        from collections import defaultdict
        label_to_indices = defaultdict(list)
        for idx, ann in enumerate(annotations):
            label_to_indices[ann.get('label')].append(idx)

        val_indices = set()
        random.seed(seed)
        for label, idxs in label_to_indices.items():
            if not idxs:
                continue
            k = int(len(idxs) * val_ratio)
            if k == 0 and val_ratio > 0:
                k = 1
            k = min(k, len(idxs))
            if k > 0:
                sampled = random.sample(idxs, k)
                val_indices.update(sampled)

        train_indices = [i for i in range(len(annotations)) if i not in val_indices]
        val_indices = sorted(list(val_indices))
        return train_indices, val_indices

    train_indices, val_indices = _stratified_split(anns, ratio, seed=seed)
    _update_splits(data, train_indices, val_indices)

    num_total = len(anns)
    pct_train = (len(train_indices) / num_total) * 100.0 if num_total else 0.0
    pct_val = (len(val_indices) / num_total) * 100.0 if num_total else 0.0
    print(f"[SHUFFLE] Total annotations: {num_total}")
    print(f"[SHUFFLE] New Train size: {len(train_indices)} ({pct_train:.1f}%)")
    print(f"[SHUFFLE] New Val size: {len(val_indices)} ({pct_val:.1f}%)")
    
    # 임시 PKL 파일 저장
    tmp_pkl_dir = 'tmp_pkl'
    os.makedirs(tmp_pkl_dir, exist_ok=True)
    # UUID를 사용하여 고유한 임시 파일 이름 생성
    temp_pkl_path = os.path.join(tmp_pkl_dir, f"{os.path.basename(input_pkl_path).replace('.pkl', '')}_{uuid.uuid4().hex[:8]}_shuffled.pkl")
    
    with open(temp_pkl_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"[SHUFFLE] Temporary shuffled PKL saved to: {temp_pkl_path}")
    return temp_pkl_path

# ----------------------------------------------------------------------
# Main Logic
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # 🚨 epochs 인수를 제거하고 Config 파일의 MAX_EPOCHS를 사용합니다.
    # parser.add_argument('--epochs', type=int, default=30, help='max_epochs override')
    parser.add_argument('--test-pkl', required=False, default='',
                        help='(optional) test annotation PKL 경로 (omit to skip test override)')
    parser.add_argument('--shuffle', action='store_true',
                        help=f'PKL의 train/val split을 재분할하고 섞습니다 (ratio={RESPLIT_RATIO}).')
    # ⭐️ [추가] 3-class 모드 옵션 추가
    parser.add_argument('--three_class', action='store_true',
                        help='3-class 모드를 사용하고 해당 설정/PKL 경로를 로드합니다. (num_classes=3)')
    # resume training from last checkpoint in work_dir
    parser.add_argument('--continue', dest='resume', action='store_true',
                        help='Resume training from last checkpoint found in work_dir (reads last_checkpoint file).')
    # override final max epochs (optional)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Optional: override total max epochs for training (e.g. 100)')
    args = parser.parse_args()

    # ⭐️ [수정] 클래스 모드에 따른 기본값 설정 (스크립트 초기에 경로 결정)
    if args.three_class:
        args.input_pkl = THREE_CLASS_INPUT_PKL
        args.cfg = THREE_CLASS_CFG
        # 3-class 모드일 경우 n_classes를 3으로 하드코딩 (나중에 추론 로직에서 오버라이드 가능)
        n_classes_override = 3
    else:
        args.input_pkl = DEFAULT_INPUT_PKL
        args.cfg = DEFAULT_CFG
        # 5-class 모드일 경우 n_classes를 5로 하드코딩
        n_classes_override = 5

    # Use module-level defaults for frequently-changed values
    args.pretrained = DEFAULT_PRETRAINED
    args.work_dir = DEFAULT_WORK_DIR
    args.device = DEVICE
    args.val_split = VAL_SPLIT
    
    print(f"[INFO] Config: {args.cfg}, PKL: {args.input_pkl}")
    
    # ----------------------------------------------------------------------
    # 1. PKL 파일 경로 처리: 셔플 여부에 따라 최종 사용할 PKL 경로 결정 및 임시 파일 추적
    # ----------------------------------------------------------------------
    
    final_pkl_path = args.input_pkl
    temp_pkl_path_to_delete = None # 삭제할 임시 파일 경로를 저장
    
    # 입력으로 주어진 PKL 파일(통합 annotation)이 존재하는지 확인
    if not os.path.isfile(args.input_pkl):
        raise FileNotFoundError(f"Annotation PKL not found: {args.input_pkl}")
    
    if args.shuffle:
        final_pkl_path = resplit_pkl(args.input_pkl, RESPLIT_RATIO)
        temp_pkl_path_to_delete = final_pkl_path # 임시 파일 경로 저장

    print(f"[OK] Using annotation PKL: {final_pkl_path}")
    
    # ----------------------------------------------------------------------
    # 2. 클래스 수 추론 및 정보 출력
    # ----------------------------------------------------------------------
    
    # infer number of classes from PKL (if labels exist) and show distribution
    # (클래스 추론은 최종 PKL 파일로 수행해도 무방함)
    with open(final_pkl_path, 'rb') as f: 
        data = pickle.load(f)
    anns = data.get('annotations', [])
    labels = [a.get('label') for a in anns if 'label' in a]
    
    n_classes_inferred = None
    if labels:
        cnt = Counter(labels)
        # 클래스 레이블이 0부터 시작한다고 가정하고 최대값 + 1을 사용
        n_classes_inferred = max(cnt.keys()) + 1 if cnt else 0 
        print(f"[OK] Detected labels: {dict(cnt)}, Inferred n_classes={n_classes_inferred}")
    else:
        print("[WARN] No 'label' field found in annotations.")
    
    # 최종적으로 사용할 n_classes 결정 (하드코딩된 값 > 추론된 값)
    n_classes_final = n_classes_override
    if n_classes_inferred is not None and n_classes_inferred != n_classes_override:
        print(f"[WARN] Inferred n_classes ({n_classes_inferred}) does not match override ({n_classes_override}). Using override value.")
    
    print(f"[INFO] Final model.cls_head.num_classes will be set to: {n_classes_final}")

    # ----------------------------------------------------------------------
    # 3. Config 파일 처리 및 임시 설정 생성
    # ----------------------------------------------------------------------
    
    cfg_path = os.path.join(MM_ROOT, args.cfg.strip())
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # 임시 finetune config 생성 (절대경로 base 상속)
    tmp_cfg_dir = 'tmp_cfg'
    os.makedirs(tmp_cfg_dir, exist_ok=True)
    # ⭐️ [수정] 임시 config 파일에 uuid를 사용하여 고유한 이름 부여 (이전 코드는 고정된 이름을 사용했습니다.)
    finetune_cfg = os.path.join(tmp_cfg_dir, f"finetune_stgcn_cfg_{uuid.uuid4().hex[:8]}.py") 
    
    # Windows 경로 문제를 해결하기 위해 슬래시(/)로 통일
    base_cfg_abs = cfg_path.replace('\\', '/')
    
    with open(finetune_cfg, 'w', encoding='utf-8') as f:
        f.write(f"_base_ = ['{base_cfg_abs}']\n")
    print(f"[OK] generated finetune config: {finetune_cfg}")

    # ----------------------------------------------------------------------
    # 4. MMAction2 tools/train.py 실행 및 자동 삭제
    # ----------------------------------------------------------------------
    
    env = os.environ.copy()
    if args.device.startswith('cuda'):
        env['CUDA_VISIBLE_DEVICES'] = args.device.split(':', 1)[1]

    cmd = [
        sys.executable,
        os.path.join(MM_ROOT, 'tools', 'train.py'),
        finetune_cfg,
        '--work-dir', args.work_dir,
        '--cfg-options',
        # Avoid setting global load_from. Instead set the backbone init_cfg checkpoint.
        f"model.backbone.init_cfg.checkpoint={args.pretrained}",
        
        # ⭐️ Legacy 오버라이드 방식 (오류 가능성 있음): train/val dataloader.dataset 하위의 ann_file 오버라이드
        f"train_dataloader.dataset.dataset.ann_file={final_pkl_path}",
        f"val_dataloader.dataset.ann_file={final_pkl_path}", # RepeatDataset 미사용 시 .dataset 생략
        
        # 기타 학습 설정 (Legacy 오버라이드 방식)
        f"train_dataloader.dataset.dataset.split=xsub_train",
        f"val_dataloader.dataset.split={args.val_split}", # RepeatDataset 미사용 시 .dataset 생략
    ]

    # If user asked to override epochs, append cfg-option for max epochs
    if args.epochs is not None:
        cmd.extend([f"train_cfg.max_epochs={args.epochs}"])

    # Test PKL 경로가 주어진 경우 처리 (Legacy 오버라이드 방식)
    if args.test_pkl:
        if not os.path.isfile(args.test_pkl):
            raise FileNotFoundError(f"Test PKL not found: {args.test_pkl}")
        # Test Dataloader도 경로를 한 단계 덜 깊게 지정 (val과 동일한 구조)
        cmd.extend([f"test_dataloader.dataset.ann_file={args.test_pkl}", f"test_dataloader.dataset.split={args.val_split}"])

    # 클래스 수가 결정된 경우, 모델 헤드 업데이트
    if n_classes_final is not None:
        cmd.extend([f"model.cls_head.num_classes={n_classes_final}"])

    # If resume requested, try to find last checkpoint in work_dir (MMEngine writes a `last_checkpoint` file)
    if args.resume:
        resume_ckpt = None
        try:
            lc_path = os.path.join(args.work_dir, 'last_checkpoint')
            if os.path.isfile(lc_path):
                with open(lc_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        resume_ckpt = content
            # fallback: look for epoch_*.pth in work_dir
            if not resume_ckpt and os.path.isdir(args.work_dir):
                files = [f for f in os.listdir(args.work_dir) if f.endswith('.pth')]
                # prefer epoch_N.pth highest N
                epoch_files = [f for f in files if f.startswith('epoch_')]
                if epoch_files:
                    # sort by epoch number
                    def epoch_num(fn):
                        try:
                            return int(fn.split('epoch_')[-1].split('.pth')[0])
                        except Exception:
                            return -1
                    epoch_files.sort(key=epoch_num, reverse=True)
                    resume_ckpt = os.path.join(args.work_dir, epoch_files[0])
        except Exception:
            resume_ckpt = None

        if resume_ckpt and os.path.isfile(resume_ckpt):
            # MMAction2's train.py expects the --resume option (with a path),
            # not --resume-from. Use --resume to pass the checkpoint path.
            cmd.extend(['--resume', resume_ckpt])
            print(f"Resuming training from checkpoint: {resume_ckpt}")
        else:
            print(f"Resume requested but no checkpoint found in work_dir={args.work_dir}. Continuing from scratch.")

    print("\n[RUNNING] ", ' '.join(cmd))
    
    try:
        # MMAction2 훈련 실행
        subprocess.run(cmd, check=True, env=env)
    finally:
        # 훈련 성공/실패와 관계없이 임시 파일 정리
        if temp_pkl_path_to_delete and os.path.exists(temp_pkl_path_to_delete):
            print(f"\n[CLEANUP] Deleting temporary shuffled PKL: {temp_pkl_path_to_delete}")
            os.remove(temp_pkl_path_to_delete)
            print("[CLEANUP] Temporary shuffled PKL cleanup complete.")
        
        # 임시 설정 파일 정리
        if os.path.exists(finetune_cfg):
            os.remove(finetune_cfg)
            print(f"[CLEANUP] Temporary config file deleted: {finetune_cfg}")


if __name__ == '__main__':
    main()