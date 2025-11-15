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
# import uuid # 주석 처리된 uuid 모듈은 실제 코드에서는 제거하거나 주석 해제해야 함

# --- Global Configuration ---
# MMAction2 루트 경로 (tools/train.py 접근용)
MM_ROOT = r"D:\mmaction2"
sys.path.append(MM_ROOT)

# Frequently-used defaults (hardcoded for this task)
DEVICE = "cuda:0"
# BATCH_SIZE와 LR은 Config 파일에서 제어하므로 스크립트에서 제거함
VAL_SPLIT = "xsub_val" # MMAction2의 split name은 그대로 사용
# --shuffle 옵션 사용 시, Validation 데이터로 사용할 비율 (10%)
RESPLIT_RATIO = 0.1 

# Fixed paths for this project/task (override here instead of CLI)
DEFAULT_INPUT_PKL = r"D:\golfDataset\crop_pkl\combined_5class.pkl"
DEFAULT_CFG = r"configs\skeleton\stgcnpp\my_stgcnpp.py"
DEFAULT_PRETRAINED = r"D:\mmaction2\checkpoints\stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth"
DEFAULT_WORK_DIR = r"D:\work_dirs\finetune_stgcn_shuffle"

# ----------------------------------------------------------------------
# PKL Manipulation Helper Functions
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

    # 모든 annotation 인덱스 가져오기
    all_indices = _get_all_annotation_indices(data)
    
    # 섞기 (재현성을 위해 시드 사용)
    random.seed(seed)
    random.shuffle(all_indices)
    
    # 비율에 따라 인덱스 분할
    num_total = len(all_indices)
    num_val = int(num_total * ratio)
    
    val_indices = all_indices[:num_val]
    train_indices = all_indices[num_val:]
    
    _update_splits(data, train_indices, val_indices)

    print(f"[SHUFFLE] Total annotations: {num_total}")
    print(f"[SHUFFLE] New Train size: {len(train_indices)} ({(len(train_indices)/num_total)*100:.1f}%)")
    print(f"[SHUFFLE] New Val size: {len(val_indices)} ({(len(val_indices)/num_total)*100:.1f}%)")
    
    # 임시 PKL 파일 저장
    tmp_pkl_dir = 'tmp_pkl'
    os.makedirs(tmp_pkl_dir, exist_ok=True)
    # 현재 시간 또는 UUID를 사용하여 고유한 임시 파일 이름 생성
    import uuid
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
    args = parser.parse_args()

    # Use module-level defaults for frequently-changed values
    # args.epochs = args.epochs # 🚨 제거
    args.input_pkl = DEFAULT_INPUT_PKL
    args.cfg = DEFAULT_CFG
    args.pretrained = DEFAULT_PRETRAINED
    args.work_dir = DEFAULT_WORK_DIR
    args.device = DEVICE
    # args.batch_size = BATCH_SIZE # 🚨 제거
    # args.lr = LR                 # 🚨 제거
    args.val_split = VAL_SPLIT
    
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
    # (클래스 추론은 원본 PKL 파일로 수행해도 무방함)
    with open(args.input_pkl, 'rb') as f: 
        data = pickle.load(f)
    anns = data.get('annotations', [])
    labels = [a.get('label') for a in anns if 'label' in a]
    
    n_classes = None
    if labels:
        cnt = Counter(labels)
        # 클래스 레이블이 0부터 시작한다고 가정하고 최대값 + 1을 사용
        n_classes = max(cnt.keys()) + 1 if cnt else 0 
        print(f"[OK] Detected labels: {dict(cnt)}, n_classes={n_classes}")
    else:
        print("[WARN] No 'label' field found in annotations; will not override num_classes.")

    # ----------------------------------------------------------------------
    # 3. Config 파일 처리 및 임시 설정 생성
    # ----------------------------------------------------------------------
    
    cfg_path = os.path.join(MM_ROOT, args.cfg.strip())
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # 임시 finetune config 생성 (절대경로 base 상속)
    tmp_cfg_dir = 'tmp_cfg'
    os.makedirs(tmp_cfg_dir, exist_ok=True)
    finetune_cfg = os.path.join(tmp_cfg_dir, 'finetune_stgcn_cfg.py')
    # Windows 경로 문제를 해결하기 위해 슬래시(/)로 통일
    base_cfg_abs = cfg_path.replace('\\', '/')
    # 🚨 .env, .env.local 등의 변수를 따로 분리하라는 지침을 적용하여,
    # 🚨 절대 경로를 config에 하드코딩하지 않고 base 상속만 사용하며,
    # 🚨 필요한 값은 실행 시 --cfg-options로 전달하는 기존 방식을 유지합니다.
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
        
        # ⭐️ 트레이닝/검증 데이터로 최종 결정된 PKL 경로 사용
        f"train_dataloader.dataset.dataset.ann_file={final_pkl_path}",
        f"val_dataloader.dataset.ann_file={final_pkl_path}",
        
        # 기타 학습 설정
        f"train_dataloader.dataset.dataset.split=xsub_train",
        f"val_dataloader.dataset.split={args.val_split}",
        
        # 🚨 BATCH_SIZE, max_epochs, lr 오버라이드를 제거하고 Config 파일 값을 사용합니다.
    ]

    # Test PKL 경로가 주어진 경우 처리
    if args.test_pkl:
        if not os.path.isfile(args.test_pkl):
            raise FileNotFoundError(f"Test PKL not found: {args.test_pkl}")
        cmd.extend([f"test_dataloader.dataset.ann_file={args.test_pkl}", f"test_dataloader.dataset.split={args.val_split}"])

    # 클래스 수가 추론된 경우, 모델 헤드 업데이트
    if n_classes is not None:
        cmd.extend([f"model.cls_head.num_classes={n_classes}"])

    print("\n[RUNNING] ", ' '.join(cmd))
    
    try:
        # MMAction2 훈련 실행
        subprocess.run(cmd, check=True, env=env)
    finally:
        # 훈련 성공/실패와 관계없이 임시 파일 정리
        if temp_pkl_path_to_delete and os.path.exists(temp_pkl_path_to_delete):
            print(f"\n[CLEANUP] Deleting temporary shuffled PKL: {temp_pkl_path_to_delete}")
            os.remove(temp_pkl_path_to_delete)
            print("[CLEANUP] Cleanup complete.")


if __name__ == '__main__':
    main()