#!/usr/bin/env python
"""
finetune_stgcn.py

crop_keypoint/**/*.csv → combined annotation PKL 생성 및
MMAction2 tools/train.py를 이용한 ST-GCN 미세조정(fine-tune)
"""

import argparse
import os
import sys
import pickle
import subprocess

# OpenPose CSV→PKL 변환 함수 임포트
from openpose_to_ntu60 import convert_csv_to_pkl

# MMAction2 루트 경로 (tools/train.py 접근용)
MM_ROOT = r"D:\mmaction2"
sys.path.append(MM_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-root',   required=True,
                        help='balanced_true, false 폴더가 있는 최상위 경로')
    parser.add_argument('--output-pkl', default='combined_ntu.pkl',
                        help='생성할 combined annotation PKL 경로')
    parser.add_argument('--cfg',        required=True,
                        help='MMAction2 config.py 경로 (MMAction2 루트 기준)')
    parser.add_argument('--pretrained', required=True,
                        help='사전학습된 ST-GCN 체크포인트(.pth)')
    parser.add_argument('--work-dir',   default='work_dirs/finetune_stgcn',
                        help='MMAction2 --work-dir')
    parser.add_argument('--device',     default='cuda:0',
                        help='CUDA 디바이스 (ex: cuda:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch_size override')
    parser.add_argument('--epochs',     type=int, default=30,
                        help='max_epochs override')
    parser.add_argument('--lr',         type=float, default=0.01,
                        help='learning rate override')
    parser.add_argument('--val-split',  default='xsub_val',
                        help='validation split 이름')
    args = parser.parse_args()

    # config 파일 경로 확인
    cfg_path = os.path.join(MM_ROOT, args.cfg.strip())
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # 1) CSV 폴더 순회하여 샘플 리스트 구축
    samples = []
    for label, class_name in enumerate(['false', 'balanced_true']):
        folder = os.path.join(args.csv_root, class_name, 'crop_keypoint')
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.endswith('.csv'):
                tmp_pkl = os.path.join('tmp', f"{os.path.splitext(fn)[0]}.pkl")
                os.makedirs(os.path.dirname(tmp_pkl), exist_ok=True)
                data = convert_csv_to_pkl(
                    csv_path=os.path.join(folder, fn),
                    pkl_path=tmp_pkl,
                    frame_dir=os.path.splitext(fn)[0],
                    label=label,
                    img_shape=(1080, 1920),
                    normalize_method='0to1',
                    confidence_threshold=0.1,
                    interpolate=False
                )
                samples.extend(data['annotations'])

    # 2) combined annotation PKL 작성
    combined = {
        'split': {'xsub_train': [s['frame_dir'] for s in samples]},
        'annotations': samples
    }
    os.makedirs(os.path.dirname(args.output_pkl) or '.', exist_ok=True)
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(combined, f)
    print(f"[OK] combined annotation saved to {args.output_pkl}")

    # 3) 임시 finetune config 생성 (절대경로 base 상속)
    tmp_cfg_dir = 'tmp_cfg'
    os.makedirs(tmp_cfg_dir, exist_ok=True)
    finetune_cfg = os.path.join(tmp_cfg_dir, 'finetune_stgcn_cfg.py')
    base_cfg_abs = cfg_path.replace('\\', '/')
    with open(finetune_cfg, 'w', encoding='utf-8') as f:
        f.write(f"_base_ = ['{base_cfg_abs}']\n")
    print(f"[OK] generated finetune config: {finetune_cfg}")

    # 4) MMAction2 tools/train.py 실행 준비
    env = os.environ.copy()
    if args.device.startswith('cuda'):
        env['CUDA_VISIBLE_DEVICES'] = args.device.split(':', 1)[1]

    cmd = [
        sys.executable,
        os.path.join(MM_ROOT, 'tools', 'train.py'),
        finetune_cfg,
        '--work-dir', args.work_dir,
        '--cfg-options',
        f"load_from={args.pretrained}",
        f"train_dataloader.dataset.dataset.ann_file={args.output_pkl}",
        f"train_dataloader.dataset.dataset.split=xsub_train",
        f"val_dataloader.dataset.ann_file={args.output_pkl}",
        f"val_dataloader.dataset.split={args.val_split}",
        f"train_dataloader.batch_size={args.batch_size}",
        f"train_cfg.max_epochs={args.epochs}",
        f"optim_wrapper.optimizer.lr={args.lr}"
    ]

    print("\n[RUNNING] ", ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    main()
