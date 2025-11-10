#!/usr/bin/env python
"""
finetune_stgcn.py

ST-GCN fine-tuning using an existing combined annotation PKL
and MMAction2 tools/train.py.
"""

import argparse
import os
import sys
import pickle
import subprocess

# MMAction2 루트 경로 (tools/train.py 접근용)
MM_ROOT = r"D:\mmaction2"
sys.path.append(MM_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-pkl', required=True,
                        help='통합 annotation PKL 파일 경로')
    parser.add_argument('--cfg',        required=True,
                        help='MMAction2 config.py 경로 (MMAction2 루트 기준)')
    parser.add_argument('--pretrained', required=True,
                        help='사전학습된 ST-GCN 체크포인트(.pth)')
    parser.add_argument('--test-pkl',   required=False, default='',
                        help='(optional) test annotation PKL 경로 (omit to skip test override)')
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

    # 입력으로 주어진 PKL 파일(통합 annotation)이 존재하는지 확인
    if not os.path.isfile(args.input_pkl):
        raise FileNotFoundError(f"Annotation PKL not found: {args.input_pkl}")
    print(f"[OK] Using annotation PKL: {args.input_pkl}")

    # infer number of classes from PKL (if labels exist) and show distribution
    with open(args.input_pkl, 'rb') as f:
        data = pickle.load(f)
    anns = data.get('annotations', [])
    labels = [a.get('label') for a in anns if 'label' in a]
    if labels:
        from collections import Counter
        cnt = Counter(labels)
        n_classes = max(cnt.keys()) + 1
        print(f"[OK] Detected labels: {dict(cnt)}, n_classes={n_classes}")
    else:
        n_classes = None
        print("[WARN] No 'label' field found in annotations; will not override num_classes.")

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
        # Avoid setting global load_from (would load full model including head).
        # Instead set the backbone init_cfg checkpoint so only the backbone is initialized.
        f"model.backbone.init_cfg.checkpoint={args.pretrained}",
        f"train_dataloader.dataset.dataset.ann_file={args.input_pkl}",
        f"train_dataloader.dataset.dataset.split=xsub_train",
        f"val_dataloader.dataset.ann_file={args.input_pkl}",
        f"val_dataloader.dataset.split={args.val_split}",
        f"train_dataloader.batch_size={args.batch_size}",
        f"train_cfg.max_epochs={args.epochs}",
        f"optim_wrapper.optimizer.lr={args.lr}"
    ]

    # if user provided a test PKL, override test dataloader ann_file
    if args.test_pkl:
        if not os.path.isfile(args.test_pkl):
            raise FileNotFoundError(f"Test PKL not found: {args.test_pkl}")
        cmd.extend([f"test_dataloader.dataset.ann_file={args.test_pkl}", f"test_dataloader.dataset.split={args.val_split}"])

    # if we inferred n_classes from the PKL, override model head num_classes to match
    if n_classes is not None:
        cmd.extend([f"model.cls_head.num_classes={n_classes}"])

    print("\n[RUNNING] ", ' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    main()
