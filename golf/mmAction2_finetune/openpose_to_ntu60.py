#!/usr/bin/env python
"""
개선된 OpenPose BODY25 → NTU-RGB+D 변환기
좌표 정규화, 신뢰도 필터링, 스켈레톤 중심화 등을 추가
"""
import argparse
import os
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd

# BODY_25 → COCO(17) 매핑 (검증 필요)
MAPPING_BODY25_TO_COCO17 = [
    0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11
]

def normalize_coordinates(keypoints, img_shape, method='0to1'):
    """
    좌표 정규화
    Args:
        keypoints: (M, T, V, 2) 형태의 키포인트
        img_shape: (height, width)
        method: '0to1' or 'center' or 'skeleton_center'
    """
    height, width = img_shape
    normalized_kp = keypoints.copy()
    
    if method == '0to1':
        # [0, 1] 범위로 정규화
        normalized_kp[:, :, :, 0] = keypoints[:, :, :, 0] / width
        normalized_kp[:, :, :, 1] = keypoints[:, :, :, 1] / height
        
    elif method == 'center':
        # [-1, 1] 범위로 중심 기준 정규화
        normalized_kp[:, :, :, 0] = (keypoints[:, :, :, 0] - width/2) / (width/2)
        normalized_kp[:, :, :, 1] = (keypoints[:, :, :, 1] - height/2) / (height/2)
        
    elif method == 'skeleton_center':
        # 스켈레톤 중심 기준 정규화
        for m in range(keypoints.shape[0]):
            for t in range(keypoints.shape[1]):
                valid_mask = (keypoints[m, t, :, 0] != 0) & (keypoints[m, t, :, 1] != 0)
                if valid_mask.any():
                    valid_kp = keypoints[m, t, valid_mask, :]
                    center_x = valid_kp[:, 0].mean()
                    center_y = valid_kp[:, 1].mean()
                    
                    # 스켈레톤 크기 (바운딩 박스 기준)
                    bbox_w = valid_kp[:, 0].max() - valid_kp[:, 0].min()
                    bbox_h = valid_kp[:, 1].max() - valid_kp[:, 1].min()
                    scale = max(bbox_w, bbox_h)
                    
                    if scale > 0:
                        normalized_kp[m, t, :, 0] = (keypoints[m, t, :, 0] - center_x) / scale
                        normalized_kp[m, t, :, 1] = (keypoints[m, t, :, 1] - center_y) / scale
    
    return normalized_kp

def filter_by_confidence(keypoints, scores, threshold=0.1):
    """신뢰도 기반 키포인트 필터링"""
    filtered_kp = keypoints.copy()
    filtered_scores = scores.copy()
    
    # 신뢰도가 낮은 키포인트를 0으로 설정
    low_confidence_mask = scores < threshold
    filtered_kp[low_confidence_mask] = 0
    filtered_scores[low_confidence_mask] = 0
    
    return filtered_kp, filtered_scores

def interpolate_missing_keypoints(keypoints, scores, method='linear'):
    """누락된 키포인트 보간"""
    interpolated_kp = keypoints.copy()
    
    for m in range(keypoints.shape[0]):
        for v in range(keypoints.shape[2]):
            for c in range(keypoints.shape[3]):
                # 해당 키포인트의 시간 시리즈
                series = keypoints[m, :, v, c]
                valid_mask = series != 0
                
                if valid_mask.any() and not valid_mask.all():
                    # 선형 보간
                    valid_indices = np.where(valid_mask)[0]
                    if len(valid_indices) > 1:
                        interpolated_values = np.interp(
                            np.arange(len(series)),
                            valid_indices,
                            series[valid_indices]
                        )
                        # 기존 유효한 값은 유지, 무효한 값만 보간
                        interpolated_kp[m, ~valid_mask, v, c] = interpolated_values[~valid_mask]
    
    return interpolated_kp

def convert_csv_to_pkl(csv_path, pkl_path, frame_dir,
                       label=0, img_shape=(1080, 1920),
                       normalize_method='0to1',
                       confidence_threshold=0.1,
                       interpolate=False):
    """개선된 CSV → PKL 변환"""
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[CSV→PKL] CSV not found: {csv_path}")
    
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if df.shape[1] != 25 * 3:
        raise ValueError(f"[CSV→PKL] Expect 25*3 columns, got {df.shape[1]}")
    
    T = len(df)
    V25 = 25
    M, C = 1, 2
    
    print(f"[INFO] Processing {T} frames with {V25} keypoints")
    
    # 1. 원본 데이터 로드
    kp25 = np.zeros((M, T, V25, C), dtype=np.float32)
    score25 = np.zeros((M, T, V25), dtype=np.float32)
    
    for t, row in df.iterrows():
        for v in range(V25):
            x, y, s = row[3*v:3*v+3]
            kp25[0, t, v, 0] = x
            kp25[0, t, v, 1] = y
            score25[0, t, v] = s
    
    print(f"[INFO] Original coordinate range: X[{kp25[:,:,:,0].min():.1f}, {kp25[:,:,:,0].max():.1f}], Y[{kp25[:,:,:,1].min():.1f}, {kp25[:,:,:,1].max():.1f}]")
    
    # 2. 신뢰도 필터링
    if confidence_threshold > 0:
        kp25, score25 = filter_by_confidence(kp25, score25, confidence_threshold)
        print(f"[INFO] Applied confidence filtering (threshold: {confidence_threshold})")
    
    # 3. 키포인트 보간
    if interpolate:
        kp25 = interpolate_missing_keypoints(kp25, score25)
        print(f"[INFO] Applied keypoint interpolation")
    
    # 4. 좌표 정규화
    kp25 = normalize_coordinates(kp25, img_shape, normalize_method)
    print(f"[INFO] Applied coordinate normalization ({normalize_method})")
    print(f"[INFO] Normalized coordinate range: X[{kp25[:,:,:,0].min():.4f}, {kp25[:,:,:,0].max():.4f}], Y[{kp25[:,:,:,1].min():.4f}, {kp25[:,:,:,1].max():.4f}]")
    
    # 5. BODY25 → COCO17 변환
    kp17 = kp25[:, :, MAPPING_BODY25_TO_COCO17, :]
    score17 = score25[:, :, MAPPING_BODY25_TO_COCO17]
    
    print(f"[INFO] Converted to COCO17 format: {kp17.shape}")
    
    # 6. NTU 형식으로 패키징
    sample = {
        'frame_dir': frame_dir,
        'label': label,
        'img_shape': img_shape,
        'original_shape': img_shape,
        'total_frames': T,
        'keypoint': kp17,
        'keypoint_score': score17
    }
    
    data = {
        'split': {'xsub_val': [frame_dir]},
        'annotations': [sample]
    }
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"[SUCCESS] Wrote PKL: {pkl_path} (T={T}, V=17, M={M}, C={C})")
    
    # 7. 변환 결과 요약
    print(f"\n=== Conversion Summary ===")
    print(f"Frames: {T}")
    print(f"Keypoints: {V25} → 17")
    print(f"Normalization: {normalize_method}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Interpolation: {interpolate}")
    
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Improved OpenPose BODY25→COCO17 PKL + ST-GCN++ Inference")
    
    # 기본 매개변수
    parser.add_argument('--csv', required=True, help='OpenPose BODY25 CSV path')
    parser.add_argument('--frame-dir', required=True, help='video ID (frame_dir)')
    parser.add_argument('--pkl', default='temp_skel.pkl', help='output PKL path')
    parser.add_argument('--label', type=int, default=0, help='dummy label')
    parser.add_argument('--img-shape', nargs=2, type=int, default=[1080,1920], help='H W')
    
    # 개선된 매개변수
    parser.add_argument('--normalize', default='0to1', 
                       choices=['0to1', 'center', 'skeleton_center'],
                       help='coordinate normalization method')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                       help='confidence threshold for filtering keypoints')
    parser.add_argument('--interpolate', action='store_true',
                       help='interpolate missing keypoints')
    
    # 추론 매개변수
    parser.add_argument('--cfg', required=True, help='MMAction2 config.py')
    parser.add_argument('--ckpt', required=True, help='pretrained .pth')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--dump', default='result.pkl', help='path to dump results')
    
    args = parser.parse_args()
    
    try:
        convert_csv_to_pkl(
            csv_path=args.csv,
            pkl_path=args.pkl,
            frame_dir=args.frame_dir,
            label=args.label,
            img_shape=tuple(args.img_shape),
            normalize_method=args.normalize,
            confidence_threshold=args.confidence_threshold,
            interpolate=args.interpolate
        )
    except Exception as e:
        print(f"[ERROR] conversion failed:\n  {e}", file=sys.stderr)
        sys.exit(1)
    
    # MMAction2 추론 실행
    cmd = [
        sys.executable, 'tools/test.py',
        args.cfg, args.ckpt,
        '--dump', args.dump,
        '--cfg-options', f"test_dataloader.dataset.ann_file={args.pkl}",
        '--cfg-options', "test_dataloader.dataset.split=xsub_val"
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1] if args.device.startswith('cuda') else ''
    
    print(f"\n[RUNNING] {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, text=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"[ERROR] inference failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(proc.returncode)
    
    if os.path.isfile(args.dump):
        print(f"[DONE] inference succeeded, results dumped to {args.dump}")
    else:
        print(f"[WARN] inference succeeded but no dump found at {args.dump}")

if __name__ == '__main__':
    main()