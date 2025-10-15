import os
import json
import base64
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests

# ================= 사용자 환경 설정 =================
# OpenPose API 엔드포인트. WSL 환경에서 Docker 컨테이너가 19030 포트로 실행 중임을 가정.
API_URL = "http://localhost:19030/openpose_predict" 

# COCO17 타깃 컬럼 순서 (2D 키포인트만)
KP_17 = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]
COLS_2D = [f"{n}_{a}" for n in KP_17 for a in ("x", "y", "c")]

# OpenPose COCO 출력(18개, Neck 포함 가능) -> 위 KP_17 순서로 재배열
_IDX_MAP_18_TO_17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

# ================= 유틸 =================
def encode_image_to_base64(image_path: Path):
    """이미지 파일을 Base64 문자열로 인코딩합니다."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_openpose_api(image_base64: str):
    """OpenPose Docker API에 POST 요청을 보내고 JSON 응답을 받습니다."""
    payload = {
        "img": image_base64,
        "turbo_without_skeleton": True # 결과 이미지 대신 데이터 처리 속도 우선
    }
    headers = {'Content-Type': 'application/json'}
    
    # RunPod과 같이 시간이 오래 걸릴 수 있으므로 타임아웃을 넉넉하게 설정
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60) 
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] API 호출 실패: {e}")
        return None

# ================= 메인 처리 함수 =================
def process_images_to_2d_csv(image_dir: Path, output_dir: Path):
    """이미지 디렉토리의 모든 프레임을 API로 처리하고 2D CSV를 생성합니다."""
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    if not image_files:
        print(f"[WARN] 처리할 이미지가 '{image_dir}'에 없습니다. 종료합니다.")
        return

    rows2d = []
    
    for image_path in tqdm(image_files, desc="Processing Frames (API Call)", unit="fr"):
        try:
            # 1. Base64 인코딩
            encoded_img = encode_image_to_base64(image_path)
            
            # 2. API 호출
            data = call_openpose_api(encoded_img)
            
            if data is None:
                # API 호출 실패 시 해당 프레임은 NaN으로 채움
                rows2d.append([np.nan] * len(COLS_2D))
                continue

            people = data.get("pose_id", []) # API에 따라 'pose_id'에 키포인트 데이터가 들어있다고 가정
            
            # API 응답에서 키포인트 데이터를 찾고, 형식을 검사
            if not isinstance(people, list) or not people:
                 # 사람이 탐지되지 않거나, 데이터 형식이 잘못된 경우
                rows2d.append([np.nan] * len(COLS_2D))
                continue

            # API는 이미지를 처리하고 키포인트 목록을 반환해야 함.
            # 이 OpenPose API는 처리된 이미지도 반환하지만, 여기서는 키포인트만 추출합니다.
            
            # 첫 번째 사람의 키포인트만 사용 (max_people=1 가정)
            kps_raw = people[0] 
            kps = np.array(kps_raw).reshape(-1, 3) # (N, 3) 형태 (x, y, c)
            
            # 18개 키포인트를 17개(COCO17)로 재배열
            kps_17 = kps[_IDX_MAP_18_TO_17, :] if kps.shape[0] >= 18 else kps 

            row_2d = []
            for (x, y, c) in kps_17:
                row_2d.extend([float(x), float(y), float(c)])

            # 컬럼 수 맞추기
            if len(row_2d) < len(COLS_2D):
                row_2d.extend([np.nan] * (len(COLS_2D) - len(row_2d)))

            rows2d.append(row_2d)

        except Exception as e:
            print(f"\n[ERROR] 프레임 {image_path.name} 처리 중 예외 발생: {e}")
            rows2d.append([np.nan] * len(COLS_2D)) # 오류 시에도 NaN 추가하여 행렬 유지
            
    # 3. CSV로 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    out2d = output_dir / "skeleton2d.csv"
    pd.DataFrame(rows2d, columns=COLS_2D).to_csv(out2d, index=False)
    print(f"\n[SAVE] 최종 2D CSV 저장 완료: {out2d}")
    print(f"총 {len(rows2d)} 프레임 처리됨.")


# ================= CLI =================
def main():
    parser = argparse.ArgumentParser(description="Docker OpenPose API를 사용하여 이미지 디렉토리 → 2D CSV 변환")
    parser.add_argument("--image-dir", required=True, type=str, help="OpenPose로 처리할 이미지 파일들이 있는 폴더 (예: color/)")
    parser.add_argument("--output", required=True, type=str, help="2D CSV 파일이 저장될 출력 폴더")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output)
    
    # 예시: 'color' 폴더가 없으면 에러
    if not image_dir.is_dir():
        print(f"[FATAL] 이미지 디렉토리가 유효하지 않습니다: {image_dir}")
        return

    process_images_to_2d_csv(image_dir, output_dir)

if __name__ == "__main__":
    main()