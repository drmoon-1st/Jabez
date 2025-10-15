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

def call_openpose_api(image_base64: str, turbo_without_skeleton: bool = False):
    """OpenPose Docker API에 POST 요청을 보내고 JSON 응답을 받습니다.

    turbo_without_skeleton=True 로 설정하면 처리 속도는 빨라지지만 키포인트를
    바로 반환하지 않는 래퍼가 있을 수 있습니다. 기본은 False (키포인트 요청).
    """
    payload = {
        "img": image_base64,
        "turbo_without_skeleton": bool(turbo_without_skeleton)
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


def extract_keypoints_from_response(data: dict):
    """API 응답에서 키포인트 목록을 안전하게 추출합니다.

    여러 OpenPose 래퍼/서비스가 서로 다른 필드명을 사용합니다. 이 함수는
    가능한 필드들을 순서대로 검사하여 (사람 수, 각 사람의 키포인트 리스트) 형태로 반환합니다.
    반환값은 list 형식(사람별 키포인트 리스트) 또는 None입니다.
    """
    if not isinstance(data, dict):
        return None

    # 공통적으로 사용되는 키들
    # 1) 'people' (OpenPose Python API, 여러 프로젝트에서 사용)
    if 'people' in data and isinstance(data['people'], list):
        ppl = data['people']
        # sometimes each person is a dict like {'pose_keypoints_2d': [...]}
        normalized = []
        for p in ppl:
            if isinstance(p, dict):
                if 'pose_keypoints_2d' in p and isinstance(p['pose_keypoints_2d'], list):
                    normalized.append(p['pose_keypoints_2d'])
                elif 'pose_keypoints' in p and isinstance(p['pose_keypoints'], list):
                    normalized.append(p['pose_keypoints'])
                elif 'keypoints' in p and isinstance(p['keypoints'], list):
                    normalized.append(p['keypoints'])
                else:
                    # unknown dict shape, try to append raw dict (caller will handle)
                    normalized.append(p)
            else:
                normalized.append(p)
        return normalized

    # 2) 'pose_keypoints_2d' 또는 'pose_keypoints' : 각 사람별로 배열이 있을 때
    if 'pose_keypoints_2d' in data and isinstance(data['pose_keypoints_2d'], list):
        return [data['pose_keypoints_2d']]
    if 'pose_keypoints' in data and isinstance(data['pose_keypoints'], list):
        return [data['pose_keypoints']]

    # also accept 'keypoints' as a top-level alias
    if 'keypoints' in data and isinstance(data['keypoints'], list):
        return [data['keypoints']]

    # 3) 'pose_id' 같은 필드를 사용하는 경우(원래 코드가 기대했던 형태)
    #    하지만 probe 결과처럼 정수(예: 0)만 반환되는 경우도 있음 -> 그런 경우는 키포인트가 없음
    if 'pose_id' in data:
        v = data['pose_id']
        # 리스트로 실제 키포인트를 담고 있는 경우
        if isinstance(v, list):
            return v
        # 정수/문자열 같은 값만 있으면 키포인트 없음
        return None

    # 4) 일부 커스텀 API는 사람별 키포인트를 'pose_0', 'pose_1' 등으로 반환할 수 있음
    poses = []
    for k, val in data.items():
        if k.startswith('pose_') and isinstance(val, list):
            poses.append(val)
    if poses:
        return poses

    return None


def try_retrieve_result_by_id(pose_id, timeout=5, interval=0.5):
    """If the server returns a job id, try common retrieval endpoints to get the full result.

    This tries a few reasonable URL patterns for backwards compatibility with async wrappers.
    Returns the parsed JSON dict if successful, otherwise None.
    """
    if pose_id is None:
        return None

    base = API_URL.rstrip('/')
    candidate_urls = [
        f"{base}/result/{pose_id}",
        f"{base}/openpose_result/{pose_id}",
        f"{base}/get_result/{pose_id}",
        f"{base}/result?id={pose_id}",
        f"{base}/get_result?id={pose_id}",
    ]

    # also a POST pattern
    candidate_posts = [
        (f"{base}/get", {'pose_id': pose_id}),
        (f"{base}/get_result", {'pose_id': pose_id}),
        (f"{base}/openpose_result", {'pose_id': pose_id}),
    ]

    # To avoid infinite loops against unreliable servers, use a combination
    # of a hard deadline and a max attempts cap with exponential backoff.
    deadline = time.time() + float(timeout)
    last_exc = None
    max_attempts = max(3, int(timeout / max(0.1, interval)))  # reasonable lower bound
    attempt = 0
    backoff = float(interval)

    while time.time() < deadline and attempt < max_attempts:
        attempt += 1
        # try GET endpoints first
        for u in candidate_urls:
            try:
                r = requests.get(u, timeout=5)
                if r.status_code != 200:
                    continue
                try:
                    j = r.json()
                    if isinstance(j, dict) and j:
                        return j
                except Exception:
                    # not JSON, skip
                    continue
            except Exception as e:
                last_exc = e

        # then try POST endpoints
        for (u, body) in candidate_posts:
            try:
                r = requests.post(u, json=body, timeout=5)
                if r.status_code != 200:
                    continue
                try:
                    j = r.json()
                    if isinstance(j, dict) and j:
                        return j
                except Exception:
                    continue
            except Exception as e:
                last_exc = e

        # exponential backoff but cap it to avoid very long sleeps
        time.sleep(min(backoff, 5.0))
        backoff *= 1.8

    if last_exc is not None:
        print(f"[WARN] result retrieval attempts finished (attempts={attempt}) with last error: {last_exc}")
    else:
        print(f"[INFO] result retrieval attempts finished (attempts={attempt}) without JSON result")
    return None

# ================= 메인 처리 함수 =================
def process_images_to_2d_csv(image_dir: Path, output_dir: Path, turbo: bool = False):
    """이미지 디렉토리의 모든 프레임을 API로 처리하고 2D CSV를 생성합니다."""
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    if not image_files:
        print(f"[WARN] 처리할 이미지가 '{image_dir}'에 없습니다. 종료합니다.")
        return

    rows2d = []
    no_detections = 0
    # pending mapping: pose_id -> list of frame indices that reference it
    pending = {}
    
    for image_path in tqdm(image_files, desc="Processing Frames (API Call)", unit="fr"):
        try:
            # 1. Base64 인코딩
            encoded_img = encode_image_to_base64(image_path)
            
            # 2. API 호출 (turbo 모드 여부에 따라 키포인트 포함 여부가 달라질 수 있음)
            data = call_openpose_api(encoded_img, turbo_without_skeleton=turbo)

            if data is None:
                # API 호출 실패 시 해당 프레임은 NaN으로 채움
                rows2d.append([np.nan] * len(COLS_2D))
                continue
            # Save raw responses for debugging when output_dir is known
            try:
                # output_dir may be a Path; create raw_responses subdir lazily
                raw_dir = output_dir / "raw_responses"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_file = raw_dir / f"{image_path.stem}.json"
                with open(raw_file, 'w', encoding='utf-8') as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
            except Exception:
                # don't break processing on debug save failures
                pass
            # 새로 추가한 헬퍼로 다양한 응답 구조에서 키포인트를 추출
            people = extract_keypoints_from_response(data)

            # If response didn't contain immediate keypoints but returned a job id
            pid = data.get('pose_id', None)
            if (not people or not isinstance(people, list) or not people) and isinstance(pid, (int, str)):
                # store pending mapping to fill later
                pending.setdefault(str(pid), []).append(len(rows2d))
                # append placeholder; will replace later if results retrieved
                rows2d.append([np.nan] * len(COLS_2D))
                # small initial log
                if len(pending) <= 3:
                    print(f"[INFO] Received job id {pid} for frame {image_path.name}; will poll for result later")
                continue

            # 사람이 탐지되지 않거나, 키포인트가 없는 경우
            if not isinstance(people, list) or not people:
                no_detections += 1
                # Debug output: show keys & save raw JSON (already saved above)
                print(f"[DEBUG] No keypoints for frame {image_path.name}. Response keys: {list(data.keys())}")
                rows2d.append([np.nan] * len(COLS_2D))
                # 첫 몇 프레임에 대해 원인 파악용 로그를 남김
                if no_detections <= 3:
                    print(f"[INFO] No keypoints in frame {image_path.name}. Response keys: {list(data.keys())}")
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
    # If there are pending job ids and polling is enabled, try to retrieve them in batch
    # Guard against undefined globals and infinite loops by enforcing attempt limits.
    global POLL_ENABLED, POLL_TIMEOUT
    if 'POLL_ENABLED' not in globals():
        POLL_ENABLED = False
    if 'POLL_TIMEOUT' not in globals():
        POLL_TIMEOUT = 5.0

    if pending and POLL_ENABLED:
        print(f"[INFO] Polling for {len(pending)} pending job(s) with timeout={POLL_TIMEOUT}s...")
        deadline = time.time() + float(POLL_TIMEOUT)
        unresolved = set(pending.keys())
        # derive a sensible max attempts based on timeout and a min interval
        min_interval = 0.3
        max_attempts = max(3, int(POLL_TIMEOUT / min_interval))
        attempts = 0
        while unresolved and time.time() < deadline and attempts < max_attempts:
            attempts += 1
            for pid in list(unresolved):
                # allocate a small per-call timeout so we don't block too long
                retrieved = try_retrieve_result_by_id(pid, timeout=1.0, interval=min_interval)
                if retrieved is None:
                    # will try again until deadline or attempts exhausted
                    continue
                people = extract_keypoints_from_response(retrieved)
                if not isinstance(people, list) or not people:
                    continue
                # fill all frames that referenced this pid
                for idx in pending.get(pid, []):
                    try:
                        kps_raw = people[0]
                        kps = np.array(kps_raw).reshape(-1, 3)
                        kps_17 = kps[_IDX_MAP_18_TO_17, :] if kps.shape[0] >= 18 else kps
                        row_2d = []
                        for (x, y, c) in kps_17:
                            row_2d.extend([float(x), float(y), float(c)])
                        if len(row_2d) < len(COLS_2D):
                            row_2d.extend([np.nan] * (len(COLS_2D) - len(row_2d)))
                        rows2d[idx] = row_2d
                    except Exception as e:
                        print(f"[WARN] failed to fill result for pid={pid} frame_idx={idx}: {e}")
                unresolved.discard(pid)
            # short sleep between rounds, bounded so we respect the overall timeout
            time.sleep(min_interval)
        if unresolved:
            print(f"[WARN] Some job ids were not resolved within timeout or attempts: {unresolved}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out2d = output_dir / "skeleton2d.csv"
    pd.DataFrame(rows2d, columns=COLS_2D).to_csv(out2d, index=False)
    print(f"\n[SAVE] 최종 2D CSV 저장 완료: {out2d}")
    print(f"총 {len(rows2d)} 프레임 처리됨. ({no_detections} frames had no detections)")


# ================= CLI =================
def main():
    parser = argparse.ArgumentParser(description="Docker OpenPose API를 사용하여 이미지 디렉토리 → 2D CSV 변환")
    parser.add_argument("--image-dir", required=True, type=str, help="OpenPose로 처리할 이미지 파일들이 있는 폴더 (예: color/)")
    parser.add_argument("--output", required=True, type=str, help="2D CSV 파일이 저장될 출력 폴더")
    parser.add_argument("--turbo", action="store_true", help="요청을 turbo_without_skeleton 모드로 보냄(속도 우선, 일부 래퍼는 키포인트 비포함)")
    parser.add_argument("--poll", action="store_true", help="pose_id와 같은 식별자만 반환하는 비동기 래퍼에 대해 결과를 폴링 시도합니다")
    parser.add_argument("--poll-timeout", type=float, default=5.0, help="폴링 시 최대 대기 시간(초) 기본=5.0")

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output)
    
    # 예시: 'color' 폴더가 없으면 에러
    if not image_dir.is_dir():
        print(f"[FATAL] 이미지 디렉토리가 유효하지 않습니다: {image_dir}")
        return

    # 전역 폴링 플래그 설정 (간단히 전역 변수로 전달)
    global POLL_ENABLED, POLL_TIMEOUT
    POLL_ENABLED = bool(args.poll)
    POLL_TIMEOUT = float(args.poll_timeout)

    process_images_to_2d_csv(image_dir, output_dir, turbo=args.turbo)

if __name__ == "__main__":
    main()