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
# OpenPose API 엔드포인트. 기본은 로컬 컨테이너의 skeleton_metric-api
API_URL = "http://localhost:19030/skeleton_metric_predict"  # 로컬 테스트용

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
def encode_file_to_base64(file_path: Path):
    """파일(이미지/비디오)을 Base64 문자열로 인코딩합니다."""
    with open(file_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode('utf-8')

def call_openpose_api_for_video(video_b64: str, s3_key: str = None, mime_type: str = "video/mp4", turbo_without_skeleton: bool = False):
    """비디오(베이스64)를 API에 전송하고 JSON 응답을 받습니다.

    payload 예시 호환성: 전송 페이로드에 'video' 필드(기본값)를 포함하고,
    S3 트래킹을 위해 's3_key'와 'mime_type'을 함께 보냅니다.
    """
    payload = {
        "video": video_b64,
        "turbo_without_skeleton": bool(turbo_without_skeleton),
    }
    if s3_key:
        payload["s3_key"] = s3_key
    if mime_type:
        payload["mime_type"] = mime_type

    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=180)
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
def process_video_to_2d_csv_from_s3key(s3_key: str, input_root: Path, output_dir: Path, turbo: bool = False):
    """S3 키(경로)를 이용해 input_root/s3_key의 mp4 파일을 읽어 API로 전송하고
    반환된 people_sequence를 skeleton2d.csv로 저장합니다.
    """
    file_path = input_root / s3_key
    if not file_path.exists():
        print(f"[FATAL] 입력 비디오 파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"[INFO] Sending video {file_path} -> API {API_URL}")
    video_b64 = encode_file_to_base64(file_path)
    data = call_openpose_api_for_video(video_b64, s3_key=s3_key, mime_type="video/mp4", turbo_without_skeleton=turbo)
    if data is None:
        print("[ERROR] API가 응답하지 않거나 실패했습니다.")
        return

    # Save raw response for debugging
    try:
        raw_dir = output_dir / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / (file_path.stem + ".json")
        with open(raw_file, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # API는 people_sequence와 frame_count를 반환하도록 설계됨
    people_sequence = data.get('people_sequence', None)
    frame_count = data.get('frame_count', None)

    # Fallback: if people_sequence absent but 'people' exists, try that
    if people_sequence is None and 'people' in data:
        people_sequence = data['people']

    if people_sequence is None and frame_count is None:
        print(f"[ERROR] API 응답에 people_sequence 또는 frame_count가 없습니다. 응답 키: {list(data.keys())}")
        return

    # If frame_count provided but people_sequence shorter, pad with empties
    if frame_count is None:
        frame_count = len(people_sequence)

    rows2d = []
    for i in range(int(frame_count)):
        frame_people = []
        if people_sequence and i < len(people_sequence):
            frame_people = people_sequence[i]

        if not frame_people:
            rows2d.append([np.nan] * len(COLS_2D))
            continue

        # use first person if list of people
        person = frame_people[0] if isinstance(frame_people, list) and len(frame_people) > 0 else frame_people

        # person may be flat list [x,y,c,...] or list of [x,y,c] lists
        try:
            p_arr = np.array(person)
            if p_arr.ndim == 1 and p_arr.size % 3 == 0:
                kps = p_arr.reshape(-1, 3)
            elif p_arr.ndim == 2 and p_arr.shape[1] == 3:
                kps = p_arr
            else:
                # unexpected shape
                rows2d.append([np.nan] * len(COLS_2D))
                continue
        except Exception:
            rows2d.append([np.nan] * len(COLS_2D))
            continue

        # reorder 18->17 if present
        kps_17 = kps[_IDX_MAP_18_TO_17, :] if kps.shape[0] >= 18 else kps
        row_2d = []
        for (x, y, c) in kps_17:
            row_2d.extend([float(x), float(y), float(c)])
        if len(row_2d) < len(COLS_2D):
            row_2d.extend([np.nan] * (len(COLS_2D) - len(row_2d)))
        rows2d.append(row_2d)

    output_dir.mkdir(parents=True, exist_ok=True)
    out2d = output_dir / "skeleton2d.csv"
    pd.DataFrame(rows2d, columns=COLS_2D).to_csv(out2d, index=False)
    print(f"\n[SAVE] 최종 2D CSV 저장 완료: {out2d} (frames={len(rows2d)})")


# ================= CLI =================
def main():
    global API_URL
    parser = argparse.ArgumentParser(description="S3 key로 지정된 MP4 파일을 API로 전송하여 skeleton2d.csv 생성")
    parser.add_argument("--s3-key", type=str, required=True, help="S3 객체 키 예: <job_id>/web_2d/filename.mp4")
    parser.add_argument("--input-root", type=str, default="input", help="로컬에서 S3 키가 위치한 루트 폴더 (기본: input)")
    parser.add_argument("--output", type=str, default="output", help="CSV가 저장될 출력 폴더 (기본: output)")
    parser.add_argument("--api-url", type=str, default=API_URL, help="API 엔드포인트 URL (기본: %(default)s)")
    parser.add_argument("--turbo", action="store_true", help="turbo_without_skeleton 모드 사용")

    args = parser.parse_args()

    s3_key = args.s3_key
    input_root = Path(args.input_root)
    output_dir = Path(args.output)

    process_video_to_2d_csv_from_s3key(s3_key, input_root, output_dir, turbo=args.turbo)

if __name__ == "__main__":
    main()