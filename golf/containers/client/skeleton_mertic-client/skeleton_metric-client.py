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


def call_openpose_api_for_s3key(s3_key: str, mime_type: str = None, turbo_without_skeleton: bool = False):
    """Call the server API by providing an S3 key only (no base64 payload).

    The server is expected to download inputs from S3 itself and process them.
    """
    payload = {
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
def send_request_for_s3key(s3_key: str, dimension: str, api_url: str = None, turbo: bool = False):
    """Send a lightweight request to the server that points it at an S3 key.

    This client no longer uploads video data — the server downloads from S3 and writes
    results back to S3. The client only validates the s3_key and requests processing.
    """
    # validate s3_key basic structure: expect at least two segments and the dimension as second
    if not s3_key or '/' not in s3_key:
        print(f"[ERROR] s3_key 형식이 잘못되었습니다. 기대형식: <user_id>/<dimension>/<...>. 받은 값: {s3_key}")
        return None

    parts = s3_key.strip('/').split('/')
    if len(parts) < 2:
        print(f"[ERROR] s3_key에 dimension 세그먼트가 없습니다: {s3_key}")
        return None

    key_dimension = parts[1]
    if key_dimension.lower() != dimension.lower():
        print(f"[ERROR] s3_key의 dimension({key_dimension})이 요청한 dimension({dimension})과 일치하지 않습니다.")
        return None

    # optional: check file extension matches dimension expectations
    fname = parts[-1]
    if dimension.lower() == '2d' and not fname.lower().endswith('.mp4'):
        print(f"[WARN] 2d 처리 예상 MP4 파일이 아닙니다: {fname}")
    if dimension.lower() == '3d' and not (fname.lower().endswith('.zip') or fname.lower().endswith('.tar') or fname.lower().endswith('.tar.gz')):
        print(f"[WARN] 3d 처리 예상 ZIP/TAR 파일이 아닙니다: {fname}")

    # decide mime_type
    mime_type = 'video/mp4' if dimension.lower() == '2d' else 'application/zip'

    # call server (no base64 payload)
    prev_api = globals().get('API_URL')
    if api_url:
        globals()['API_URL'] = api_url
    print(f"[INFO] 요청 전송: s3_key={s3_key}, dimension={dimension}, api={globals().get('API_URL')}")
    resp = call_openpose_api_for_s3key(s3_key, mime_type=mime_type, turbo_without_skeleton=turbo)

    # restore API_URL if we temporarily set it
    if api_url:
        globals()['API_URL'] = prev_api

    if resp is None:
        print("[ERROR] 서버가 응답하지 않았습니다.")
        return None

    # The server uploads results to S3; the client only prints server reply (job id / status)
    print(f"[INFO] 서버 응답: {json.dumps(resp, ensure_ascii=False)}")
    return resp


# ================= CLI =================
def main():
    global API_URL
    parser = argparse.ArgumentParser(description="S3에 업로드된 입력을 서버가 직접 처리하도록 s3_key를 전달합니다.")
    parser.add_argument("--s3-key", type=str, required=True, help="S3 객체 키 예: <user_id>/<dimension>/<job_id>/input.mp4")
    parser.add_argument("--dimension", type=str, choices=['2d', '3d'], required=True, help="처리할 스켈레톤 차원")
    parser.add_argument("--api-url", type=str, default=API_URL, help="API 엔드포인트 URL (기본: %(default)s)")
    parser.add_argument("--turbo", action="store_true", help="turbo_without_skeleton 모드 사용")

    args = parser.parse_args()

    API_URL = args.api_url
    s3_key = args.s3_key
    dimension = args.dimension
    send_request_for_s3key(s3_key, dimension, api_url=API_URL, turbo=args.turbo)

if __name__ == "__main__":
    main()

# 입력 명령어 예시
# python skeleton_metric-client.py --s3-key c4282d5c-5091-7071-aa3d-6f5d97e2fd8e/3d/d70b7ead-4e14-48f5-a92c-11698399ca3b.zip --dimension 3d --api-url http://localhost:19030/skeleton_metric_predict