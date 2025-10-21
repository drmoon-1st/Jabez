import requests
import json
import os
import time
import base64

# --- 설정 ---
# API 서버의 URL (호스트 포트 19031을 사용)
# API_URL = "http://localhost:19031/mmaction_stgcn_test"
API_URL = "https://j0ixr5rvft4ccm-19031.proxy.runpod.net/mmaction_stgcn_test"
# 실제 키포인트 데이터 파일 경로
TEMP_INPUT_CSV = "skeleton2d.csv"
# ----------------

def test_api_connection(file_path):
    """
    CSV 파일을 Base64로 인코딩하여 JSON 형태로 API 서버에 전송하고 임베딩을 수신합니다.
    개선: timeout, 파일 크기 경고, 응답 JSON 예외 처리, 타입 검증 추가
    """
    if not os.path.exists(file_path):
        print(f"❌ 테스트 파일이 존재하지 않습니다: {file_path}")
        print("    'skeleton2d.csv' 파일이 현재 스크립트와 같은 폴더에 있는지 확인하세요.")
        return

    file_size = os.path.getsize(file_path)
    if file_size > 10 * 1024 * 1024:  # 10 MB 경고 기준 (조정 가능)
        print(f"⚠️ 경고: 파일 크기 {file_size / (1024*1024):.1f}MB — 메모리/전송에 문제가 있을 수 있습니다.")

    print(f"\n🚀 API 서버 ({API_URL})에 JSON/Base64 요청 전송 중...")

    try:
        with open(file_path, 'rb') as f:
            encoded_csv = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 파일 Base64 인코딩 오류: {e}")
        return

    payload = {'csv_base64': encoded_csv}

    try:
        start_time = time.time()
        # timeout: (connect, read) — 필요에 맞게 조정
        response = requests.post(API_URL, json=payload, timeout=(5, 60))    # 60초 읽기 타임아웃
        end_time = time.time()
    except requests.exceptions.Timeout:
        print("❌ 타임아웃: 서버 응답이 지연되었습니다.")
        return
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: API 서버에 접속할 수 없습니다. 포트(19031) 또는 서버 상태를 확인하세요.")
        return
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 중 오류 발생: {e}")
        return

    # HTTP 오류 처리
    if not response.ok:
        print(f"\n❌ API 요청 실패 (HTTP 상태 코드: {response.status_code})")
        text_preview = response.text[:1000]
        # JSON 형식일 수 있으니 안전하게 파싱 시도
        try:
            err = response.json()
            print(f"   서버 오류 메시지: {err.get('error', err)}")
        except Exception:
            print(f"   서버 응답(비JSON): {text_preview}...")
        return

    # 정상 응답: JSON 파싱 안전 처리
    try:
        data = response.json()
    except ValueError:
        print("❌ 응답 JSON 파싱 실패 — 서버가 JSON을 반환하지 않았습니다.")
        print(f"   응답 텍스트(미리보기): {response.text[:500]}...")
        return

    # 새 API 응답: 전체 추론 결과(result) 받기
    result = data.get('result')
    if result is None:
        print("❌ 응답에 'result' 필드가 없습니다. 전체 추론 결과를 반환하도록 서버를 확인하세요.")
        print(f"   전체 응답 미리보기: {json.dumps(data)[:1000]}")
        return

    print(f"\n✅ API 요청 성공 (처리 시간: {end_time - start_time:.2f}초)")
    print(f"   num_samples: {result.get('num_samples')}")
    preds = result.get('predictions', [])
    # server now returns a simple list of booleans in result['predictions']
    # Print first 3 and the full predictions as JSON so frontend can consume
    # the booleans directly (true/false without quotes).
    import json

    try:
        first3_json = json.dumps(preds[:3])
        full_json = json.dumps(preds)
    except Exception:
        # fallback to string representation if JSON serialization fails
        first3_json = str(preds[:3])
        full_json = str(preds)

    print(f"   predictions (first 3): {first3_json}")
    print(f"   predictions_json: {full_json}")
    if 'accuracy' in result:
        print(f"   accuracy: {result.get('accuracy'):.4f}")
    else:
        print("   accuracy: not provided (inference-only)")

# --- 실행 ---
if __name__ == "__main__":
    test_api_connection(TEMP_INPUT_CSV)