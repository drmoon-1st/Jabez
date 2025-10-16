import requests
import json
import os
import time
import base64

# --- 설정 ---
# API 서버의 URL (호스트 포트 19031을 사용)
API_URL = "http://127.0.0.1:19031/mmaction_stgcn_embed"
# 실제 키포인트 데이터 파일 경로
TEMP_INPUT_CSV = "skeleton2d.csv"
# ----------------

def test_api_connection(file_path):
    """
    CSV 파일을 Base64로 인코딩하여 JSON 형태로 API 서버에 전송하고 임베딩을 수신합니다.
    """
    if not os.path.exists(file_path):
        print(f"❌ 테스트 파일이 존재하지 않습니다: {file_path}")
        print("    'skeleton2d.csv' 파일이 현재 스크립트와 같은 폴더에 있는지 확인하세요.")
        return

    print(f"\n🚀 API 서버 ({API_URL})에 JSON/Base64 요청 전송 중...")

    # 1. CSV 파일의 내용을 읽고 Base64 문자열로 인코딩
    try:
        with open(file_path, 'rb') as f:
            # 파일을 바이너리(rb)로 읽고 Base64 인코딩 후 UTF-8 문자열로 디코딩
            encoded_csv = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 파일 Base64 인코딩 오류: {e}")
        return

    # 2. 서버가 기대하는 JSON 페이로드 생성
    # 서버는 'csv_base64'라는 키로 Base64 문자열을 기대합니다.
    payload = {
        'csv_base64': encoded_csv
    }
    
    # 3. JSON 데이터 전송
    try:
        start_time = time.time()
        # requests의 'json=' 인자를 사용하여 JSON 데이터(payload)를 전송합니다.
        response = requests.post(API_URL, json=payload)
        end_time = time.time()
            
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: API 서버에 접속할 수 없습니다. 포트(19031) 또는 서버 상태를 확인하세요.")
        return
    except Exception as e:
        print(f"❌ 요청 중 알 수 없는 오류 발생: {e}")
        return
        
    # --- 응답 처리 ---
    if response.status_code == 200:
        data = response.json()
        embedding = data.get('embedding', [])
        dim = data.get('embedding_dim')
        
        print(f"\n✅ API 요청 성공 (처리 시간: {end_time - start_time:.2f}초)")
        print(f"   수신된 임베딩 차원: {dim}D")
        print(f"   임베딩 샘플 (앞 5개): {embedding[:5]}...")
        print("   API 서버가 임베딩을 성공적으로 반환했습니다.")
    else:
        print(f"\n❌ API 요청 실패 (HTTP 상태 코드: {response.status_code})")
        # 서버 로그 확인이 중요함.
        try:
            error_data = response.json()
            # 서버 오류 메시지를 'error' 키로 받도록 api_server.py가 설계됨
            print(f"   서버 오류 메시지: {error_data.get('error', response.text[:100])}")
        except json.JSONDecodeError:
            print(f"   서버 오류 메시지: {response.text[:100]}...")

# --- 실행 ---
if __name__ == "__main__":
    test_api_connection(TEMP_INPUT_CSV)