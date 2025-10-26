# app/routers/result_router.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

# websocket manager
from app.websocket_manager import manager   # manager는 app.websocket_manager에서
from app.webhook_auth import verify_runpod_signature

# 추가: S3 / DB 클라이언트 임포트
from app.s3_client import S3Client
from app.db_client import DBClient

router = APIRouter(
    prefix="/result",
    tags=["analysis_result"]
)

s3_client = S3Client()
db_client = DBClient()

# ----------------------------------------------------
# 1. WebSocket 연결 엔드포인트: /result/ws/analysis/{job_id}
# ----------------------------------------------------
# 지금은 안씀, 혹시나 싶어서 남겨둠
@router.websocket("/ws/analysis/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    # 기존 경로: job_id를 경로에서 받아 즉시 등록하는 경우
    await websocket.accept()
    await manager.connect(job_id, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(job_id)
        print(f"[WS] {job_id} 연결 해제됨.")
    except Exception as e:
        manager.disconnect(job_id)
        print(f"[WS] {job_id} 처리 중 오류: {e}")


@router.websocket("/ws/analysis")
async def websocket_preconnect(websocket: WebSocket):
    """
    Preconnect endpoint (주석 참고):
    - FE가 먼저 WebSocket을 엽니다.
      권장: subprotocol으로 "Bearer <JWT>" 전달 (new WebSocket(url, ['Bearer <jwt>']))
      대안: ?token=<jwt> 쿼리 파라미터 (덜 안전)
    - FE가 연결된 후 텍스트 메시지로 등록 요청을 보냅니다:
      {"action":"register", "job_id":"<JOB_ID>"}
    - 서버는 토큰을 검증해 user_id를 얻고, DB에서 job의 owner를 조회해 일치할 때만 연결을 등록합니다.
    """

    # 1) WS 업그레이드 수락: 이제 서버-클라이언트 간 메시지 송수신 가능
    await websocket.accept()

    temp_id = None  # 등록된 job_id를 임시로 저장 (연결 해제 시 정리용)
    try:
        # 2) 토큰 추출 우선순위:
        #    1) Sec-WebSocket-Protocol(subprotocol) - 권장: 'Bearer <JWT>'
        #    2) ?token=<jwt> 쿼리 파라미터
        #    3) HttpOnly 쿠키 (id_token 또는 access_token) - 현재 로그인 흐름에서 token_router가 HttpOnly 쿠키로 설정
        #       브라우저는 동일 출처(relative URL 사용)로 WebSocket 업그레이드시 쿠키를 자동으로 전송하므로
        #       클라이언트에서 토큰을 JS로 노출하지 않아도 인증을 수행할 수 있습니다.
        proto = websocket.headers.get("sec-websocket-protocol")
        token = None
        if proto:
            # 여러 서브프로토콜을 보낼 수 있으므로, 첫 항목을 사용합니다.
            token = proto.split(",")[0].strip()
        if not token:
            # subprotocol이 없다면 query param에서 꺼내봅니다: ws://host/ws/analysis?token=<jwt>
            token = websocket.query_params.get("token")
        if not token:
            # 마지막으로 HttpOnly 쿠키에 저장된 id_token/access_token을 확인
            # (token_router.py는 로그인 시 HttpOnly 쿠키로 id_token/access_token을 설정합니다)
            try:
                cookie_token = websocket.cookies.get("id_token") or websocket.cookies.get("access_token")
                if cookie_token:
                    token = cookie_token
            except Exception:
                # websocket.cookies 속성이 없거나 예외 발생 시 무시
                token = None

        # 3) FE가 보낸 첫 텍스트 메시지를 읽음 (이 메시지에서 register 요청을 기대)
        #    예: '{"action":"register","job_id":"abcd-1234"}'
        msg = await websocket.receive_text()
        try:
            import json
            payload = json.loads(msg)
        except Exception:
            # 메시지가 JSON이 아니면 오류 응답 후 연결 종료
            await websocket.send_json({"error": "invalid_json"})
            await websocket.close(code=1003)  # 1003: unsupported data
            return

        # 4) 메시지 내용 검사: action이 register이고 job_id가 있어야 함
        action = payload.get("action")
        if action == "register" and payload.get("job_id"):
            # FE가 올바른 등록 요청을 보냈다 -> 처리 진행
            job_id = payload.get("job_id")
            temp_id = job_id  # 추후 연결 해제 시 매핑을 지우기 위해 저장

            # 5) 토큰 검증: raw token -> user_id 추출
            #    (auth_utils.get_user_id_from_token는 JWT 서명/claims를 검증하여 sub(user id)를 반환)
            from auth_utils import get_user_id_from_token
            user_id = get_user_id_from_token(token) if token else None

            # 6) DB에서 job의 소유자(owner)를 조회
            #    (여기서 owner가 없으면 job이 아직 DB에 기록되지 않았거나 job_id가 잘못된 것)
            owner_id = db_client.get_job_owner(job_id)
            if owner_id is None:
                # owner가 없다는 것은 job 레코드가 존재하지 않음 -> 등록 실패
                # FE에 에러 응답 후 연결 종료
                await websocket.send_json({"error": "job_not_found"})
                await websocket.close(code=1008)  # 1008: policy violation (사용자 정의로 사용)
                return

            # 7) owner와 토큰에서 얻은 user_id 비교 (권한 검증)
            #    - user_id가 없거나 owner와 불일치면 등록 거부
            if user_id is None or str(user_id) != str(owner_id):
                # 권한 없음
                await websocket.send_json({"error": "forbidden"})
                await websocket.close(code=1008)
                return

            # 8) 검증 통과: manager에 job_id와 websocket(user_id 포함) 등록
            #    이후 runpod/webhook 처리 시 manager.send_result_to_client(job_id, url)로 푸시 가능
            await manager.connect(job_id, websocket, user_id)
            # FE에 등록 성공 메시지 전송
            await websocket.send_json({"status": "registered", "job_id": job_id})

            # 9) 연결 유지 루프: 여기서는 클라이언트가 보내는 메시지를 계속 수신하여
            #    연결이 살아있는 한 루프를 유지합니다. (client가 끊으면 WebSocketDisconnect 발생)
            while True:
                await websocket.receive_text()

        else:
            # 메시지 포맷이 예상과 다르면 에러 응답 및 연결 종료
            await websocket.send_json({"error": "expected register with job_id"})
            await websocket.close(code=1003)
    except WebSocketDisconnect:
        # 10) 클라이언트가 연결을 끊으면 매핑을 정리
        if temp_id:
            manager.disconnect(temp_id)
    except Exception as e:
        # 11) 기타 예외: 매핑 정리 후 로그
        if temp_id:
            manager.disconnect(temp_id)
        print(f"[WS preconnect error]: {e}")


# ----------------------------------------------------
# 2. Webhook 알림 엔드포인트: /result/webhook/job/complete (RunPod 전용)
# ----------------------------------------------------
class WebhookData(BaseModel):
    job_id: str
    s3_result_path: str

# RunPod에서 작업 완료 시 호출하는 Webhook 엔드포인트
@router.post("/webhook/job/complete", status_code=202) # 202 Accepted
async def handle_webhook(request: Request):
    # 0) 서명/타임스탬프 검증 (예외 발생 시 401)
    await verify_runpod_signature(request)

    # 1) 본문 파싱 (검증 후에 수행)
    data = await request.json()
    job_id = data.get("job_id")
    s3_key = data.get("s3_result_path") or data.get("s3_result_key")
    if not job_id or not s3_key:
        raise HTTPException(status_code=400, detail="job_id and s3_result_path required")

    # 2) DB 업데이트: PROCESSING_DONE (RunPod 완료)
    try:
        db_client.update_upload_status(job_id=job_id, status="PROCESSING_DONE", s3_result_path=s3_key)
    except Exception as e:
        print(f"[DB Update Error] job_id={job_id} error={e}")

    # 3) presigned GET 생성
    try:
        presigned_url = s3_client.create_presigned_get_url(s3_key)
    except Exception as e:
        print(f"[S3 Presign Error] key={s3_key} error={e}")
        raise HTTPException(status_code=500, detail="Failed to generate presigned result URL")

    # 4) 상태를 GETTING으로 바꾸고 WS로 푸시
    try:
        db_client.update_upload_status(job_id=job_id, status="GETTING")
    except Exception as e:
        print(f"[DB Update Warning] Failed to set GETTING for job_id={job_id}: {e}")

    try:
        pushed = await manager.send_result_to_client(job_id, presigned_url)
    except Exception as e:
        print(f"[WS Push Exception] job_id={job_id} error={e}")
        pushed = False

    if pushed:
        return {"message": "Result pushed successfully.", "result_url": presigned_url}
    else:
        return {"message": "Client not active. Result URL generated.", "result_url": presigned_url}