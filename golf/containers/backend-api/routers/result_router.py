# app/routers/result_router.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Optional

# 💡 websocket_manager 모듈 임포트
from websocket_manager import manager 
# 💡 DB, S3 클라이언트 (필요시 임포트)
# from db_client import DBClient 

router = APIRouter(
    prefix="/result", 
    tags=["analysis_result"]
)

# ----------------------------------------------------
# 1. WebSocket 연결 엔드포인트: /result/ws/analysis/{job_id}
# ----------------------------------------------------
@router.websocket("/ws/analysis/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    # 💡 1단계: 연결 등록 (WebSocket Accept 포함)
    await manager.connect(job_id, websocket)

    try:
        # 💡 연결 유지: 클라이언트 연결이 끊길 때까지 대기
        while True:
            await websocket.receive_text() # 메시지 수신 (연결 유지 목적)

    except WebSocketDisconnect:
        # 💡 연결 해제
        manager.disconnect(job_id)
        print(f"[WS] {job_id} 연결 해제됨.")
    except Exception as e:
        manager.disconnect(job_id)
        print(f"[WS] {job_id} 처리 중 오류: {e}")


# ----------------------------------------------------
# 2. Webhook 알림 엔드포인트: /result/webhook/job/complete (RunPod 전용)
# ----------------------------------------------------
class WebhookData(BaseModel):
    job_id: str
    s3_result_path: str

@router.post("/webhook/job/complete", status_code=202) # 202 Accepted는 비동기 작업 수락을 의미
async def handle_webhook(data: WebhookData):
    # 💡 9단계: RunPod Webhook 수신
    job_id = data.job_id
    
    # [주의] 8단계: 이전에 RunPod Worker는 이미 RDS에 COMPLETED 상태를 기록해야 합니다.

    # 💡 10단계: WebSocket 푸시 실행
    result_pushed = await manager.send_result_to_client(job_id, data.s3_result_path)
    
    if result_pushed:
        return {"message": "Result pushed successfully."}
    else:
        # 클라이언트 연결이 끊겼더라도, 결과는 이미 RDS에 있으므로 문제 없음
        return {"message": "Client not active, result saved in DB."}