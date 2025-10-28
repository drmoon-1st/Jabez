# app/websocket_manager.py

from typing import Dict, Optional
from fastapi import WebSocket

# 💡 운영 환경에서는 Redis 또는 RDBMS를 사용하여 이 상태를 관리해야 합니다.
# 현재는 개발 편의를 위해 메모리 딕셔너리를 사용합니다.
class ConnectionManager:
    def __init__(self):
        # Job ID (UUID)와 활성 WebSocket 연결 정보(웹소켓, user_id)를 매핑
        # value: {"websocket": WebSocket, "user_id": Optional[str]}
        self.active_connections: Dict[str, dict] = {}

    async def connect(self, job_id: str, websocket: WebSocket, user_id: Optional[str] = None):
        """연결 수락 및 등록

        Accepts optional user_id so callers that have validated the token
        can attach the owner id to the connection. This avoids signature
        mismatch when result_router passes user_id to connect().
        """
        # Note: endpoint should call websocket.accept() before invoking connect()
        # so we only register the websocket here.
        self.active_connections[job_id] = {"websocket": websocket, "user_id": user_id}

    def disconnect(self, job_id: str):
        """연결 해제"""
        if job_id in self.active_connections:
            del self.active_connections[job_id]

    async def send_result_to_client(self, job_id: str, result_url: str) -> bool:
        """특정 Job ID 클라이언트에게 결과 URL 푸시"""
        entry = self.active_connections.get(job_id)
        if entry:
            websocket = entry.get("websocket")
            try:
                print(f"[DEBUG] Sending result to client job_id={job_id} url={result_url}")
                await websocket.send_json({
                    "status": "COMPLETED",
                    "result_url": result_url
                })
                print(f"[DEBUG] send_json succeeded for job_id={job_id}")
                # 푸시 후, 클라이언트가 연결을 닫도록 유도합니다.
                return True
            except Exception as e:
                print(f"[WS Push Error] Job ID {job_id}: {e}")
                self.disconnect(job_id)
        return False

manager = ConnectionManager()