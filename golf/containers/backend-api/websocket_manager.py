# app/websocket_manager.py

from typing import Dict
from fastapi import WebSocket

# 💡 운영 환경에서는 Redis 또는 RDBMS를 사용하여 이 상태를 관리해야 합니다.
# 현재는 개발 편의를 위해 메모리 딕셔너리를 사용합니다.
class ConnectionManager:
    def __init__(self):
        # Job ID (UUID)와 활성 WebSocket 연결 객체를 매핑
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        """연결 수락 및 등록"""
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect(self, job_id: str):
        """연결 해제"""
        if job_id in self.active_connections:
            del self.active_connections[job_id]

    async def send_result_to_client(self, job_id: str, result_url: str) -> bool:
        """특정 Job ID 클라이언트에게 결과 URL 푸시"""
        if job_id in self.active_connections:
            websocket = self.active_connections[job_id]
            try:
                await websocket.send_json({
                    "status": "COMPLETED",
                    "result_url": result_url
                })
                # 푸시 후, 클라이언트가 연결을 닫도록 유도합니다.
                return True
            except Exception as e:
                print(f"[WS Push Error] Job ID {job_id}: {e}")
                self.disconnect(job_id)
        return False

manager = ConnectionManager()