# backend-api/routers/upload_router.py

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4
import os
import logging
import traceback
from datetime import datetime

# 로컬 모듈 임포트
from app.db_client import DBClient
from app.s3_client import S3Client
from app.auth_utils import get_current_user_id 

# 환경 변수에서 S3 버킷 이름 로드 (s3_client에서 사용)
S3_VIDEO_BUCKET_NAME = os.getenv("S3_VIDEO_BUCKET_NAME")
S3_RESULT_BUCKET_NAME = os.getenv("S3_RESULT_BUCKET_NAME")

# 로거 설정
logger = logging.getLogger(__name__)

# 💡 수정: prefix를 "/upload"로 설정하여 token_router와 형식 통일
router = APIRouter(
    prefix="/upload", # main.py에서 /api와 결합되어 최종 URL: /api/upload
    tags=["upload"]
)
db_client = DBClient()
s3_client = S3Client()

# 클라이언트로부터 받을 요청 본문 구조
class UploadStartPayload(BaseModel):
    # DB 스키마와 매핑되는 필수 정보
    upload_source: str = Field(..., description="업로드 경로: 2D 또는 3D")
    original_filename: str
    file_type: str
    file_size_bytes: int

    # 비회원 식별자 (선택 사항)
    non_member_identifier: Optional[str] = None
    
# S3 Key 생성 로직 (사용자 ID, 소스, UUID를 조합하여 고유 경로 생성)
def create_s3_key(user_id: Optional[str], non_member_id: Optional[str], source: str, file_type: str) -> str:
    """S3 버킷 내에 저장될 고유한 키 경로를 생성합니다."""
    # 소유자 식별자를 우선 사용
    owner_id = user_id if user_id else (non_member_id if non_member_id else "unknown")
    
    # 파일 확장자 추출 (MIME 타입에서)
    ext = file_type.split('/')[-1] if '/' in file_type else file_type.lower()
    if ext == 'zip' and source == '3D':
        ext = 'zip'
    elif ext == 'mp4' and source == '2D':
        ext = 'mp4'
    else:
        ext = 'dat' # 기본값
        
    upload_uuid = str(uuid4())
    
    # 최종 S3 Key 구조: [소유자 ID]/[업로드 소스]/[UUID].[확장자]
    s3_key = f"{owner_id}/{source.lower()}/{upload_uuid}.{ext}"
    
    # 💡 수정: S3 Key와 함께 Job ID로 사용할 upload_uuid를 함께 반환
    return s3_key, upload_uuid # websocket을 위해 프론트도 작업 id를 알아야함


@router.post("/")
async def start_upload(
    payload: UploadStartPayload,
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    업로드 시작: S3 presigned URL 생성, DB에 업로드 intent 기록 후 presigned_url 반환
    """
    # 보안 강화: 반드시 인증된 사용자만 업로드 허용
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required for upload")
    
    try:
        # 1) S3 키 생성
        s3_key, job_id = create_s3_key(user_id, payload.non_member_identifier, payload.upload_source, payload.file_type)

        # 2) presigned URL 생성
        try:
            presigned_url = s3_client.create_presigned_url(s3_key, payload.file_type, payload.file_size_bytes)
            print(f"Generated presigned URL: {presigned_url}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"S3 presigned URL 생성 실패: {e}")

        # 3) DB에 업로드 intent 기록 (메서드 시그니처에 맞춰 인자 전달)
        try:
            inserted_id = db_client.insert_upload_intent(
                job_id=job_id,
                user_id=user_id,
                non_member_identifier=payload.non_member_identifier,
                upload_source=payload.upload_source,
                s3_key=s3_key,
                filename=payload.original_filename,
                filetype=payload.file_type,
                file_size_bytes=payload.file_size_bytes
            )
            print(f"DB record created with Job ID: {inserted_id}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB 기록 실패: {e}")

        # 3.a) 상태 업데이트: presigned URL 생성 직후, 프론트에 전달 전 S3_PUTING으로 기록
        try:
            db_client.update_upload_status(job_id=job_id, status='PROCESSING')
        except Exception as e:
            # 로그만 남기고 계속 진행 (프로토타입)
            print(f"[DB Update Warning] Failed to set PROCESSING for job_id={job_id}: {e}")

        # 4) presigned URL과 S3 Key 반환
        return {"presigned_url": presigned_url, 
                "job_id": job_id}
    except Exception as e:
        # 서버 콘솔에 전체 스택트레이스 출력 (디버그용)
        logger.error("start_upload 예외 발생: %s", e)
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)

        # 개발/디버그 편의로 상세 메시지 응답 (운영에서는 숨길 것)
        raise HTTPException(status_code=500, detail=f"internal error: {str(e)}")