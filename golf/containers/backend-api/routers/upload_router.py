# backend-api/routers/upload_router.py

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Optional
from pydantic import BaseModel, Field
from uuid import uuid4
import os

# 로컬 모듈 임포트
from db_client import DBClient
from s3_client import S3Client
from auth_utils import get_current_user_id 

# 환경 변수에서 S3 버킷 이름 로드 (s3_client에서 사용)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

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
    upload_source: str = Field(..., description="업로드 경로: WEB_2D 또는 EXE_3D")
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
    if ext == 'zip' and source == 'EXE_3D':
        ext = 'zip'
    elif ext == 'mp4' and source == 'WEB_2D':
        ext = 'mp4'
    else:
        ext = 'dat' # 기본값
        
    upload_uuid = str(uuid4())
    
    # 최종 S3 Key 구조: [소유자 ID]/[업로드 소스]/[UUID].[확장자]
    s3_key = f"{owner_id}/{source.lower()}/{upload_uuid}.{ext}"
    return s3_key


@router.post("/")
async def start_upload(
    payload: UploadStartPayload,
    user_id: Optional[str] = Depends(get_current_user_id)
):
    """
    업로드 시작: S3 presigned URL 생성, DB에 업로드 intent 기록 후 presigned_url 반환
    """
    # 1) S3 키 생성
    s3_key = create_s3_key(user_id, payload.non_member_identifier, payload.upload_source, payload.file_type)

    # 2) presigned URL 생성
    try:
        presigned_url = s3_client.create_presigned_url(s3_key, payload.file_type, payload.file_size_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 presigned URL 생성 실패: {e}")

    # 3) DB에 업로드 intent 기록 (메서드 시그니처에 맞춰 인자 전달)
    try:
        db_client.insert_upload_intent(
            user_id=user_id,
            non_member_identifier=payload.non_member_identifier,
            upload_source=payload.upload_source,
            s3_key=s3_key,
            filename=payload.original_filename,
            filetype=payload.file_type,
            file_size_bytes=payload.file_size_bytes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 기록 실패: {e}")

    return {"presigned_url": presigned_url, "s3_key": s3_key}