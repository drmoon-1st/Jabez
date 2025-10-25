# backend-api/s3_client.py

# boto3를 사용해 s3 통신, .env의 키와 S3 버킷 이름 사용

import os
import boto3
from botocore.config import Config

# .env에서 AWS 정보 로드 (로컬 개발 시 boto3가 자동으로 로드합니다)
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

PRESIGNED_URL_EXPIRATION = 3600 # 1시간

class S3Client:
    def __init__(self):
        # boto3는 환경 변수, 설정 파일, IAM Role 순서로 자격 증명을 자동 로드
        self.s3 = boto3.client('s3', region_name=AWS_REGION, config=Config(signature_version='s3v4'))
        self.bucket_name = S3_BUCKET_NAME

    def create_presigned_url(self, object_key: str, file_type: str, file_size: int) -> str:
        """S3 PUT 요청을 위한 Presigned URL을 생성합니다."""
        try:
            put_url = self.s3.generate_presigned_url(
                ClientMethod='put_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_key,
                    'ContentType': file_type,
                    'ContentLength': file_size
                },
                ExpiresIn=PRESIGNED_URL_EXPIRATION,
                HttpMethod='PUT'
            )
            return put_url

        except Exception as e:
            # 실제 배포 환경에서는 더 구체적인 예외 처리 및 로깅 필요
            print(f"S3 Presigned URL 생성 오류: {e}")
            raise