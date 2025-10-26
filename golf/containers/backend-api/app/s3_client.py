# backend-api/s3_client.py

# boto3를 사용해 s3 통신, .env의 키와 S3 버킷 이름 사용

import os
import boto3
from botocore.exceptions import ClientError

S3_VIDEO_BUCKET_NAME = os.getenv("S3_VIDEO_BUCKET_NAME")
S3_RESULT_BUCKET_NAME = os.getenv("S3_RESULT_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PRESIGNED_URL_EXPIRATION = 3600  # 1 hour

class S3Client:
    def __init__(self):
        self._client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION
        )

    def create_presigned_url(self, object_key: str, file_type: str, file_size: int) -> str:
        """
        PUT 용 presigned URL 반환.
        - file_type(Content-Type)을 서명에 포함시키면 클라이언트도 동일한 Content-Type을 반드시 보냄.
        - Content-Length는 서명에 포함하지 않음(브라우저가 자동으로 설정하므로 서명에 포함하면 문제 발생).
        """
        params = {
            'Bucket': S3_VIDEO_BUCKET_NAME,
            'Key': object_key,
        }
        # Content-Type을 서명에 포함하려면 아래 주석 해제(프론트에서 동일하게 설정)
        if file_type:
            params['ContentType'] = file_type

        url = self._client.generate_presigned_url(
            ClientMethod='put_object',
            Params=params,
            ExpiresIn=PRESIGNED_URL_EXPIRATION,
            HttpMethod='PUT'
        )
        return url
    
    def create_presigned_get_url(self, key: str, bucket_name: str = None, expires_in: int = 3600) -> str:
        """
        결과 S3 버킷에 저장된 객체(key)에 대한 presigned GET URL 생성.
        기본 버킷은 S3_RESULT_BUCKET_NAME.
        """
        target_bucket = bucket_name or S3_RESULT_BUCKET_NAME
        if not target_bucket:
            raise RuntimeError("S3_RESULT_BUCKET_NAME not configured")

        try:
            url = self._client.generate_presigned_url(
                'get_object',
                Params={'Bucket': target_bucket, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            raise RuntimeError(f"Failed to create presigned GET URL: {e}")