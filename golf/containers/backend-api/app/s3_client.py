# backend-api/s3_client.py

# boto3를 사용해 s3 통신, .env의 키와 S3 버킷 이름 사용

import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from typing import Optional

S3_VIDEO_BUCKET_NAME = os.getenv("S3_VIDEO_BUCKET_NAME")
S3_RESULT_BUCKET_NAME = os.getenv("S3_RESULT_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PRESIGNED_URL_EXPIRATION = 3600  # 1 hour

class S3Client:
    def __init__(self):
        # Use signature v4 and virtual-host addressing to avoid redirect/host mismatch
        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"}
        )
        # aws access key로 s3 presigned url 생성에 접근
        # 관리 잘못하면 보안 이슈가 될 수 있으니 주의(s3 과금 어마어마할 것)
        self._client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION,
            config=config,
        )

    # s3 presigned URL 생성 메서드
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
    
    def create_presigned_get_url(self, object_key: str, bucket_name: Optional[str] = None, expires_in: int = PRESIGNED_URL_EXPIRATION) -> str:
        """
        Generate a presigned GET URL using the configured boto3 client.

        Accepts the object key as the primary argument. If `bucket_name` is not
        provided, `S3_RESULT_BUCKET_NAME` will be used. This matches existing
        caller usage which passes only the s3 key.
        """
        bucket = bucket_name or S3_RESULT_BUCKET_NAME
        if not bucket:
            raise RuntimeError("S3 result bucket is not configured (S3_RESULT_BUCKET_NAME)")

        try:
            url = self._client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": object_key},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            # Raise a RuntimeError so callers return 500 with a helpful message
            raise RuntimeError(f"Failed to create presigned GET URL: {e}")