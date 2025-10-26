# backend-api/s3_client.py

# boto3를 사용해 s3 통신, .env의 키와 S3 버킷 이름 사용

import os
import boto3
from botocore.config import Config

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
PRESIGNED_URL_EXPIRATION = 3600  # 1 hour

class S3Client:
    def __init__(self):
        # region 명시, signature v4 사용, 가상호스트 스타일로 생성하여 리다이렉트 방지
        config = Config(
            region_name=AWS_REGION,
            signature_version='s3v4',
            s3={'addressing_style': 'virtual'}
        )
        self.client = boto3.client('s3', region_name=AWS_REGION, config=config)

    def create_presigned_url(self, object_key: str, file_type: str, file_size: int) -> str:
        """
        PUT 용 presigned URL 반환.
        - file_type(Content-Type)을 서명에 포함시키면 클라이언트도 동일한 Content-Type을 반드시 보냄.
        - Content-Length는 서명에 포함하지 않음(브라우저가 자동으로 설정하므로 서명에 포함하면 문제 발생).
        """
        params = {
            'Bucket': S3_BUCKET_NAME,
            'Key': object_key,
        }
        # Content-Type을 서명에 포함하려면 아래 주석 해제(프론트에서 동일하게 설정)
        if file_type:
            params['ContentType'] = file_type

        url = self.client.generate_presigned_url(
            ClientMethod='put_object',
            Params=params,
            ExpiresIn=PRESIGNED_URL_EXPIRATION,
            HttpMethod='PUT'
        )
        return url