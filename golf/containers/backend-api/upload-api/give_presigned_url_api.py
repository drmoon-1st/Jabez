from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3

app = FastAPI()

# S3 클라이언트 설정 (EC2 IAM Role 또는 ~/.aws/credentials 사용)
s3_client = boto3.client("s3", region_name="us-east-1")
BUCKET_NAME = "realsense-s3"


# 요청 바디 모델
class PresignedUrlRequest(BaseModel):
    object_name: str = "test/upload.bin"


@app.post("/get_presigned_url")
def get_presigned_url(request: PresignedUrlRequest):
    try:
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": BUCKET_NAME, "Key": request.object_name},
            ExpiresIn=120,  # 2분
        )
        return {"url": url, "key": request.object_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 로컬 실행용
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=29000)
