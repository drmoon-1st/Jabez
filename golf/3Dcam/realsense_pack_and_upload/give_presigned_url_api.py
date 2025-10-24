from flask import Flask, request, jsonify
import boto3

app = Flask(__name__)

# S3 클라이언트 (EC2 IAM Role 사용 권장, 키 파일은 ~/.aws/credentials에 설정해도 됨)
s3_client = boto3.client("s3", region_name="us-east-1")  # 리전 지정
BUCKET_NAME = "realsense-s3"

@app.route("/get_presigned_url", methods=["POST"])
def get_presigned_url():
    data = request.get_json()
    object_name = data.get("object_name", "test/upload.bin")

    try:
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": BUCKET_NAME, "Key": object_name},
            ExpiresIn=120   # 2분
        )
        return jsonify({"url": url, "key": object_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
