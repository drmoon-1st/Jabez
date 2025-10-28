import os
import time
import hmac
import hashlib
from fastapi import HTTPException, Request

# 환경변수: 공유 비밀 (RunPod에 동일값 설정)
RUNPOD_WEBHOOK_SECRET = os.getenv("RUNPOD_WEBHOOK_SECRET")
# 헤더 이름
HEADER_SIG = "X-Runpod-Signature"
HEADER_TS = "X-Runpod-Timestamp"
# 타임스탬프 허용 오차(초)
TIMESTAMP_TOLERANCE = int(os.getenv("WEBHOOK_TIMESTAMP_TOLERANCE", "300"))  # 5 minutes

async def verify_runpod_signature(request: Request) -> None:
    """
    Request body must be same bytes that signer used.
    Signature scheme: HEX(HMAC_SHA256(secret, f"{timestamp}.{body_bytes}"))
    Headers required: X-Runpod-Timestamp, X-Runpod-Signature
    Raises HTTPException(401) on failure.
    """
    if not RUNPOD_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    ts = request.headers.get(HEADER_TS)
    sig = request.headers.get(HEADER_SIG)
    if not ts or not sig:
        raise HTTPException(status_code=401, detail="Missing webhook signature headers")

    try:
        ts_int = int(ts)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid timestamp")

    now = int(time.time())
    if abs(now - ts_int) > TIMESTAMP_TOLERANCE:
        raise HTTPException(status_code=401, detail="Webhook timestamp outside tolerance")

    body = await request.body()  # bytes
    mac = hmac.new(RUNPOD_WEBHOOK_SECRET.encode(), msg=(f"{ts}.".encode() + body), digestmod=hashlib.sha256)
    expected = mac.hexdigest()

    # constant-time compare
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    # OK: return None