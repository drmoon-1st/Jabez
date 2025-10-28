#!/usr/bin/env python3
"""
send_webhook.py

Simple script to simulate RunPod (or any worker) calling your backend webhook
`/api/result/webhook/job/complete`.

Signature scheme matches `app/webhook_auth.py` in this repo:
    signature = HMAC_SHA256(RUNPOD_WEBHOOK_SECRET, f"{timestamp}.{body_bytes}")
Headers sent: X-Runpod-Timestamp, X-Runpod-Signature

Usage examples (PowerShell):
  $env:RUNPOD_WEBHOOK_SECRET = 'mysecret'
  python .\lambda-api\send_webhook.py --url http://localhost:29001/api/result/webhook/job/complete --job-id my-job-123 --s3-key videos/result/abc/result.json

Or pass secret explicitly:
  python .\lambda-api\send_webhook.py --secret mysecret --job-id myjob --s3-key path/to/result.json

"""

from dotenv import load_dotenv
load_dotenv()  # Load .env if present
import time
import hmac
import hashlib
import json
import os
import sys

try:
    import requests
except Exception:
    print("Missing dependency: requests. Install with: pip install requests")
    sys.exit(1)

# DB access removed for Lambda-friendly static webhook sender

# --- 백엔드 API 주소 ---
BACKEND_API_BASE_URL = os.getenv("BACKEND_API_BASE_URL")
# If BACKEND_API_BASE_URL is provided (e.g. http://localhost:29001 or http://host:port/api),
# append the webhook path without duplicating '/api'. We'll simply append the path
# '/result/webhook/job/complete' to the base URL.
BACKEND_API_WEBHOOK_URL = (f"{BACKEND_API_BASE_URL.rstrip('/')}/result/webhook/job/complete" \
                          if BACKEND_API_BASE_URL else None)
print(f"[Config] BACKEND_API_WEBHOOK_URL={BACKEND_API_WEBHOOK_URL}")

RUNPOD_WEBHOOK_SECRET = os.getenv("RUNPOD_WEBHOOK_SECRET")

# test용 dumy 값, 실제로는 데이터 흐름에서 자연스럽게 공급되는값, 지금은 백엔드에서 s3로 보낸후, 결과를 기다리는 backend에 전해줄 값 
S3_KEY = os.getenv("S3_KEY")
JOB_ID = os.getenv("JOB_ID")


def make_signature(secret: str, timestamp: int, body_bytes: bytes) -> str:
    mac = hmac.new(secret.encode("utf-8"), msg=(f"{timestamp}.".encode("utf-8") + body_bytes), digestmod=hashlib.sha256)
    return mac.hexdigest()


def send_webhook(url: str, secret: str, job_id: str, s3_key: str, extra: dict = None, verbose: bool = True):
    payload = {
        "job_id": job_id,
        "s3_result_path": s3_key,
    }
    if extra:
        payload.update(extra)

    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ts = int(time.time())
    sig = make_signature(secret, ts, body_bytes)

    headers = {
        "Content-Type": "application/json",
        "X-Runpod-Timestamp": str(ts),
        "X-Runpod-Signature": sig,
    }

    if verbose:
        print(f"POST {url}")
        print("Headers:")
        for k, v in headers.items():
            print(f"  {k}: {v}")
        print("Body:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print()

    resp = requests.post(url, headers=headers, data=body_bytes, timeout=15)
    if verbose:
        print(f"Response: {resp.status_code}")
        try:
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(resp.text)
    return resp


def main():
    # Load configuration from environment variables (static behavior)
    # Enforce: always derive webhook URL from BACKEND_API_BASE_URL
    if not BACKEND_API_BASE_URL:
        print("Error: BACKEND_API_BASE_URL environment variable must be set")
        sys.exit(2)
    url = BACKEND_API_WEBHOOK_URL
    # prefer RUNPOD_WEBHOOK_SECRET loaded at module import
    secret = RUNPOD_WEBHOOK_SECRET

    # JOB_ID and S3_KEY are expected to be provided via environment for static runs.
    # If JOB_ID is missing, generate a timestamped test id.
    job_id = JOB_ID
    s3_key = S3_KEY

    # Optional: extra JSON string in env (EXTRA_JSON) will be parsed and merged.
    extra = None
    extra_raw = os.getenv("EXTRA_JSON")
    if extra_raw:
        try:
            extra = json.loads(extra_raw)
        except Exception as e:
            print(f"Failed to parse EXTRA_JSON env: {e}")
            sys.exit(2)

    # Verbose control: set VERBOSE=0 or VERBOSE=false to disable
    verbose_env = os.getenv("VERBOSE", "1")
    verbose = False if verbose_env.lower() in ("0", "false", "no") else True

    # Validate required values
    if not secret:
        print("Error: no RUNPOD_WEBHOOK_SECRET provided in environment")
        sys.exit(2)
    if not url:
        print("Error: no WEBHOOK_URL configured")
        sys.exit(2)
    if not s3_key:
        print("Error: no S3_KEY / S3_RESULT_PATH provided in environment")
        sys.exit(2)

    if verbose:
        print("Configuration:")
        print(f"  WEBHOOK_URL={url}")
        print(f"  JOB_ID={job_id}")
        print(f"  S3_KEY={s3_key}")
        print(f"  EXTRA_JSON={extra}")

    send_webhook(url, secret, job_id, s3_key, extra=extra, verbose=verbose)


if __name__ == "__main__":
    main()
