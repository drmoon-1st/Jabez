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
import argparse
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

DEFAULT_URL = "http://localhost:29001/api/result/webhook/job/complete"
DEFAULT_SECRET_ENV = "RUNPOD_WEBHOOK_SECRET"


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
    p = argparse.ArgumentParser(description="Send a signed webhook to the backend to simulate RunPod completion")
    p.add_argument("--url", default=os.getenv("WEBHOOK_URL", DEFAULT_URL), help="Full webhook URL (default: %(default)s)")
    p.add_argument("--secret", default=os.getenv(DEFAULT_SECRET_ENV), help=f"Webhook secret; or set env {DEFAULT_SECRET_ENV}")
    p.add_argument("--job-id", required=True, help="Job ID to send")
    p.add_argument("--s3-key", required=True, help="S3 key/path to the result file")
    p.add_argument("--extra", help="Extra JSON to merge into payload (e.g. '{\"foo\":123}')")
    p.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable verbose output")
    args = p.parse_args()

    if not args.secret:
        print(f"Error: no secret provided and env {DEFAULT_SECRET_ENV} not set")
        sys.exit(2)

    extra = None
    if args.extra:
        try:
            extra = json.loads(args.extra)
        except Exception as e:
            print(f"Failed to parse --extra JSON: {e}")
            sys.exit(2)

    send_webhook(args.url, args.secret, args.job_id, args.s3_key, extra=extra, verbose=args.verbose)


if __name__ == "__main__":
    main()
