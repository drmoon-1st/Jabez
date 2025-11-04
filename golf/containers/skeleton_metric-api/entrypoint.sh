#!/bin/bash
set -e

# NOTE: .env sourcing removed. Container must be started with required runtime env vars.

# Activate any environment setup if needed (OpenPose image may already have it)
# cd /opt/openpose-api

# Start FastAPI server with uvicorn
export OPENPOSE_BATCH_SIZE=${OPENPOSE_BATCH_SIZE:-32}
export OPENPOSE_NUM_GPU=${OPENPOSE_NUM_GPU:-1}
export OPENPOSE_NUM_GPU_START=${OPENPOSE_NUM_GPU_START:-0}
export OPENPOSE_WRITE_IMAGES=${OPENPOSE_WRITE_IMAGES:-0}
# Ensure received payload dir and buckets have defaults if not provided
export RECEIVED_PAYLOAD_DIR=${RECEIVED_PAYLOAD_DIR:-/opt/skeleton_metric-api/received_payloads}
# Standardized env names used by the codebase:
# - S3_VIDEO_BUCKET_NAME : source/input videos (required)
# - S3_RESULT_BUCKET_NAME: where processed results/overlays are uploaded (required)
export S3_VIDEO_BUCKET_NAME=${S3_VIDEO_BUCKET_NAME:-}
export S3_RESULT_BUCKET_NAME=${S3_RESULT_BUCKET_NAME:-}
export AWS_REGION=${AWS_REGION:-}
exec python3 -m uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-19030} --workers 1
