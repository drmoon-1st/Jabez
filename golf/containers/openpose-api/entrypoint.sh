#!/bin/bash
set -e

# Activate any environment setup if needed (OpenPose image may already have it)
# cd /opt/openpose-api

# Start FastAPI server with uvicorn
export OPENPOSE_BATCH_SIZE=${OPENPOSE_BATCH_SIZE:-32}
export OPENPOSE_NUM_GPU=${OPENPOSE_NUM_GPU:-1}
export OPENPOSE_NUM_GPU_START=${OPENPOSE_NUM_GPU_START:-0}
export OPENPOSE_WRITE_IMAGES=${OPENPOSE_WRITE_IMAGES:-0}
exec python3 -m uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-19030} --workers 1
