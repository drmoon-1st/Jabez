#!/bin/bash
set -e

# Activate any environment setup if needed (OpenPose image may already have it)
# cd /opt/openpose-api

# Start FastAPI server with uvicorn
exec python3 -m uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-19030} --workers 1
