import os
import traceback
import json
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from openpose.skeleton_interpolate import interpolate_sequence
# The openpose wrapper provides procedural helpers — import the actual names that exist
from openpose.openpose import run_openpose_on_video, run_openpose_on_dir, OpenPoseProcessVideo

from controller import process_and_save, upload_result_to_s3

app = FastAPI()

class OpenPoseRequest(BaseModel):
    # legacy field: some clients send 'video'
    video: Optional[str] = None
    # new client payload fields
    s3_key: Optional[str] = None
    file_base64: Optional[str] = None
    turbo_without_skeleton: Optional[bool] = True

# NOTE: OpenPose logic has been moved to `openpose/openpose.py` and is imported above.

# Clustering / normalization code removed per request.
# This server now returns raw OpenPose keypoints only (people list and rendered image).

# --- FastAPI endpoint ---
@app.post("/skeleton_metric_predict")
async def skeleton_metric_predict(req: OpenPoseRequest, background_tasks: BackgroundTasks):
    try:
        turbo_without_skeleton = req.turbo_without_skeleton

        # s3_key MUST be present and have the format <user_id>/<dimension>/<job_id>
        if not req.s3_key:
            raise HTTPException(status_code=400, detail="Missing required 's3_key' in request. Must be '<user_id>/<dimension>/<job_id>'.")
        parts = req.s3_key.split('/')
        if len(parts) < 3:
            raise HTTPException(status_code=400, detail="Malformed 's3_key'. Expected format: '<user_id>/<dimension>/<job_id>'")

        # derive identifiers from s3_key when provided, and prepare dest dir
        s3_key = req.s3_key
        user_id = None
        dimension = None
        job_id = None
        if s3_key:
            parts = s3_key.split('/')
            if len(parts) >= 1:
                user_id = parts[0]
            if len(parts) >= 2:
                dimension = parts[1]
            if len(parts) >= 3:
                # strip file extension from the last segment so job_id is a clean identifier
                try:
                    job_id = Path(parts[-1]).stem
                except Exception:
                    job_id = parts[-1]

        base_dir = Path(os.environ.get('RECEIVED_PAYLOAD_DIR', Path.cwd() / 'received_payloads'))
        user_id = user_id or 'unknown_user'
        job_id = job_id or f'job_{int(os.times()[4])}'
        dest_dir = base_dir / user_id / job_id  # /opt/skeleton_metric-api/<user_id>/<job_id>/ 폴더
        dest_dir.mkdir(parents=True, exist_ok=True)

        # save sanitized payload metadata
        try:
            pdata = req.dict()
            if 'file_base64' in pdata:
                pdata.pop('file_base64', None)
            if 'video' in pdata:
                pdata.pop('video', None)
            payload_path = dest_dir / 'payload.json'
            with payload_path.open('w', encoding='utf-8') as pf:
                json.dump(pdata, pf, ensure_ascii=False, indent=2)
        except Exception:
            traceback.print_exc()

        # schedule background processing: controller will download video from S3 using s3_key
        # pass dimension and job_id so controller can handle 2d(mp4) vs 3d(zip) and name result as <job_id>.json
        background_tasks.add_task(process_and_save, req.s3_key, dimension, job_id, turbo_without_skeleton, dest_dir)
        # schedule upload of result.json and overlay videos to result S3 bucket after processing
        
        # background_tasks.add_task(upload_result_to_s3, dest_dir, job_id, s3_key)
        # s3 업로드는 비용으로 인해 잠시 중지

        # return an immediate accepted response with job info
        return JSONResponse(status_code=202, content={
            'message': 'Accepted',
            'user_id': user_id,
            'job_id': job_id,
            'result_path': str(dest_dir / f"{job_id}.json")
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 19030))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)
