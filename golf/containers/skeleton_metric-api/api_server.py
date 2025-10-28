import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from openpose.skeleton_interpolate import interpolate_sequence
from openpose.openpose import (
    OpenPoseImageProcessing,
    OpenPoseProcessImageSequence,
    OpenPoseProcessVideo,
)

app = FastAPI()

class OpenPoseRequest(BaseModel):
    video: Optional[str] = None
    turbo_without_skeleton: Optional[bool] = True

# NOTE: OpenPose logic has been moved to `openpose/openpose.py` and is imported above.

# Clustering / normalization code removed per request.
# This server now returns raw OpenPose keypoints only (people list and rendered image).

# --- FastAPI endpoint ---
@app.post("/skeleton_metric_predict")
async def skeleton_metric_predict(req: OpenPoseRequest):
    try:
        turbo_without_skeleton = req.turbo_without_skeleton
        # only support video input (base64-encoded mp4)
        video_b64 = req.video
        if not video_b64:
            raise ValueError("Request must include 'video' field with base64-encoded MP4 data.")

        batch_results = OpenPoseProcessVideo(video_b64)
        # pick first person per frame (or empty list)
        sequence = [ (ppl[0] if (ppl and len(ppl) > 0) else []) for ppl in batch_results ]
        interpolated = interpolate_sequence(sequence, conf_thresh=0.0, method='linear', fill_method='none')
        people_sequence = [([person] if person else []) for person in interpolated]
        response_payload = {
            'message': 'OK',
            'pose_id': None,
            'people_sequence': people_sequence,
            'frame_count': len(people_sequence)
        }
        return JSONResponse(content=response_payload)
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
