# api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import sys
import tempfile
import base64
import json
import traceback
from pathlib import Path
import numpy as np

# MMAction2 (STGCN) 관련 라이브러리 임포트
# Dockerfile의 FROM dannymaq/mmaction:v6-nano 이미지가 이들을 모두 포함하고 있어야 합니다.
import torch
import torch.nn as nn
import pickle
import pandas as pd
from mmengine.config import Config
from mmengine.runner import Runner

# ----------------------------------------------------------------------
# 1. STGCN 임베딩 추출 핵심 로직 (제공된 코드를 API 함수 내에서 호출할 수 있도록 정리)
# ----------------------------------------------------------------------

# NOTE: 이 함수는 제공된 코드를 기반으로 하므로,
# 설정 파일(my_stgcnpp.py) 및 체크포인트(.pth) 파일은
# API 서버 코드와 '같은 폴더'나 접근 가능한 경로에 있어야 합니다.

def extract_stgcn_embedding(crop_csv_path: Path) -> np.ndarray:
    """
    CSV 경로를 받아 STGCN 모델을 통해 임베딩(Embedding)을 추출합니다.
    추출된 임베딩은 numpy 배열 형태로 반환됩니다.
    """
    # Environment: assume config (my_stgcnpp_api.py) and checkpoint are colocated with this file
    repo_dir = Path(__file__).parent
    CFG = str(repo_dir / 'my_stgcnpp_api.py')

    # locate checkpoint: prefer local stgcn_*.pth then common locations
    CKPT = None
    candidates = [repo_dir / 'stgcn_70p.pth', repo_dir / 'stgcn_70p.pth', repo_dir.parent / 'stgcn_70p.pth']
    for c in candidates:
        if c.exists():
            CKPT = str(c)
            break
    if CKPT is None:
        # fallback: any stgcn_*.pth in repo_dir
        for p in repo_dir.glob('stgcn_*.pth'):
            CKPT = str(p)
            break
    if CKPT is None:
        raise FileNotFoundError(f"No STGCN checkpoint found in {repo_dir} (expected stgcn_*.pth)")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert CSV -> PKL (mmaction2 expected ann_file)
    def csv_to_pkl(csv_path: Path, out_pkl: Path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        F = df.shape[0]
        V = 17
        COCO_NAMES = [
            "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow",
            "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"
        ]
        x_cols = [f"x_{i}" for i in range(V)]
        y_cols = [f"y_{i}" for i in range(V)]
        score_cols = [f"score_{i}" for i in range(V)]
        if all(col in df.columns for col in x_cols) and all(col in df.columns for col in y_cols) and all(col in df.columns for col in score_cols):
            arr = np.stack([
                np.stack([df[x_cols].values, df[y_cols].values, df[score_cols].values], axis=2)
            ], axis=0)[0]
        elif all(f"{name}_x" in df.columns for name in COCO_NAMES) and all(f"{name}_y" in df.columns for name in COCO_NAMES) and all(f"{name}_c" in df.columns for name in COCO_NAMES):
            arr = np.stack([
                np.stack([
                    df[[f"{name}_x" for name in COCO_NAMES]].values,
                    df[[f"{name}_y" for name in COCO_NAMES]].values,
                    df[[f"{name}_c" for name in COCO_NAMES]].values
                ], axis=2)
            ], axis=0)[0]
        else:
            raise ValueError("CSV must be OpenPose COCO17 format (x_0..x_16,y_0..y_16,score_0..16) or Name_x/Name_y/Name_c")

        keypoint = arr[:, :, :2]
        keypoint_score = arr[:, :, 2]
        keypoint = np.expand_dims(keypoint, axis=0)
        keypoint_score = np.expand_dims(keypoint_score, axis=0)
        ann = {
            'frame_dir': csv_path.stem,
            'total_frames': F,
            'keypoint': keypoint,
            'keypoint_score': keypoint_score,
            'label': 0,
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'metainfo': {'frame_dir': csv_path.stem, 'img_shape': (1080, 1920)}
        }
        data = {
            'annotations': [ann],
            'split': {'xsub_val': [csv_path.stem]}
        }
        with open(out_pkl, 'wb') as f:
            pickle.dump(data, f, protocol=4)

    # create temp pkl file in system temp directory to avoid permission issues
    tmp_pkl = Path(tempfile.gettempdir()) / (crop_csv_path.stem + '_' + next(tempfile._get_candidate_names()) + '.pkl')
    csv_to_pkl(crop_csv_path, tmp_pkl)

    cfg = Config.fromfile(CFG)
    # override ann_file robustly for cfg variants
    if hasattr(cfg, 'test_dataloader'):
        # dataset may be a dict-like Config; ensure ann_file set and remove split if present
        try:
            ds = cfg.test_dataloader.dataset
            if isinstance(ds, dict) and 'split' in ds:
                ds.pop('split', None)
            ds.ann_file = str(tmp_pkl)
        except Exception:
            cfg.test_dataloader.dataset.ann_file = str(tmp_pkl)
    elif hasattr(cfg, 'data') and hasattr(cfg.data, 'test') and hasattr(cfg.data.test, 'dataset'):
        try:
            ds = cfg.data.test.dataset
            if isinstance(ds, dict) and 'split' in ds:
                ds.pop('split', None)
            ds.ann_file = str(tmp_pkl)
        except Exception:
            cfg.data.test.dataset.ann_file = str(tmp_pkl)
    else:
        print('Warning: could not find test_dataloader in config to override ann_file')

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(CKPT, map_location=DEVICE)
    model = runner.model.to(DEVICE)
    model.eval()

    last_lin = next((m for m in model.cls_head.modules() if isinstance(m, nn.Linear)), None)
    if last_lin is None:
        raise RuntimeError('cls_head contains no nn.Linear; cannot extract embedding')
    feat_dim = last_lin.in_features

    embs = []
    with torch.no_grad():
        for batch in runner.test_dataloader:
            data_samples = batch.get('data_samples', None) if isinstance(batch, dict) else None
            inputs = batch.get('inputs', None) if isinstance(batch, dict) else None
            # support legacy dataloader batch formats
            if data_samples is None:
                # try to infer single-sample dataset behavior
                try:
                    # some Runners yield a list of dicts
                    data_samples = batch['data_samples']
                    inputs = batch['inputs']
                except Exception:
                    raise RuntimeError('Unexpected batch format from test_dataloader')

            for i, ds in enumerate(data_samples):
                clip_embs = []
                def hook(m, inp, out):
                    clip_embs.append(inp[0].cpu().squeeze(0))
                handle = last_lin.register_forward_hook(hook)
                if isinstance(inputs, list):
                    inp = inputs[i].unsqueeze(0).to(DEVICE)
                elif isinstance(inputs, dict):
                    inp = {k: v[i].unsqueeze(0).to(DEVICE) for k, v in inputs.items()}
                elif torch.is_tensor(inputs):
                    inp = inputs[i].unsqueeze(0).to(DEVICE)
                else:
                    handle.remove()
                    raise TypeError('Unsupported inputs type from dataloader')
                model.forward(inp, [ds], mode='predict')
                handle.remove()
                if not clip_embs:
                    clip_embs.append(torch.zeros(feat_dim))
                video_emb = torch.stack(clip_embs, 0).mean(0).cpu().numpy()
                video_emb = np.nan_to_num(video_emb, nan=0.0, posinf=0.0, neginf=0.0)
                embs.append(video_emb)

    if not embs:
        tmp_pkl.unlink(missing_ok=True)
        raise RuntimeError('no embeddings extracted')

    em_arr = np.stack(embs, 0)
    tmp_pkl.unlink(missing_ok=True)
    return em_arr[0]


# ----------------------------------------------------------------------
# 2. FastAPI 설정
# ----------------------------------------------------------------------

app = FastAPI(title="mmaction-stgcn-api")


class CSVBase64Request(BaseModel):
    csv_base64: str


@app.post('/mmaction_stgcn_embed')
def mmaction_stgcn_embed_endpoint(payload: CSVBase64Request):
    """Accepts a JSON body with base64-encoded CSV and returns STGCN embedding as JSON."""
    temp_csv_path = None
    try:
        csv_base64 = payload.csv_base64
        if not csv_base64:
            raise HTTPException(status_code=400, detail='csv_base64 is required')

        try:
            csv_data = base64.b64decode(csv_base64).decode('utf-8')
        except Exception:
            raise HTTPException(status_code=400, detail='CSV payload not valid Base64')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_csv:
            tmp_csv.write(csv_data)
            temp_csv_path = Path(tmp_csv.name)

        print(f"Saved temp CSV: {temp_csv_path}")

        embedding = extract_stgcn_embedding(temp_csv_path)
        embedding_list = embedding.tolist()

        print(f"STGCN embedding extracted. shape: {embedding.shape}")

        return JSONResponse(status_code=200, content={
            'message': 'OK',
            'embedding': embedding_list,
            'embedding_dim': len(embedding_list)
        })

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"API error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'Server error: {str(e)}')
    finally:
        if temp_csv_path is not None and temp_csv_path.exists():
            try:
                os.remove(temp_csv_path)
            except Exception:
                pass


# ----------------------------------------------------------------------
# 3. 애플리케이션 실행
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # 요청하신 포트 19031 설정
    port = int(os.environ.get('PORT', 19031)) 
    print(f"Flask API 서버를 0.0.0.0:{port}에서 시작합니다.")
    
    # 0.0.0.0 호스트로 설정하여 컨테이너 외부에서 접근 가능하게 합니다.
    # Flask 기본 서버는 프로덕션 용도가 아님을 주의하세요.
    app.run(host='0.0.0.0', port=port, debug=False)