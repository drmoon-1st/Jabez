import datetime, os, threading, sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

_repo_dir = Path(__file__).parent
DEBUG_LOG = _repo_dir / "api_server_debug.log"

def debug_log(msg: str):
    ts = datetime.datetime.now().isoformat()
    pid = os.getpid()
    tid = threading.get_ident()
    line = f"{ts} PID:{pid} TID:{tid} - {msg}"
    try:
        print(line); sys.stdout.flush()
    except Exception:
        pass
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def csv_to_pkl(csv_path: Path, out_pkl: Path):
    # 간단한 CSV->PKL 변환: 기존의 csv_to_pkl 로직을 복사해 넣으세요
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, sep="\t", encoding="utf-8-sig")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    COCO_NAMES = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"]
    missing = []
    for n in COCO_NAMES:
        for s in ["_x","_y","_c"]:
            if f"{n}{s}" not in df.columns:
                missing.append(f"{n}{s}")
    if missing:
        raise ValueError(f"missing cols: {missing[:10]}")
    arr = np.stack([
        np.stack([
            df[[f"{name}_x" for name in COCO_NAMES]].values,
            df[[f"{name}_y" for name in COCO_NAMES]].values,
            df[[f"{name}_c" for name in COCO_NAMES]].values
        ], axis=2)
    ], axis=0)[0]
    keypoint = np.expand_dims(arr[:, :, :2], axis=0)
    keypoint_score = np.expand_dims(arr[:, :, 2], axis=0)
    ann = {
        "frame_dir": csv_path.stem,
        "total_frames": df.shape[0],
        "keypoint": keypoint,
        "keypoint_score": keypoint_score,
        "label": 0,
        "img_shape": (1080, 1920),
        "original_shape": (1080, 1920),
        "metainfo": {"frame_dir": csv_path.stem},
    }
    data = {"annotations": [ann], "split": {"xsub_val": [csv_path.stem]}}
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f, protocol=4)