import tempfile, pandas as pd, json, os
from pathlib import Path
import sys
sys.path.insert(0, r'D:\Jabez\golf\containers\skeleton_metric-api')
from metric_algorithm import run_metrics_from_context

# create a tiny tidy df_2d: 2 frames, one joint each
rows = [
    {'frame':0,'person_idx':0,'joint_idx':0,'x':10,'y':20,'conf':0.9},
    {'frame':1,'person_idx':0,'joint_idx':0,'x':12,'y':22,'conf':0.8}
]
df = pd.DataFrame(rows)
ctx = {'df_2d': df}
with tempfile.TemporaryDirectory() as td:
    print('tempdir', td)
    res = run_metrics_from_context(ctx, dest_dir=td, job_id='testjob', dimension='2d')
    print('runner returned keys:', list(res.keys()))
    print('metrics keys:', list(res.get('metrics',{}).keys()))
    metrics_path = Path(td) / 'testjob_metric_result.json'
    if metrics_path.exists():
        txt = metrics_path.read_text(encoding='utf-8')
        print('metric_result.json content:\n', txt)
    else:
        print('metric_result.json not created')
    print('files in tempdir:', [p.name for p in Path(td).iterdir()])
