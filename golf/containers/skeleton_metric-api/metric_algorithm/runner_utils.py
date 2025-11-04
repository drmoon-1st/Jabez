"""Utilities to standardize metric modules' run_from_context interface.

Usage:
  - Import helpers and use the example template below to implement
    `run_from_context(ctx)` in metric modules.

Contract (recommended for run_from_context):
  - Accepts: ctx (dict) containing keys like 'wide2', 'wide3', 'dest_dir', 'job_id', 'img_dir', 'fps'
  - Returns: JSON-serializable dict (only ints/floats/strs/lists/dicts/None)
  - Writes per-metric CSV and overlay mp4 under dest_dir with job-prefixed names
  - If S3 bucket env var set, helper can upload overlay and return S3 info

This file provides small helpers to make that reliable and consistent.
"""
from pathlib import Path
import os
import json
import boto3
import numpy as _np
import pandas as _pd
from typing import Any, Dict, Optional


def normalize_value(v: Any):
    """Recursively convert numpy/pandas types to Python built-ins for JSON serialization."""
    # numpy scalar
    if isinstance(v, (_np.generic,)):
        try:
            return v.item()
        except Exception:
            return float(v)

    # numpy array -> list
    if isinstance(v, _np.ndarray):
        return v.tolist()

    # pandas
    if isinstance(v, (_pd.Series,)):
        return v.tolist()
    if isinstance(v, (_pd.DataFrame,)):
        # convert DataFrame to list-of-records (shallow)
        return v.where(_pd.notnull(v), None).to_dict(orient="records")

    # dict/list/tuple
    if isinstance(v, dict):
        return {str(k): normalize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalize_value(x) for x in v]

    # basic types
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v

    # fallback: try to stringify
    try:
        return str(v)
    except Exception:
        return None


def normalize_result(obj: Any) -> Any:
    """Normalize an arbitrary result to JSON-serializable structure.

    Typical usage: out['metrics'][name] = normalize_result(result)
    """
    return normalize_value(obj)


def ensure_dir(p: Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_df_csv(df: _pd.DataFrame, dest_dir: Path, job_id: str, metric: str, suffix: str = 'metrics.csv') -> str:
    """Write DataFrame to dest_dir/{job_id}_{metric}_{suffix} and return path string."""
    dest_dir = Path(dest_dir)
    ensure_dir(dest_dir)
    path = dest_dir / f"{job_id}_{metric}_{suffix}"
    # convert pandas NaN to empty for CSV (CSV will be read by metric CLIs too)
    df.to_csv(path, index=False)
    return str(path)


def relocate_overlay(local_overlay: str, dest_dir: Path, job_id: str, metric: str, target_name: Optional[str] = None) -> str:
    """Move or rename an existing overlay file into dest_dir with standardized job-prefixed name.

    If local_overlay already in dest_dir with the right name, returns it.
    """
    dest_dir = Path(dest_dir)
    ensure_dir(dest_dir)
    if target_name is None:
        target_name = f"{job_id}_{metric}_overlay.mp4"
    target = dest_dir / target_name
    src = Path(local_overlay)
    if not src.exists():
        raise FileNotFoundError(str(src))
    # if src and target are same, just return
    try:
        if src.resolve() == target.resolve():
            return str(target)
    except Exception:
        pass
    # copy/move - prefer rename to preserve speed when same filesystem
    try:
        src.replace(target)
    except Exception:
        # fallback to copy
        import shutil
        shutil.copy2(str(src), str(target))
    return str(target)


def upload_overlay_to_s3(local_path: str, job_id: str, metric: str, bucket_envs=('S3_RESULT_BUCKET_NAME', 'RESULT_S3_BUCKET')) -> Optional[Dict[str, str]]:
    """Upload overlay to S3 if bucket configured. Returns {'bucket','key'} or None.

    Key pattern: {job_id}_{metric}_overlay.mp4
    """
    bucket = None
    for env in bucket_envs:
        bucket = os.environ.get(env)
        if bucket:
            break
    if not bucket:
        return None

    path = Path(local_path)
    if not path.exists():
        return None

    key = f"{job_id}_{metric}_overlay.mp4"
    s3 = boto3.client('s3')
    # Upload (let boto3 raise if fails)
    s3.upload_file(str(path), bucket, key)
    return {'bucket': bucket, 'key': key}


# ---------------------------------------------------------------------------
# Example template for run_from_context (copy into metric modules):
#
# from metric_algorithm.runner_utils import (
#     normalize_result, write_df_csv, relocate_overlay, upload_overlay_to_s3, ensure_dir
# )
#
# def run_from_context(ctx: dict):
#     dest = Path(ctx.get('dest_dir', '.'))
#     job_id = str(ctx.get('job_id', 'job'))
#     wide3 = ctx.get('wide3')
#     wide2 = ctx.get('wide2')
#     fps = int(ctx.get('fps', 30))
#     metric = 'my_metric'
#     ensure_dir(dest)
#     out = {}
#     try:
#         if wide3 is not None:
#             # perform metric computation using existing module functions
#             # e.g., df_metrics = compute_metric_df(wide3)
#             metrics_csv = write_df_csv(df_metrics, dest, job_id, metric)
#             out['metrics_csv'] = metrics_csv
#             out['summary'] = normalize_result({'mean': df_metrics['value'].mean()})
#     except Exception as e:
#         return {'error': str(e)}
#
#     # overlay (if available)
#     try:
#         if wide2 is not None:
#             # generate overlay mp4 at temp location or directly to dest
#             overlay_local = str(dest / f"{job_id}_{metric}_overlay.mp4")
#             # call overlay drawing function which writes overlay_local
#             out['overlay_mp4'] = overlay_local
#             # optionally upload to S3
#             s3info = upload_overlay_to_s3(overlay_local, job_id, metric)
#             if s3info:
#                 out['overlay_s3'] = s3info
#     except Exception as e:
#         out.setdefault('overlay_error', str(e))
#
#     return normalize_result(out)
#
# ---------------------------------------------------------------------------
