import os
import json
import threading
import traceback
import base64
from pathlib import Path
from typing import Optional

import requests
import pandas as pd


def _normalize_wide_df(wide2: pd.DataFrame) -> pd.DataFrame:
    joint_map = {
        'nose': 'Nose', 'leye': 'LEye', 'reye': 'REye', 'lear': 'LEar', 'rear': 'REar',
        'lshoulder': 'LShoulder', 'rshoulder': 'RShoulder', 'lelbow': 'LElbow', 'relbow': 'RElbow',
        'lwrist': 'LWrist', 'rwrist': 'RWrist', 'lhip': 'LHip', 'rhip': 'RHip',
        'lknee': 'LKnee', 'rknee': 'RKnee', 'lankle': 'LAnkle', 'rankle': 'RAnkle',
        'neck': 'Neck', 'chest': 'Chest', 'midhip': 'MidHip'
    }
    col_rename = {}
    for col in wide2.columns:
        lc = col.lower()
        if lc.endswith('_conf'):
            base = lc[:-5]; suf = '_c'
        elif lc.endswith('_score'):
            base = lc[:-6]; suf = '_c'
        elif lc.endswith('_c'):
            base = lc[:-2]; suf = '_c'
        elif lc.endswith('_x'):
            base = lc[:-2]; suf = '_x'
        elif lc.endswith('_y'):
            base = lc[:-2]; suf = '_y'
        else:
            continue
        joint_name = joint_map.get(base, None)
        if joint_name is None:
            joint_name = base.capitalize()
        newcol = f"{joint_name}{suf}"
        col_rename[col] = newcol
    if col_rename:
        wide2 = wide2.rename(columns=col_rename)
    return wide2


def _validate_csv_matches_coco(csv_path: Path) -> Optional[list]:
    try:
        try:
            df_check = pd.read_csv(csv_path, encoding='utf-8-sig')
            cols_clean = [c.strip().replace('\ufeff', '') for c in df_check.columns]
        except Exception:
            df_check = pd.read_csv(csv_path, sep='\t', encoding='utf-8-sig')
            cols_clean = [c.strip().replace('\ufeff', '') for c in df_check.columns]
        COCO_NAMES = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"]
        missing = []
        for n in COCO_NAMES:
            for s in ("_x","_y","_c"):
                if f"{n}{s}" not in cols_clean:
                    missing.append(f"{n}{s}")
        return missing if missing else None
    except Exception:
        return ['__validation_error__']


def sanitize_skeleton_csv(csv_path: Path) -> Path:
    """Read a CSV (possibly with duplicate/variant columns), merge duplicate aliases,
    and write back a cleaned CSV that contains canonical COCO columns: <Joint>_x, _y, _c.

    Returns the Path to the cleaned CSV (overwrites csv_path in-place).
    """
    try:
        # read without mangle to preserve duplicates if any
        import pandas as _pd
        try:
            df = _pd.read_csv(csv_path, encoding='utf-8-sig', mangle_dupe_cols=False)
        except TypeError:
            # older pandas may not have mangle_dupe_cols; fallback to default
            df = _pd.read_csv(csv_path, encoding='utf-8-sig')

        # Flatten possible multi-columns into a map: lowername -> list of colnames
        colmap = {}
        for c in df.columns:
            colmap.setdefault(c.lower(), []).append(c)

        COCO_NAMES = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle"]
        out_cols = []
        rows = []
        for _, r in df.iterrows():
            out_row = {}
            lower_row = {str(k).lower(): v for k, v in r.items()}
            for j in COCO_NAMES:
                # candidate keys
                x_keys = [f"{j}__x", f"{j}_x", f"{j}_X", f"{j}__X"]
                y_keys = [f"{j}__y", f"{j}_y", f"{j}_Y", f"{j}__Y"]
                c_keys = [f"{j}__c", f"{j}_c", f"{j}_conf", f"{j}_score"]
                # find first available
                x_val = None
                y_val = None
                c_val = None
                for k in x_keys:
                    if k.lower() in lower_row:
                        x_val = lower_row[k.lower()]
                        break
                for k in y_keys:
                    if k.lower() in lower_row:
                        y_val = lower_row[k.lower()]
                        break
                for k in c_keys:
                    if k.lower() in lower_row:
                        c_val = lower_row[k.lower()]
                        break
                out_row[f"{j}_x"] = x_val if x_val is not None else ''
                out_row[f"{j}_y"] = y_val if y_val is not None else ''
                # default confidence to 0 if missing (MMACTION expects _c columns present)
                try:
                    out_row[f"{j}_c"] = float(c_val) if c_val is not None and str(c_val).strip() != '' else 0.0
                except Exception:
                    out_row[f"{j}_c"] = 0.0
            rows.append(out_row)
        out_df = _pd.DataFrame(rows, columns=[f"{j}_{s}" for j in COCO_NAMES for s in ('x','y','c')])
        out_df.to_csv(csv_path, index=False)
        return csv_path
    except Exception:
        # if anything goes wrong, return original path unchanged
        return csv_path


def _mmaction_post_worker(input_path: Path, api_url: str, resp_path: Path, job_id_local: str, dimension_local: str):
    try:
        txt = input_path.read_text(encoding='utf-8')
        b64 = base64.b64encode(txt.encode('utf-8')).decode('utf-8')
        # send only csv_base64 payload to match the original client API
        payload = {'csv_base64': b64}
        try:
            resp = requests.post(api_url, json=payload, timeout=(5, 600))
            if resp.ok:
                try:
                    content = {'ok': True, 'status_code': resp.status_code, 'resp_json': resp.json()}
                except Exception:
                    content = {'ok': True, 'status_code': resp.status_code, 'text': resp.text}
            else:
                content = {'ok': False, 'status_code': resp.status_code, 'text': resp.text}
        except Exception as e:
            content = {'ok': False, 'error': str(e), 'trace': traceback.format_exc()}
    except Exception as e:
        content = {'ok': False, 'error': str(e), 'trace': traceback.format_exc()}
    try:
        # atomic write for response file
        tmp = resp_path.with_suffix(resp_path.suffix + '.tmp')
        tmp.write_text(json.dumps(content), encoding='utf-8')
        try:
            tmp.replace(resp_path)
        except Exception:
            try:
                os.replace(str(tmp), str(resp_path))
            except Exception:
                # final fallback
                resp_path.write_text(json.dumps(content), encoding='utf-8')
    except Exception:
        pass


def start_mmaction_from_csv(csv_path: Path, dest_dir: str, job_id: str, dimension: str, response_payload: dict, mmaction_api: Optional[str]=None):
    """Validate an already-written skeleton CSV and start a background POST worker.

    This function expects the CSV to already match the COCO wide format (or will validate and
    write an error artifact). It returns a dict similar to other starters.
    """
    dest_dir = Path(dest_dir)
    resp_path = dest_dir / f"{job_id}_stgcn_response.json"
    try:
        mmaction_api = mmaction_api or response_payload.setdefault('debug', {}).get('mmaction_api') or os.environ.get('MMACTION_API_URL') or 'http://mmaction2:19031/mmaction_stgcn_test'
        response_payload.setdefault('debug', {})['mmaction_api'] = mmaction_api

        missing = _validate_csv_matches_coco(Path(csv_path))
        if missing:
            try:
                err_obj = {'error': 'missing_mmaction_columns', 'missing': missing[:50]}
                (dest_dir / f"{job_id}_stgcn_error.json").write_text(json.dumps(err_obj), encoding='utf-8')
            except Exception:
                pass
            response_payload.setdefault('debug', {})['mmaction_missing_cols'] = missing[:50]
            return {'csv_ready': False, 'csv_path': Path(csv_path), 'response_path': resp_path, 'thread_started': False}

        # start worker thread to POST the CSV
        try:
            t = threading.Thread(target=_mmaction_post_worker, args=(Path(csv_path), mmaction_api, resp_path, job_id, dimension), daemon=True)
            t.start()
            response_payload.setdefault('debug', {})['mmaction_thread_started'] = True
            return {'csv_ready': True, 'csv_path': Path(csv_path), 'response_path': resp_path, 'thread_started': True, 'thread': t}
        except Exception:
            response_payload.setdefault('debug', {})['mmaction_thread_start_error'] = True
            return {'csv_ready': True, 'csv_path': Path(csv_path), 'response_path': resp_path, 'thread_started': False, 'thread': None}
    except Exception:
        response_payload.setdefault('debug', {})['mmaction_prepare_error'] = traceback.format_exc()
        return {'csv_ready': False, 'csv_path': Path(csv_path), 'response_path': resp_path, 'thread_started': False}


def start_mmaction_from_dfs(df_2d, df_3d, dest_dir: str, job_id: str, dimension: str, response_payload: dict, mmaction_api: Optional[str]=None):
    """
    Prepare MMACTION CSV from df_2d or df_3d, validate columns, and start a background POST worker.
    Returns a dict: {'csv_ready': bool, 'csv_path': Path, 'response_path': Path, 'thread_started': bool}
    """
    dest_dir = Path(dest_dir)
    mmaction_input_csv = dest_dir / f"{job_id}_mmaction_input.csv"
    mmaction_response_path = dest_dir / f"{job_id}_stgcn_response.json"
    thread_started = False
    csv_ready = False
    try:
        mmaction_api = mmaction_api or response_payload.setdefault('debug', {}).get('mmaction_api') or os.environ.get('MMACTION_API_URL') or 'http://mmaction2:19031/mmaction_stgcn_test'
        response_payload.setdefault('debug', {})['mmaction_api'] = mmaction_api

        if df_2d is not None and isinstance(df_2d, pd.DataFrame) and not df_2d.empty:
            from metric_algorithm.runner_utils import tidy_to_wide
            wide2 = tidy_to_wide(df_2d, dimension='2d', person_idx=0) if (isinstance(df_2d, pd.DataFrame) and not df_2d.empty) else pd.DataFrame()
            if not isinstance(wide2, pd.DataFrame) or wide2.empty:
                raise ValueError('tidy->wide conversion produced empty wide DataFrame')
            wide2 = _normalize_wide_df(wide2)
            expected_joints = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
            missing_cols = []
            for j in expected_joints:
                for sfx in ('_x','_y','_c'):
                    coln = f"{j}{sfx}"
                    if coln not in wide2.columns:
                        missing_cols.append(coln)
            if missing_cols:
                # write explicit error artifact
                err_obj = {'error': 'missing_mmaction_columns', 'missing': missing_cols[:50]}
                try:
                    (dest_dir / f"{job_id}_stgcn_error.json").write_text(json.dumps(err_obj), encoding='utf-8')
                except Exception:
                    pass
                response_payload.setdefault('debug', {})['mmaction_missing_cols'] = missing_cols[:50]
                return {'csv_ready': False, 'csv_path': mmaction_input_csv, 'response_path': mmaction_response_path, 'thread_started': False}
            wide2.to_csv(mmaction_input_csv, index=False)
            csv_ready = True
        elif df_3d is not None and isinstance(df_3d, pd.DataFrame) and not df_3d.empty:
            cols_needed = [c for c in ('frame','person_idx','joint_idx','x','y','conf') if c in df_3d.columns]
            if cols_needed:
                df2_from_3d = df_3d[[c for c in ('frame','person_idx','joint_idx','x','y','conf') if c in df_3d.columns]].copy()
            else:
                df2_from_3d = pd.DataFrame()
            if df2_from_3d.empty:
                raise ValueError('Unable to derive tidy 2D from df_3d')
            from metric_algorithm.runner_utils import tidy_to_wide
            wide2 = tidy_to_wide(df2_from_3d, dimension='2d', person_idx=0) if (not df2_from_3d.empty) else pd.DataFrame()
            if not isinstance(wide2, pd.DataFrame) or wide2.empty:
                raise ValueError('tidy->wide conversion produced empty wide DataFrame from df_3d')
            wide2 = _normalize_wide_df(wide2)
            expected_joints = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
            missing_cols = []
            for j in expected_joints:
                for sfx in ('_x','_y','_c'):
                    coln = f"{j}{sfx}"
                    if coln not in wide2.columns:
                        missing_cols.append(coln)
            if missing_cols:
                err_obj = {'error': 'missing_mmaction_columns', 'missing': missing_cols[:50]}
                try:
                    (dest_dir / f"{job_id}_stgcn_error.json").write_text(json.dumps(err_obj), encoding='utf-8')
                except Exception:
                    pass
                response_payload.setdefault('debug', {})['mmaction_missing_cols'] = missing_cols[:50]
                return {'csv_ready': False, 'csv_path': mmaction_input_csv, 'response_path': mmaction_response_path, 'thread_started': False}
            wide2.to_csv(mmaction_input_csv, index=False)
            csv_ready = True

        # validate written csv
        if csv_ready:
            # sanitize duplicates and ensure _c columns exist
            try:
                sanitize_skeleton_csv(mmaction_input_csv)
            except Exception:
                pass
            missing = _validate_csv_matches_coco(mmaction_input_csv)
            if missing:
                try:
                    err_obj = {'error': 'missing_mmaction_columns', 'missing': missing[:50]}
                    (dest_dir / f"{job_id}_stgcn_error.json").write_text(json.dumps(err_obj), encoding='utf-8')
                except Exception:
                    pass
                response_payload.setdefault('debug', {})['mmaction_missing_cols'] = missing[:50]
                return {'csv_ready': False, 'csv_path': mmaction_input_csv, 'response_path': mmaction_response_path, 'thread_started': False}

        # start worker
        if csv_ready:
            try:
                t = threading.Thread(target=_mmaction_post_worker, args=(mmaction_input_csv, mmaction_api, mmaction_response_path, job_id, dimension), daemon=True)
                t.start()
                thread_started = True
                response_payload.setdefault('debug', {})['mmaction_thread_started'] = True
                # record thread in return map
            except Exception:
                response_payload.setdefault('debug', {})['mmaction_thread_start_error'] = True

    except Exception:
        response_payload.setdefault('debug', {})['mmaction_prepare_error'] = traceback.format_exc()

    # include thread handle if started
    return {'csv_ready': csv_ready, 'csv_path': mmaction_input_csv, 'response_path': mmaction_response_path, 'thread_started': thread_started, 'thread': (t if 't' in locals() else None)}


def start_mmaction_post_if_csv_exists(dest_dir: str, job_id: str, dimension: str, response_payload: dict, mmaction_api: Optional[str]=None):
    """If a CSV already exists, start a background POST worker (if response not present)."""
    dest_dir = Path(dest_dir)
    mmaction_input_csv = dest_dir / f"{job_id}_mmaction_input.csv"
    mmaction_response_path = dest_dir / f"{job_id}_stgcn_response.json"
    started = False
    try:
        if mmaction_input_csv.exists() and not mmaction_response_path.exists():
            mmaction_api = mmaction_api or response_payload.setdefault('debug', {}).get('mmaction_api') or os.environ.get('MMACTION_API_URL') or 'http://mmaction2:19031/mmaction_stgcn_test'
            # sanitize CSV before posting
            try:
                sanitize_skeleton_csv(mmaction_input_csv)
            except Exception:
                pass
            t = threading.Thread(target=_mmaction_post_worker, args=(mmaction_input_csv, mmaction_api, mmaction_response_path, job_id, dimension), daemon=True)
            t.start()
            response_payload.setdefault('debug', {})['mmaction_thread_started_later'] = True
            # expose that a thread was started; caller can still poll for resp file
            response_payload.setdefault('debug', {})['mmaction_thread_handle_later'] = True
            started = True
            return {'started': True, 'thread': t, 'response_path': mmaction_response_path, 'csv_path': mmaction_input_csv}
    except Exception:
        response_payload.setdefault('debug', {})['mmaction_thread_later_error'] = True
    return {'started': started, 'thread': None, 'response_path': mmaction_response_path, 'csv_path': mmaction_input_csv}
