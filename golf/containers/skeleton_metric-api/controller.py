"""
Controller (OpenPose + Metrics Orchestrator)
===========================================

이 모듈은 다음을 수행합니다.
- 입력(2D: mp4, 3D: zip)을 S3에서 다운로드하여 /tmp 하위에서 처리
- OpenPose로 포즈 키포인트 추출 → 프레임별 JSON/이미지 생성 → DataFrame 구성(df_2d/df_3d)
- 메트릭 러너(metric_algorithm)를 호출해 각 메트릭을 실행하고 산출물(csv/mp4/summary)을 수집
- 최종 결과 JSON(<job_id>.json)을 저장하고, 산출물 디렉터리(csv/, mp4/, img/, openpose_img/)를 정리

환경 변수
- 입력 버킷: S3_VIDEO_BUCKET_NAME 또는 S3_BUCKET/AWS_S3_BUCKET
- 결과 버킷: S3_RESULT_BUCKET_NAME 또는 RESULT_S3_BUCKET

주의
- 본 파일은 openpose 패키지 경로가 PYTHONPATH에 잡혀 있어야 합니다
    (예: .../golf/containers/skeleton_metric-api 가 sys.path에 포함되어야 openpose 임포트 가능).
"""

import traceback
import json
import os
import boto3
from pathlib import Path
from typing import Optional, Tuple, List
import tempfile
import zipfile
import shutil

import numpy as np
import pandas as pd
# Import mmaction_client robustly: when this module is executed as a top-level script
# (e.g. inside a container via `python controller.py` or uvicorn importing module by name),
# relative imports like `from . import mmaction_client` can fail with
# "attempted relative import with no known parent package". Try relative first,
# then fall back to absolute import which works when the package root is on sys.path.
try:
    from . import mmaction_client
except Exception:
    import mmaction_client

from openpose.skeleton_interpolate import interpolate_sequence
from openpose.openpose import run_openpose_on_video, run_openpose_on_dir, _sanitize_person_list


def _safe_write_json(path: Path, obj: dict):
    """Atomically write JSON to `path` by writing to a temp file and replacing.

    Uses json.dump with default=str so Path objects won't break serialization.
    """
    try:
        tmp = Path(str(path) + '.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        # atomic replace
        try:
            tmp.replace(path)
        except Exception:
            # fallback to os.replace
            os.replace(str(tmp), str(path))
    except Exception:
        # best-effort: try non-atomic write
        try:
            with path.open('w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass


def run_metrics_in_process(dimension: str, ctx: dict):
    """메트릭 실행 훅.

    Parameters
    ----------
    dimension : str
        '2d' 또는 '3d'. 결과 JSON이나 메트릭 내부 로직에서 필요.
    ctx : dict
        `process_and_save` 함수의 locals()를 기반으로 한 문맥. 여기에는
        df_2d / df_3d (존재 시), people_sequence, dest_dir, job_id 등이 포함.

    Behavior
    --------
    - `metric_algorithm.run_metrics_from_context` 를 찾아 실행합니다.
    - 메트릭 모듈/실행 실패는 전체 파이프라인을 멈추지 않고 None 반환.

    Returns
    -------
    dict | None
        메트릭 실행 결과(모듈별 딕셔너리) 또는 실패 시 None.
    """
    try:
        try:
            from metric_algorithm import run_metrics_from_context
        except Exception:
            traceback.print_exc()
            return None

        dest_dir = ctx.get('dest_dir') or ctx.get('dest') or None
        job_id = ctx.get('job_id') or ctx.get('job') or None

        if dest_dir is None or job_id is None:
            try:
                res = run_metrics_from_context(
                    ctx,
                    dest_dir=str(dest_dir) if dest_dir is not None else '.',
                    job_id=str(job_id) if job_id is not None else 'unknown',
                    dimension=dimension,
                )
            except Exception:
                traceback.print_exc()
                return None
        else:
            try:
                res = run_metrics_from_context(
                    ctx,
                    dest_dir=str(dest_dir),
                    job_id=str(job_id),
                    dimension=dimension,
                )
            except Exception:
                traceback.print_exc()
                return None
        return res
    except Exception:
        traceback.print_exc()
        return None


def process_and_save(s3_key: str, dimension: str, job_id: str, turbo_without_skeleton: bool, dest_dir: Path):
    """입력 다운로드 → OpenPose 실행 → 보간 → 결과 저장(+메트릭 실행)까지 전체 파이프라인.

    매개변수
    - s3_key: 입력 객체 키(2D: mp4, 3D: zip)
    - dimension: '2d' | '3d'
    - job_id: 산출물 파일명 접두에 쓰일 식별자
    - turbo_without_skeleton: (예약) skeleton 추출 스킵 플래그. 현재 로직에서는 사용하지 않음.
    - dest_dir: 산출물 저장 디렉터리 경로

    처리 개요
    1) S3에서 입력 다운로드 (/tmp 하위 작업 디렉터리 사용)
    2) 2D: mp4 → run_openpose_on_video → 프레임별 JSON 파싱 → df_2d 생성 → 원본 프레임 추출(img/)
       3D: zip → color/, depth/ 추출 → run_openpose_on_dir(color/) → 깊이 샘플링으로 Z 추정 → intrinsics로 X/Y 계산 → df_3d 생성
    3) interpolate_sequence로 people_sequence 생성(프레임별 첫 번째 사람 기준)
    4) <job_id>.json 저장 + OpenPose 렌더 이미지 openpose_img/ 보존
    5) run_metrics_in_process로 메트릭 실행 → 결과(csv, overlay mp4, summary 등)를 job json에 병합
    6) 산출물 정리(csv/, mp4/ 하위로 이동) 후 job json 경로 업데이트

    반환
    - response_payload(dict): message, people_sequence, frame_count, skeleton_rows/columns 등 포함
    """
    try:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 입력 버킷을 환경변수에서 결정합니다.
        bucket = os.environ.get('S3_VIDEO_BUCKET_NAME') or os.environ.get('S3_BUCKET') or os.environ.get('AWS_S3_BUCKET')
        if not bucket:
            raise RuntimeError('S3_VIDEO_BUCKET_NAME (or S3_BUCKET/AWS_S3_BUCKET) environment variable is not set')

        s3 = boto3.client('s3')
        key = s3_key.lstrip('/')

        # /tmp 하위에 작업 디렉터리를 만들고(컨테이너 친화적), 디버깅을 위해 자동 삭제하지 않습니다.
        # 운영 환경에서는 정리 정책/주기를 고려하세요.
        tmpdir = tempfile.mkdtemp(dir='/tmp')
        tmp_dir = Path(tmpdir)
        # If you later want automatic cleanup, replace above with:
        # with tempfile.TemporaryDirectory(dir='/tmp') as tmpdir:
        #     tmp_dir = Path(tmpdir)
        output_json_dir = tmp_dir / 'json'
        output_json_dir.mkdir(parents=True, exist_ok=True)
        output_img_dir = tmp_dir / 'img'
        output_img_dir.mkdir(parents=True, exist_ok=True)

        result_by_frame = []

        if dimension == '2d':  # --- 2D 처리 경로 ---
            # download mp4 and run openpose on video
            local_video = tmp_dir / 'input.mp4'
            s3.download_file(bucket, key, str(local_video))
            run_openpose_on_video(str(local_video), str(output_json_dir), str(output_img_dir))

            # OpenPose 결과(JSON)를 파싱하여 프레임/사람/관절 단위의 tidy(long) DataFrame(df_2d)을 생성합니다.
            json_paths = sorted([p for p in output_json_dir.iterdir() if p.suffix == '.json'])
            rows = []  # rows for DataFrame
            for frame_idx, jp in enumerate(json_paths):
                with jp.open('r', encoding='utf-8') as f:
                    jdata = json.load(f)
                raw_people = jdata.get('people', [])
                ppl = []
                for person_idx, p in enumerate(raw_people):
                    if 'pose_keypoints_2d' in p:
                        kps = p['pose_keypoints_2d']
                        person = [kps[idx:idx+3] for idx in range(0, len(kps), 3)]
                        person = _sanitize_person_list(person)
                        ppl.append(person)
                result_by_frame.append(ppl)

                # convert to long-format rows: one row per joint per person
                for person_idx, person in enumerate(ppl):
                    for joint_idx, (x, y, c) in enumerate(person):
                        rows.append({
                            'frame': frame_idx,
                            'person_idx': person_idx,
                            'joint_idx': joint_idx,
                            'x': float(x),
                            'y': float(y),
                            'conf': float(c)
                        })

            df_2d = pd.DataFrame(rows)
            # 원본 RGB 프레임을 dest_dir/img로 추출하여 메트릭 오버레이가 실제 프레임에 그려지도록 합니다.
            try:
                dest_img_dir = Path(dest_dir) / 'img'
                dest_img_dir.mkdir(parents=True, exist_ok=True)
                if 'local_video' in locals() and Path(local_video).exists():
                    try:
                        import cv2 as _cv2
                        cap = _cv2.VideoCapture(str(local_video))
                        idx = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            outp = dest_img_dir / f"{idx:06d}.png"
                            try:
                                _cv2.imwrite(str(outp), frame)
                            except Exception:
                                pass
                            idx += 1
                        cap.release()
                    except Exception:
                        # If OpenCV not available or extraction fails, fall back to copying
                        # OpenPose rendered images later (there is a fallback further down).
                        pass
            except Exception:
                pass
        elif dimension == '3d':  # --- 3D 처리 경로 ---
                # download zip and extract expected color/ and depth/ folders
                local_zip = tmp_dir / 'input.zip'
                s3.download_file(bucket, key, str(local_zip))

                # 안전한 압축 해제(zip-slip 방지)와 간단한 용량 제한 처리
                MAX_FILES = 5000
                MAX_TOTAL_UNCOMPRESSED = 1_000_000_000  # bytes
                total_uncompressed = 0
                file_count = 0
                with zipfile.ZipFile(local_zip, 'r') as zf:
                    for zi in zf.infolist():
                        file_count += 1
                        if file_count > MAX_FILES:
                            raise RuntimeError('Zip contains too many files')
                        # Protect against zip-slip
                        extracted_path = tmp_dir / zi.filename
                        if not str(extracted_path.resolve()).startswith(str(tmp_dir.resolve())):
                            raise RuntimeError('Zip contains unsafe path')
                        total_uncompressed += zi.file_size
                        if total_uncompressed > MAX_TOTAL_UNCOMPRESSED:
                            raise RuntimeError('Zip total size too large')
                    zf.extractall(path=tmp_dir)

                # color/ 와 depth/ 디렉터리가 있어야 합니다(대소문자 유연 처리 포함).
                color_dir = tmp_dir / 'color'
                depth_dir = tmp_dir / 'depth'
                if not color_dir.exists() or not depth_dir.exists():
                    # attempt to find them case-insensitively
                    dirs = {p.name.lower(): p for p in tmp_dir.iterdir() if p.is_dir()}
                    color_dir = dirs.get('color', color_dir)
                    depth_dir = dirs.get('depth', depth_dir)
                if not color_dir.exists() or not depth_dir.exists():
                    raise RuntimeError('Expected color/ and depth/ folders in zip')

                # Build a sorted list of depth files as a robust fallback mapping
                depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix == '.npy'])
                depth_count = len(depth_files)
                depth_map_info = {'depth_count': depth_count}

                # 트리 전체에서 intrinsics를 탐색하여 카메라 파라미터/스케일 정보를 사용합니다.
                intr = None
                intr_source = None
                try:
                    # look in obvious places first, then do a recursive search
                    intr_candidates = [tmp_dir / 'intrinsics.json', color_dir / 'intrinsics.json', depth_dir / 'intrinsics.json']
                    if not any([c.exists() for c in intr_candidates]):
                        for cand in tmp_dir.rglob('*.json'):
                            try:
                                txt = cand.read_text(encoding='utf-8')
                                if 'cx' in txt and 'fx' in txt:
                                    intr_candidates.append(cand)
                                    break
                            except Exception:
                                continue
                    for cand in intr_candidates:
                        if not cand or not Path(cand).exists():
                            continue
                        try:
                            intr_full_try = json.loads(Path(cand).read_text(encoding='utf-8'))
                            for key in ('color_intrinsics', 'intrinsics', 'intrinsics_color', 'camera_intrinsics'):
                                if key in intr_full_try:
                                    intr = intr_full_try.get(key)
                                    intr_source = str(cand)
                                    intr_full = intr_full_try
                                    break
                            if intr is None and isinstance(intr_full_try, dict) and all(k in intr_full_try for k in ('cx', 'cy', 'fx', 'fy')):
                                intr = intr_full_try
                                intr_source = str(cand)
                                intr_full = intr_full_try
                            if intr is not None:
                                break
                        except Exception:
                            continue
                except Exception:
                    intr = None

                # 깊이 파일 샘플 검증(최대 10개): dtype/shape/유효 픽셀 등 점검
                depth_validation = []
                try:
                    if depth_count == 0:
                        # persist validation info for debugging
                        (Path(dest_dir) / 'depth_validation.json').write_text(json.dumps({'error': 'no_npy_files'}, indent=2))
                        raise RuntimeError('No depth .npy files found in depth directory')

                    # choose up to 10 samples evenly spaced
                    sample_n = min(10, depth_count)
                    if sample_n <= 0:
                        sample_idxs = []
                    else:
                        step = max(1, depth_count // sample_n)
                        sample_idxs = list(range(0, depth_count, step))[:sample_n]

                    invalid_samples = 0
                    for idx in sample_idxs:
                        p = depth_files[idx]
                        stat = {'name': p.name}
                        try:
                            arr = np.load(str(p))
                            stat['dtype'] = str(arr.dtype)
                            stat['shape'] = arr.shape
                            total = int(arr.size)
                            finite = np.isfinite(arr)
                            finite_count = int(finite.sum())
                            valid = finite & (arr > 0)
                            valid_count = int(valid.sum())
                            stat['total_pixels'] = total
                            stat['finite_count'] = finite_count
                            stat['valid_count'] = valid_count
                            stat['min'] = float(np.nanmin(arr[finite])) if finite.any() else None
                            stat['max'] = float(np.nanmax(arr[finite])) if finite.any() else None
                            stat['median_valid'] = float(np.nanmedian(arr[valid])) if valid.any() else None
                            if valid_count == 0:
                                invalid_samples += 1
                        except Exception as e:
                            stat['error'] = str(e)
                            invalid_samples += 1
                        depth_validation.append(stat)

                    # persist validation info for debugging
                    try:
                        (Path(dest_dir) / 'depth_validation.json').write_text(json.dumps(depth_validation, indent=2))
                    except Exception:
                        pass

                    # 깊이 스케일 추론(휴리스틱)
                    # - intrinsics.meta.depth_scale 우선
                    # - 샘플 통계가 크면 mm→m(0.001)로 가정
                    depth_scale_factor = 1.0
                    inferred_unit = 'meters'
                    try:
                        if 'intr_full' in locals() and isinstance(intr_full, dict):
                            meta = intr_full.get('meta') or intr_full.get('meta_data') or {}
                            if isinstance(meta, dict) and 'depth_scale' in meta:
                                # intr_full may have depth_scale if raw depth was saved; prefer it
                                try:
                                    ds = float(meta.get('depth_scale'))
                                    if ds != 0 and ds != 1.0:
                                        depth_scale_factor = ds
                                        inferred_unit = f'raw_scale_{ds}'
                                except Exception:
                                    pass
                        # fallback heuristic based on sample stats
                        if depth_scale_factor == 1.0:
                            for s in depth_validation:
                                md = s.get('median_valid')
                                dt = s.get('dtype','')
                                if (md is not None and md > 1000) or ('int' in dt.lower() and (s.get('max') or 0) > 1000):
                                    depth_scale_factor = 0.001
                                    inferred_unit = 'millimeters'
                                    break
                    except Exception:
                        pass

                    # record inferred scale into depth_map_info for diagnostics
                    try:
                        depth_map_info['inferred_depth_scale'] = depth_scale_factor
                        depth_map_info['inferred_unit'] = inferred_unit
                        if intr_source:
                            depth_map_info['intrinsic_source'] = intr_source
                        (Path(dest_dir) / 'depth_map_info.json').write_text(json.dumps(depth_map_info, indent=2))
                    except Exception:
                        pass

                    # If all sampled depth files are invalid, bail out early with an error log
                    if len(sample_idxs) > 0 and invalid_samples >= len(sample_idxs):
                        err_msg = f"All sampled depth .npy files appear invalid (depth_count={depth_count}). Aborting 3D processing."
                        try:
                            (Path(dest_dir) / 'result_error.txt').write_text(err_msg)
                        except Exception:
                            pass
                        return {'message': 'ERROR', 'detail': err_msg}
                except Exception as e:
                    # If validation discovery itself fails, log and continue to attempt processing
                    try:
                        (Path(dest_dir) / 'depth_validation_error.txt').write_text(traceback.format_exc())
                    except Exception:
                        pass

                # color 디렉터리에 대해 OpenPose 실행
                # openpose.openpose는 --write_images 경로로 dirname을 사용하므로,
                # output_img_dir 하위 파일 경로를 인자로 넘겨 해당 폴더에 이미지가 저장되도록 합니다.
                run_openpose_on_dir(str(color_dir), str(output_json_dir), str(output_img_dir / 'out'))

                # Pair JSON outputs with depth .npy by frame index ordering and build DataFrame
                json_paths = sorted([p for p in output_json_dir.iterdir() if p.suffix == '.json'])
                rows = []
                # Build a sorted list of depth files as a robust fallback mapping
                depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix == '.npy'])
                depth_count = len(depth_files)
                depth_map_info = {'depth_count': depth_count}

                # Try to locate intrinsics in common locations and accept multiple key names
                intr = None
                intr_search_candidates = [tmp_dir / 'intrinsics.json', color_dir / 'intrinsics.json', depth_dir / 'intrinsics.json']
                for cand in intr_search_candidates:
                    if cand.exists():
                        try:
                            intr_full = json.loads(cand.read_text(encoding='utf-8'))
                            # common key names used by recorder tools
                            for key in ('color_intrinsics', 'intrinsics', 'intrinsics_color', 'camera_intrinsics'):
                                if key in intr_full:
                                    intr = intr_full.get(key)
                                    intr_source = str(cand)
                                    break
                            # fallback: if the json itself looks like the intrinsics dict
                            if intr is None and isinstance(intr_full, dict) and all(k in intr_full for k in ('cx', 'cy', 'fx', 'fy')):
                                intr = intr_full
                                intr_source = str(cand)
                            if intr is not None:
                                break
                        except Exception:
                            # ignore parse errors and continue searching
                            intr = None

                # Save a small debug listing into dest_dir so operators can inspect
                try:
                    dbg = {
                        'color_files': [p.name for p in sorted(color_dir.iterdir())[:50]],
                        'depth_files': [p.name for p in depth_files[:50]],
                        'intrinsic_source': intr_source if 'intr_source' in locals() else (intr_source if 'intr_source' not in locals() and 'intr_source' in globals() else None),
                    }
                    # write to dest_dir (persist across tmpdir removal)
                    try:
                        (Path(dest_dir) / 'debug_file_listing.json').write_text(json.dumps(dbg, indent=2))
                    except Exception:
                        pass
                except Exception:
                    pass

                # per-frame debug entries to help diagnose NaN Z values
                frame_depth_debug = []

                for frame_idx, jp in enumerate(json_paths):
                    # depth file expected to be zero-padded with same index (e.g., 000001.npy)
                    depth_path = depth_dir / f"{frame_idx:06d}.npy"
                    if not depth_path.exists():
                        # try without zero-pad pattern match
                        depth_candidates = sorted(depth_dir.glob(f"*{frame_idx}*.npy"))
                        if depth_candidates:
                            depth_path = depth_candidates[0]
                        else:
                            # fallback: if number of depth files equals number of json frames or at least > frame_idx,
                            # map by sorted order (assume same ordering)
                            if depth_count > frame_idx:
                                depth_path = depth_files[frame_idx]
                                depth_map_info[f'frame_{frame_idx}'] = f'used_index_map:{depth_path.name}'
                            else:
                                # explicit error with debug info
                                raise FileNotFoundError(f"Depth not found for frame {frame_idx} in {depth_dir} (depth_count={depth_count})")
                    # load depth array; guard against malformed files
                    try:
                        depth_m = np.load(str(depth_path))
                    except Exception:
                        # register debug info and continue (Z will be NaN)
                        depth_m = None

                    with jp.open('r', encoding='utf-8') as f:
                        jdata = json.load(f)
                    raw_people = jdata.get('people', [])
                    ppl = []
                    for person_idx, p in enumerate(raw_people):
                        if 'pose_keypoints_2d' in p:
                            kps = p['pose_keypoints_2d']
                            person = [kps[idx:idx+3] for idx in range(0, len(kps), 3)]
                            person = _sanitize_person_list(person)
                            ppl.append(person)
                    result_by_frame.append(ppl)

                    # compute 3D coordinates using intrinsics (found earlier in the tmp tree)
                    # NOTE: do not reassign `intr` here; it was discovered earlier and should persist

                    for person_idx, person in enumerate(ppl):
                        for joint_idx, (x, y, c) in enumerate(person):
                            X, Y, Zm = (np.nan, np.nan, np.nan)
                            sample_info = {
                                'frame': frame_idx,
                                'json_name': jp.name,
                                'depth_name': depth_path.name if depth_path is not None else None,
                                'person_idx': person_idx,
                                'joint_idx': joint_idx,
                                'x': float(x),
                                'y': float(y),
                                'conf': float(c),
                                'intrinsics_present': bool(intr),
                                'patch_shape': None,
                                'vals_count': 0,
                                'Z_median': None,
                            }
                            if intr is not None and depth_m is not None:
                                xi, yi = int(round(x)), int(round(y))
                                # robust depth sampling (patch median)
                                try:
                                    H, W = depth_m.shape[:2]
                                    r = 2
                                    x0, x1 = max(0, xi-r), min(W, xi+r+1)
                                    y0, y1 = max(0, yi-r), min(H, yi+r+1)
                                    patch = depth_m[y0:y1, x0:x1]
                                    sample_info['patch_shape'] = patch.shape
                                    vals = patch[np.isfinite(patch) & (patch > 0)]
                                    sample_info['vals_count'] = int(vals.size)
                                    Z = float(np.median(vals)) if vals.size else np.nan
                                    sample_info['Z_median'] = None if (np.isnan(Z) or not np.isfinite(Z)) else float(Z)
                                    if sample_info['Z_median'] is not None:
                                        Zm = float(sample_info['Z_median'])
                                        # coerce intrinsics into floats and guarded access
                                        try:
                                            cx = float(intr.get('cx'))
                                            cy = float(intr.get('cy'))
                                            fx = float(intr.get('fx'))
                                            fy = float(intr.get('fy'))
                                            # (original behavior) do not modify Zm here; use raw sampled depth units
                                            X = (x - cx) * Zm / fx
                                            Y = (y - cy) * Zm / fy
                                        except Exception:
                                            # leave X/Y as NaN if intrinsics malformed
                                            pass
                                except Exception:
                                    # if anything goes wrong sampling, keep NaNs and record minimal info
                                    pass
                            # append tidy row
                            rows.append({
                                'frame': frame_idx,
                                'person_idx': person_idx,
                                'joint_idx': joint_idx,
                                'x': float(x), 'y': float(y), 'conf': float(c),
                                'X': float(X) if not np.isnan(X) else np.nan,
                                'Y': float(Y) if not np.isnan(Y) else np.nan,
                                'Z': float(Zm) if not np.isnan(Zm) else np.nan,
                            })
                            # record sample debug for first N frames or all (keeps small)
                            if frame_idx < 200:  # limit debug size
                                frame_depth_debug.append(sample_info)

                # persist depth mapping info and per-frame depth debug to dest_dir for diagnostics
                try:
                    dm_path = Path(dest_dir) / 'depth_map_info.json'
                    dm = depth_map_info.copy()
                    dm['sample'] = depth_files[:50]
                    dm_path.write_text(json.dumps(dm, indent=2))
                except Exception:
                    pass

                try:
                    dbg_path = Path(dest_dir) / 'frame_depth_debug.json'
                    dbg_path.write_text(json.dumps(frame_depth_debug, indent=2))
                except Exception:
                    pass

                df_3d = pd.DataFrame(rows)

                # Build a 2D tidy DataFrame (frame, person_idx, joint_idx, x, y, conf)
                # using the parsed OpenPose JSON (result_by_frame) so overlay uses pure 2D pixel coords.
                try:
                    rows2 = []
                    for frame_idx, ppl in enumerate(result_by_frame):
                        for person_idx, person in enumerate(ppl):
                            for joint_idx, (x, y, c) in enumerate(person):
                                rows2.append({'frame': frame_idx, 'person_idx': person_idx, 'joint_idx': joint_idx, 'x': float(x), 'y': float(y), 'conf': float(c)})
                    df_2d = pd.DataFrame(rows2)
                except Exception:
                    df_2d = pd.DataFrame()

                # Also produce 'wide' DataFrames expected by metric modules: wide2 (2D pixels) and wide3 (3D X/Y/Z)
                try:
                    from metric_algorithm.runner_utils import tidy_to_wide
                    wide2 = tidy_to_wide(df_2d, dimension='2d', person_idx=0) if (not df_2d.empty) else pd.DataFrame()
                except Exception:
                    wide2 = pd.DataFrame()

                try:
                    # df_3d already contains rows with X,Y,Z columns - convert to wide using tidy_to_wide
                    # The tidy rows used for df_3d are in variable 'rows' above
                    df3_tidy = pd.DataFrame(rows)
                    wide3 = tidy_to_wide(df3_tidy, dimension='3d', person_idx=0) if (not df3_tidy.empty) else pd.DataFrame()
                except Exception:
                    wide3 = pd.DataFrame()

                # Persist original RGB frames into dest_dir/img BEFORE tmpdir is removed so operators
                # and metric modules can access them. For 2D input we extract frames from the input MP4;
                # for 3D input we copy the recorder's color/ images. Keep OpenPose rendered images (if any)
                # available under dest_dir/openpose_img for debugging.
                try:
                    dest_img_dir = Path(dest_dir) / 'img'
                    dest_img_dir.mkdir(parents=True, exist_ok=True)

                    # Prefer original color frames (if present). For 3D runs color_dir contains originals.
                    try:
                        # If color_dir exists in tmp tree, copy those images as the canonical RGB frames
                        if 'color_dir' in locals() and Path(color_dir).exists():
                            for p in sorted(Path(color_dir).iterdir()):
                                try:
                                    if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                                        shutil.copy2(str(p), str(dest_img_dir / p.name))
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # If we have an input mp4 (2D path), try to extract frames into dest/img so overlays draw on RGB
                    try:
                        if 'local_video' in locals() and Path(local_video).exists():
                            try:
                                import cv2 as _cv2
                                cap = _cv2.VideoCapture(str(local_video))
                                idx = 0
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    outp = dest_img_dir / f"{idx:06d}.png"
                                    try:
                                        _cv2.imwrite(str(outp), frame)
                                    except Exception:
                                        # fallback: skip frame write
                                        pass
                                    idx += 1
                                cap.release()
                            except Exception:
                                # if OpenCV missing or fails, fall back to copying OpenPose images below
                                pass
                    except Exception:
                        pass

                    # If no original frames were copied/extracted, fall back to copying OpenPose images
                    # but DO NOT copy OpenPose-rendered images (they are stored separately in openpose_img).
                    try:
                        has_rgb = any(p for p in dest_img_dir.iterdir())
                    except Exception:
                        has_rgb = False
                    if not has_rgb and output_img_dir.exists():
                        for p in sorted(output_img_dir.iterdir()):
                            try:
                                if p.is_file():
                                    lname = p.name.lower()
                                    # skip rendered/debug images (these are persisted under openpose_img)
                                    if 'render' in lname or 'openpose' in lname:
                                        continue
                                    shutil.copy2(str(p), str(dest_img_dir / p.name))
                            except Exception:
                                pass

                    # Also persist OpenPose rendered images into dest_dir/openpose_img for debugging
                    try:
                        openpose_dest = Path(dest_dir) / 'openpose_img'
                        openpose_dest.mkdir(parents=True, exist_ok=True)
                        if output_img_dir.exists():
                            for p in sorted(output_img_dir.iterdir()):
                                try:
                                    if p.is_file():
                                        shutil.copy2(str(p), str(openpose_dest / p.name))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    pass

                # If no overlay mp4 yet, create a simple mp4 from copied images (written into dest_dir)
                try:
                    # Prefer using original color frames if available (avoid using OpenPose-rendered images)
                    imgs = []
                    try:
                        # prefer color_dir (original frames) when present
                        if 'color_dir' in locals() and Path(color_dir).exists():
                            imgs = sorted([p for p in Path(color_dir).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                            src_desc = 'color_dir'
                        else:
                            imgs = sorted([p for p in dest_img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]) if dest_img_dir.exists() else []
                            src_desc = 'dest_img_dir'
                        # deduplicate by name while preserving order
                        seen = set()
                        uniq = []
                        for p in imgs:
                            if p.name in seen:
                                continue
                            seen.add(p.name)
                            uniq.append(p)
                        imgs = uniq
                    except Exception:
                        imgs = []
                        src_desc = 'none'

                    if imgs:
                        try:
                            from metric_algorithm.runner_utils import images_to_mp4
                            out_mp4 = Path(dest_dir) / f"{job_id}_overlay.mp4"
                            created, used = images_to_mp4(imgs, out_mp4, fps=30.0, resize=None, filter_rendered=True, write_debug=True)
                            if created:
                                try:
                                    (Path(dest_dir) / 'overlay_debug.json').write_text(json.dumps({'overlay_source': src_desc, 'images_used': used}), encoding='utf-8')
                                except Exception:
                                    pass
                        except Exception:
                            # ignore if helper missing or fails
                            pass
                except Exception:
                    pass

            # end of tmp work; we intentionally DO NOT remove tmp_dir here so files remain for debugging
            # If automatic cleanup is desired, uncomment the following line to remove the tmp dir:
            # shutil.rmtree(str(tmp_dir))

        # after temp work is done, validate dimension and build interpolated sequence
        if dimension not in ('2d', '3d'):
            raise RuntimeError(f'Unsupported dimension: {dimension}')

        # pick first person per frame (or empty list) and interpolate
        sequence = [(ppl[0] if (ppl and len(ppl) > 0) else []) for ppl in result_by_frame]
        interpolated = interpolate_sequence(sequence, conf_thresh=0.0, method='linear', fill_method='none')
        people_sequence = [([person] if person else []) for person in interpolated]

        response_payload = {
            'message': 'OK',
            'people_sequence': people_sequence,
            'frame_count': len(people_sequence)
        }

        # include dimension and prepare a prefixed basename for all job-derived files
        try:
            response_payload['dimension'] = dimension
        except Exception:
            pass
        result_basename = f"{dimension}_{job_id}"

        # Attach minimal DataFrame summary (not full CSV) for downstream verification.
        # DataFrames (df_2d or df_3d) are kept in memory for metrics; here we include simple metadata.
        try:
            if dimension == '2d':
                response_payload['skeleton_rows'] = len(df_2d)
                response_payload['skeleton_columns'] = list(df_2d.columns)
            elif dimension == '3d':
                response_payload['skeleton_rows'] = len(df_3d)
                response_payload['skeleton_columns'] = list(df_3d.columns)
        except Exception:
            pass

        # write a debug/partial JSON early so operators can inspect intermediate state
        partial_out_path = Path(dest_dir) / f"{result_basename}.partial.json"
        out_path = Path(dest_dir) / f"{result_basename}.json"

        # If an async MMACTION thread was started, wait briefly for its response file
        try:
            mmaction_resp_path = None
            dbg = response_payload.get('debug', {})
            # candidates for response path: earlier helpers set mmaction_start.response_path or use standard name
            if isinstance(dbg.get('mmaction_start'), dict) and dbg['mmaction_start'].get('response_path'):
                mmaction_resp_path = Path(dbg['mmaction_start'].get('response_path'))
            # fallback to canonical path in dest_dir (prefixed)
            if mmaction_resp_path is None:
                mmaction_resp_path = Path(dest_dir) / f"{result_basename}_stgcn_response.json"

            # only wait if thread started flag is present
            thread_started = dbg.get('mmaction_thread_started') or dbg.get('mmaction_thread_started_later') or response_payload.setdefault('debug', {}).get('mmaction_thread_started')
            if thread_started:
                # allow override via env var
                try:
                    wait_secs = int(os.environ.get('MMACTION_WAIT_SECONDS', '30'))
                except Exception:
                    wait_secs = 30
                # poll for file existence with small sleep intervals
                import time
                elapsed = 0
                interval = 0.5
                while elapsed < wait_secs:
                    if mmaction_resp_path.exists():
                        break
                    time.sleep(interval)
                    elapsed += interval
                # if response file appeared, merge into response_payload
                try:
                    if mmaction_resp_path.exists():
                        try:
                            resp_txt = mmaction_resp_path.read_text(encoding='utf-8')
                            resp_obj = json.loads(resp_txt)
                        except Exception:
                            resp_obj = None
                        if resp_obj:
                            # if the POST wrapper returned a 'resp_json' with 'result', merge it
                            rj = resp_obj.get('resp_json') if isinstance(resp_obj, dict) else None
                            if isinstance(rj, dict) and 'result' in rj:
                                response_payload['stgcn_inference'] = rj['result']
                            else:
                                # if worker wrote full result at top-level, keep it
                                response_payload['stgcn_inference'] = resp_obj
                except Exception:
                    pass
        except Exception:
            pass

        try:
            _safe_write_json(partial_out_path, response_payload)
        except Exception:
            # fallback to normal write
            try:
                _safe_write_json(out_path, response_payload)
            except Exception:
                pass

        # Persist OpenPose rendered images into dest_dir/openpose_img for debugging and
        # avoid copying them into dest_dir/img to prevent mixing rendered + original frames.
        try:
            openpose_dest = Path(dest_dir) / 'openpose_img'
            openpose_dest.mkdir(parents=True, exist_ok=True)
            for p in sorted(output_img_dir.iterdir() if output_img_dir.exists() else []):
                try:
                    if p.is_file():
                        # Always copy rendered/openpose images into openpose_img for diagnostics
                        shutil.copy2(str(p), str(openpose_dest / p.name))
                except Exception:
                    pass
        except Exception:
            pass

        # Optionally run local metrics using the in-memory DataFrames; run_metrics_in_process will write CSVs into dest_dir
        metrics_res = None
        try:
            metrics_res = run_metrics_in_process(dimension, locals())
        except Exception:
            # metrics failures shouldn't take down processing
            traceback.print_exc()

        # If the metric runner produced structured output, merge it into the job result JSON
        try:
            if isinstance(metrics_res, dict):
                # Get per-module payload
                metrics_payload = metrics_res.get('metrics') if 'metrics' in metrics_res else metrics_res

                cleaned_metrics = {}
                for mname, mval in (metrics_payload.items() if isinstance(metrics_payload, dict) else []):
                    # If module returned non-dict (e.g., error string), keep as-is
                    if not isinstance(mval, dict):
                        cleaned_metrics[mname] = mval
                        continue

                    cleaned = {}

                    # Keep summary if available
                    if 'summary' in mval and isinstance(mval['summary'], dict):
                        cleaned['summary'] = mval['summary']

                    # Collect CSV paths from values (strings or lists)
                    csv_paths = []
                    for k, v in mval.items():
                        # skip overlay/mp4 etc.
                        if k.lower().startswith('overlay'):
                            continue
                        if isinstance(v, str) and v.lower().endswith('.csv'):
                            p = Path(v)
                            if not p.exists():
                                # try relative to dest_dir
                                p = Path(dest_dir) / v
                            if p.exists():
                                csv_paths.append(p)
                        elif isinstance(v, (list, tuple)):
                            for it in v:
                                if isinstance(it, str) and it.lower().endswith('.csv'):
                                    p = Path(it)
                                    if not p.exists():
                                        p = Path(dest_dir) / it
                                    if p.exists():
                                        csv_paths.append(p)

                    # If we found CSVs, read and merge them into frame_data
                    if csv_paths:
                        try:
                            dfs = []
                            for p in csv_paths:
                                try:
                                    df = pd.read_csv(p)
                                except Exception:
                                    # skip unreadable csv
                                    continue
                                # normalize frame column name
                                if 'frame' not in df.columns and 'Frame' in df.columns:
                                    df = df.rename(columns={'Frame': 'frame'})
                                if 'frame' in df.columns:
                                    df = df.set_index('frame')
                                else:
                                    # use row order as frame index
                                    df.index.name = 'frame'
                                dfs.append(df)
                            if dfs:
                                # merge on index (frame)
                                from functools import reduce
                                def _join(a, b):
                                    return a.join(b, how='outer', lsuffix='_l', rsuffix='_r')
                                merged = reduce(_join, dfs) if len(dfs) > 1 else dfs[0]

                                frame_data = {}
                                for idx, row in merged.iterrows():
                                    # cast index to int if possible
                                    try:
                                        fkey = int(idx)
                                    except Exception:
                                        fkey = str(idx)
                                    rowd = {}
                                    for col, val in row.items():
                                        if pd.isna(val):
                                            rowd[col] = None
                                        else:
                                            # convert numpy scalar to python type
                                            try:
                                                if hasattr(val, 'item'):
                                                    val = val.item()
                                            except Exception:
                                                pass
                                            rowd[col] = val
                                    frame_data[str(fkey)] = rowd
                                cleaned['frame_data'] = frame_data
                        except Exception:
                            # non-fatal — include raw csv paths if conversion failed
                            cleaned['frame_data_error'] = 'failed to read/convert csv'

                    # Keep overlay info (s3 or local) if present
                    if 'overlay_mp4' in mval:
                        cleaned['overlay_mp4'] = mval.get('overlay_mp4')
                    if 'overlay_s3' in mval:
                        cleaned['overlay_s3'] = mval.get('overlay_s3')

                    # If no summary and no frame_data, preserve original (may include error info)
                    if not cleaned:
                        cleaned_metrics[mname] = mval
                    else:
                        cleaned_metrics[mname] = cleaned

                response_payload['metrics'] = cleaned_metrics

                # Note: metrics result file paths are intentionally not injected into the
                # top-level response payload (consumer expects metrics under 'metrics').

                # overwrite the job json on disk with enriched payload so upload picks it up
                try:
                    out_path = Path(dest_dir) / f"{job_id}.json"
                    # Only persist allowed top-level keys in the job JSON to match
                    # expected consumer schema (avoid injecting extraneous top-level fields).
                    allowed = ('frame_count', 'dimension', 'skeleton_rows', 'skeleton_columns', 'debug', 'metrics', 'stgcn_inference')
                    filtered = {k: response_payload[k] for k in allowed if k in response_payload}
                    _safe_write_json(out_path, filtered)
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()
        # --- Prepare and persist canonical skeleton2d.csv in dest_dir for MMACTION client ---
        try:
            # prefer df_2d wide conversion produced earlier by tidy_to_wide or recreate from df_2d
            try:
                from metric_algorithm.runner_utils import tidy_to_wide
                wide2 = tidy_to_wide(df_2d, dimension='2d', person_idx=0) if (isinstance(df_2d, pd.DataFrame) and not df_2d.empty) else (wide2 if 'wide2' in locals() else pd.DataFrame())
            except Exception:
                wide2 = wide2 if 'wide2' in locals() else pd.DataFrame()

            # Normalize column names/order to canonical skeleton2d.csv ordering
            if isinstance(wide2, pd.DataFrame) and not wide2.empty:
                # ensure columns are in the exact order as sample skeleton2d.csv
                COCO_ORDER = [
                    'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
                ]
                cols = []
                for j in COCO_ORDER:
                    cols.extend([f"{j}_x", f"{j}_y", f"{j}_c"])
                # Build a canonical skeleton2d.csv with exact COCO columns (Nose..RAnkle + _x/_y/_c)
                try:
                    COCO_ORDER = [
                        'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
                    ]
                    cols_canonical = []
                    for j in COCO_ORDER:
                        cols_canonical.extend([f"{j}_x", f"{j}_y", f"{j}_c"])

                    # Prepare lower-case lookup for available aliases in wide2
                    records = wide2.to_dict(orient='records') if not wide2.empty else []
                    ske_rows = []
                    for rec in records:
                        # build a lowercased key->value map to match aliases case-insensitively
                        lc_map = {str(k).lower(): v for k, v in rec.items()}

                        newr = {}
                        for j in COCO_ORDER:
                            # check common aliases for x
                            x_keys = [f"{j}__x", f"{j}_x", f"{j}_X", f"{j}__X"]
                            y_keys = [f"{j}__y", f"{j}_y", f"{j}_Y", f"{j}__Y"]
                            c_keys = [f"{j}__c", f"{j}_c", f"{j}_conf", f"{j}_score"]
                            # lowercase-map versions
                            x_val = None
                            y_val = None
                            c_val = None
                            for k in x_keys:
                                if k.lower() in lc_map:
                                    x_val = lc_map[k.lower()]
                                    break
                            for k in y_keys:
                                if k.lower() in lc_map:
                                    y_val = lc_map[k.lower()]
                                    break
                            for k in c_keys:
                                if k.lower() in lc_map:
                                    c_val = lc_map[k.lower()]
                                    break
                            # coerce to floats or NaN-like None
                            try:
                                newr[f"{j}_x"] = float(x_val) if x_val is not None else ''
                            except Exception:
                                newr[f"{j}_x"] = ''
                            try:
                                newr[f"{j}_y"] = float(y_val) if y_val is not None else ''
                            except Exception:
                                newr[f"{j}_y"] = ''
                            try:
                                newr[f"{j}_c"] = float(c_val) if c_val is not None else ''
                            except Exception:
                                newr[f"{j}_c"] = ''
                        ske_rows.append(newr)

                    ske_df = pd.DataFrame(ske_rows, columns=cols_canonical)
                    ske_path = Path(dest_dir) / 'skeleton2d.csv'
                    ske_df.to_csv(ske_path, index=False)
                    response_payload.setdefault('debug', {})['mmaction_input_csv'] = str(ske_path)
                    try:
                        mmaction_start = mmaction_client.start_mmaction_from_csv(ske_path, dest_dir, job_id, dimension, response_payload)
                        response_payload.setdefault('debug', {})['mmaction_start'] = mmaction_start
                        # If the client returned a thread handle, attempt to join it (with timeout)
                        try:
                            th = None
                            if isinstance(mmaction_start, dict):
                                th = mmaction_start.get('thread')
                            # also consider debug flags that indicate a later-started thread
                            if th and hasattr(th, 'join'):
                                try:
                                    # allow a short timeout to avoid blocking long-running inference
                                    join_timeout = float(os.environ.get('MMACTION_JOIN_SECONDS', '5'))
                                except Exception:
                                    join_timeout = 5.0
                                try:
                                    th.join(timeout=join_timeout)
                                except Exception:
                                    pass
                                # after join (or timeout), read response file and merge (prefixed name)
                                try:
                                    resp_path = Path(dest_dir) / f"{result_basename}_stgcn_response.json"
                                    if resp_path.exists():
                                        try:
                                            resp_txt = resp_path.read_text(encoding='utf-8')
                                            resp_obj = json.loads(resp_txt)
                                        except Exception:
                                            resp_obj = None
                                        if resp_obj:
                                            rj = resp_obj.get('resp_json') if isinstance(resp_obj, dict) else None
                                            if isinstance(rj, dict) and 'result' in rj:
                                                response_payload['stgcn_inference'] = rj['result']
                                            else:
                                                response_payload['stgcn_inference'] = resp_obj
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        response_payload.setdefault('debug', {})['mmaction_start_error'] = True
                except Exception:
                    response_payload.setdefault('debug', {})['mmaction_skeleton_write_error'] = True
        except Exception:
            traceback.print_exc()

        # If metrics did not create an overlay mp4, create a minimal overlay from copied images
        try:
            # look for any mp4 in dest_dir
            mp4s = list(Path(dest_dir).glob(f"{job_id}*.mp4"))
            if not mp4s:
                img_dir_check = Path(dest_dir) / 'img'
                # prefer color_dir when available (color_dir may not be in this scope, try to detect)
                imgs = []
                try:
                    if 'color_dir' in locals() and Path(color_dir).exists():
                        imgs = sorted([p for p in Path(color_dir).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                        src_desc = 'color_dir'
                    else:
                        imgs = sorted([p for p in img_dir_check.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]) if img_dir_check.exists() else []
                        src_desc = 'dest_img_dir'
                    # dedupe while preserving order
                    seen = set()
                    uniq = []
                    for p in imgs:
                        if p.name in seen:
                            continue
                        seen.add(p.name)
                        uniq.append(p)
                    imgs = uniq
                except Exception:
                    imgs = []
                    src_desc = 'none'

                if imgs:
                    try:
                        from metric_algorithm.runner_utils import images_to_mp4
                        out_mp4 = Path(dest_dir) / f"{job_id}_overlay.mp4"
                        created, used = images_to_mp4(imgs, out_mp4, fps=30.0, resize=None, filter_rendered=True, write_debug=True)
                        if created:
                            try:
                                (Path(dest_dir) / 'overlay_debug.json').write_text(json.dumps({'overlay_source': src_desc, 'images_used': used}), encoding='utf-8')
                            except Exception:
                                pass
                    except Exception:
                        # ignore if helper missing or fails
                        pass
        except Exception:
            pass

        # --- Reorganize received_payloads layout: move CSVs -> csv/, MP4s -> mp4/
        try:
            csv_dir = Path(dest_dir) / 'csv'
            mp4_dir = Path(dest_dir) / 'mp4'
            csv_dir.mkdir(parents=True, exist_ok=True)
            mp4_dir.mkdir(parents=True, exist_ok=True)

            # Move CSV files from dest root into csv/
            for p in sorted(Path(dest_dir).glob('*.csv')):
                try:
                    target = csv_dir / p.name
                    if target.exists():
                        # avoid overwrite; keep existing
                        p.unlink()
                    else:
                        p.replace(target)
                except Exception:
                    # non-fatal
                    pass

            # Move MP4 files from dest root into mp4/
            for p in sorted(Path(dest_dir).glob('*.mp4')):
                try:
                    target = mp4_dir / p.name
                    if target.exists():
                        p.unlink()
                    else:
                        p.replace(target)
                except Exception:
                    pass

            # Update response_payload.metrics overlay paths to be relative into mp4/ if files were moved
            try:
                if 'metrics' in response_payload and isinstance(response_payload['metrics'], dict):
                    for mname, mval in response_payload['metrics'].items():
                        if isinstance(mval, dict) and 'overlay_mp4' in mval and mval.get('overlay_mp4'):
                            ov = mval.get('overlay_mp4')
                            try:
                                ovp = Path(ov)
                                if not ovp.exists():
                                    # maybe the file was moved to mp4/
                                    cand = mp4_dir / ovp.name
                                    if cand.exists():
                                        # set relative path so downstream readers resolve against dest_dir
                                        mval['overlay_mp4'] = str(Path('mp4') / cand.name)
                                else:
                                    # file still at root; move it into mp4/ (already attempted) and update path
                                    mval['overlay_mp4'] = str(Path('mp4') / ovp.name)
                            except Exception:
                                # leave as-is if anything goes wrong
                                pass
            except Exception:
                pass

            # rewrite job json with updated relative paths
            try:
                out_path = Path(dest_dir) / f"{result_basename}.json"
                if not out_path.exists():
                    out_path = Path(dest_dir) / f"{job_id}.json"
                _safe_write_json(out_path, response_payload)
            except Exception:
                pass
            # --- Enrich missing per-metric data from metric CSVs in csv/ ---
            try:
                # look for metric CSVs (prefixed or legacy) and populate frame_data/summary
                def _read_metrics_csv(candidate: Path):
                    try:
                        df = pd.read_csv(candidate)
                        # normalize frame column
                        if 'Frame' in df.columns and 'frame' not in df.columns:
                            df = df.rename(columns={'Frame': 'frame'})
                        if 'frame' in df.columns:
                            df = df.set_index('frame')
                        else:
                            df.index.name = 'frame'
                        frame_data = {}
                        for idx, row in df.iterrows():
                            try:
                                fkey = int(idx)
                            except Exception:
                                fkey = str(idx)
                            rowd = {}
                            for col, val in row.items():
                                if pd.isna(val):
                                    rowd[col] = None
                                else:
                                    try:
                                        if hasattr(val, 'item'):
                                            val = val.item()
                                    except Exception:
                                        pass
                                    rowd[col] = val
                            frame_data[str(fkey)] = rowd
                        return frame_data
                    except Exception:
                        return None

                for metric_name in ('com_speed', 'swing_speed'):
                    try:
                        # prefer prefixed csv: '<dimension>_<job_id>_<metric>_metrics.csv'
                        pref = None
                        for p in csv_dir.iterdir():
                            if p.is_file() and p.name.endswith(f"_{job_id}_{metric_name}_metrics.csv"):
                                pref = p
                                break
                        if pref is None:
                            # fallback legacy
                            legacy = csv_dir / f"{job_id}_{metric_name}_metrics.csv"
                            pref = legacy if legacy.exists() else None
                        if pref and pref.exists():
                            frame_data = _read_metrics_csv(pref)
                            if frame_data:
                                response_payload.setdefault('metrics', {}).setdefault(metric_name, {})['frame_data'] = frame_data
                                # if the metric has a separate summary JSON, try to attach it
                                summary_pref = Path(dest_dir) / f"{metric_name}_summary.json"
                                # also try metric-specific patterns
                                maybe_summary = Path(dest_dir) / f"{job_id}_{metric_name}_summary.json"
                                if summary_pref.exists():
                                    try:
                                        response_payload['metrics'][metric_name]['summary'] = json.loads(summary_pref.read_text(encoding='utf-8'))
                                    except Exception:
                                        pass
                                elif maybe_summary.exists():
                                    try:
                                        response_payload['metrics'][metric_name]['summary'] = json.loads(maybe_summary.read_text(encoding='utf-8'))
                                    except Exception:
                                        pass
                    except Exception:
                        pass
            except Exception:
                pass
            # --- Also try to read combined metric_result.json and merge per-metric data ---
            try:
                metrics_result_path = None
                for p in Path(dest_dir).iterdir():
                    if p.is_file() and p.name.endswith(f"_{job_id}_metric_result.json"):
                        metrics_result_path = p
                        break
                if metrics_result_path is None:
                    legacy_mr = Path(dest_dir) / f"{job_id}_metric_result.json"
                    if legacy_mr.exists():
                        metrics_result_path = legacy_mr
                if metrics_result_path and metrics_result_path.exists():
                    try:
                        mr = json.loads(metrics_result_path.read_text(encoding='utf-8'))
                        # mr may be structured with per-metric dicts
                        if isinstance(mr, dict):
                            for mname, mval in (mr.get('metrics') or mr).items():
                                try:
                                    if mname not in response_payload.setdefault('metrics', {}):
                                        # no existing entry: copy entire metric dict
                                        response_payload['metrics'][mname] = mval
                                        continue

                                    # merge into existing metric entry
                                    tgt = response_payload['metrics'][mname]
                                    if not isinstance(mval, dict):
                                        # non-dict value: only set if missing
                                        if mname not in response_payload['metrics']:
                                            response_payload['metrics'][mname] = mval
                                        continue

                                    # 1) summary: prefer metric_result's summary (overwrite or set)
                                    if 'summary' in mval and isinstance(mval.get('summary'), dict):
                                        try:
                                            tgt['summary'] = mval['summary']
                                        except Exception:
                                            pass

                                    # 2) frame-wise data: accept either 'frame_data' or convert 'metrics_data' -> 'frame_data'
                                    # If metric_result provides 'frame_data', merge it. If it provides 'metrics_data',
                                    # attempt to extract a primary timeseries (e.g., 'com_speed_timeseries') and convert
                                    try:
                                        # merge existing frame_data if present
                                        if 'frame_data' in mval and isinstance(mval['frame_data'], dict):
                                            tgt_fd = tgt.setdefault('frame_data', {})
                                            for fk, fv in mval['frame_data'].items():
                                                if fk not in tgt_fd:
                                                    tgt_fd[fk] = fv

                                        elif 'metrics_data' in mval and isinstance(mval['metrics_data'], dict):
                                            # pick the most likely timeseries subkey
                                            md = mval['metrics_data']
                                            timeseries_key = None
                                            for k in md.keys():
                                                if 'timeseries' in k.lower() or 'time_series' in k.lower():
                                                    timeseries_key = k
                                                    break
                                            if timeseries_key is None:
                                                # fallback to first key
                                                keys = list(md.keys())
                                                if keys:
                                                    timeseries_key = keys[0]
                                            if timeseries_key and isinstance(md.get(timeseries_key), dict):
                                                src_ts = md.get(timeseries_key)
                                                tgt_fd = tgt.setdefault('frame_data', {})
                                                for fk, fv in src_ts.items():
                                                    if fk not in tgt_fd:
                                                        tgt_fd[str(fk)] = fv
                                            # also preserve raw metrics_data for completeness
                                            if 'metrics_data' not in tgt:
                                                tgt['metrics_data'] = mval['metrics_data']
                                    except Exception:
                                        pass

                                    # 3) overlay fields: copy or set overlay_mp4/overlay_s3
                                    for ok in ('overlay_mp4', 'overlay_s3'):
                                        try:
                                            if ok in mval and mval.get(ok):
                                                tgt[ok] = mval.get(ok)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            # ensure metrics_path points to the file we read
                            # Do not inject metrics_path into final payload; keep metrics under 'metrics'
                    except Exception:
                        pass
            except Exception:
                pass
            # Verify/repair skeleton2d.csv in csv/ directory: ensure COCO _x/_y/_c columns exist
            try:
                ske_dir = csv_dir
                ske_csv = ske_dir / 'skeleton2d.csv'
                if ske_csv.exists():
                    try:
                        missing = mmaction_client._validate_csv_matches_coco(ske_csv)
                        if missing:
                            # attempt to rebuild from df_2d if available
                            if 'df_2d' in locals() and isinstance(df_2d, pd.DataFrame) and not df_2d.empty:
                                from metric_algorithm.runner_utils import tidy_to_wide
                                try:
                                    wide2_recon = tidy_to_wide(df_2d, dimension='2d', person_idx=0)
                                except Exception:
                                    wide2_recon = pd.DataFrame()
                                if isinstance(wide2_recon, pd.DataFrame) and not wide2_recon.empty:
                                    # recreate canonical skeleton csv rows
                                    COCO_ORDER = [
                                        'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
                                    ]
                                    cols_canonical = [f"{j}_x" for j in COCO_ORDER] + [f"{j}_y" for j in COCO_ORDER] + [f"{j}_c" for j in COCO_ORDER]
                                    # build ske_rows similar to earlier logic
                                    ske_rows = []
                                    for rec in wide2_recon.to_dict(orient='records'):
                                        lc_map = {str(k).lower(): v for k, v in rec.items()}
                                        newr = {}
                                        for j in COCO_ORDER:
                                            x_keys = [f"{j}__x", f"{j}_x", f"{j}_X", f"{j}__X"]
                                            y_keys = [f"{j}__y", f"{j}_y", f"{j}_Y", f"{j}__Y"]
                                            c_keys = [f"{j}__c", f"{j}_c", f"{j}_conf", f"{j}_score"]
                                            x_val = next((lc_map[k.lower()] for k in x_keys if k.lower() in lc_map), None)
                                            y_val = next((lc_map[k.lower()] for k in y_keys if k.lower() in lc_map), None)
                                            c_val = next((lc_map[k.lower()] for k in c_keys if k.lower() in lc_map), None)
                                            try:
                                                newr[f"{j}_x"] = float(x_val) if x_val is not None else ''
                                            except Exception:
                                                newr[f"{j}_x"] = ''
                                            try:
                                                newr[f"{j}_y"] = float(y_val) if y_val is not None else ''
                                            except Exception:
                                                newr[f"{j}_y"] = ''
                                            try:
                                                newr[f"{j}_c"] = float(c_val) if c_val is not None else ''
                                            except Exception:
                                                newr[f"{j}_c"] = ''
                                        ske_rows.append(newr)
                                    ske_df = pd.DataFrame(ske_rows, columns=[f"{j}_{s}" for j in COCO_ORDER for s in ('x','y','c')])
                                    ske_df.to_csv(ske_csv, index=False)
                                    response_payload.setdefault('debug', {})['mmaction_input_csv_rebuilt'] = str(ske_csv)
                                else:
                                    response_payload.setdefault('debug', {})['mmaction_input_rebuild_failed'] = True
                            else:
                                response_payload.setdefault('debug', {})['mmaction_input_missing_df2'] = True
                    except Exception:
                        response_payload.setdefault('debug', {})['mmaction_input_validation_error'] = True
                else:
                    # No skeleton file in csv/ — try to write one from df_2d if present
                    if 'df_2d' in locals() and isinstance(df_2d, pd.DataFrame) and not df_2d.empty:
                        try:
                            from metric_algorithm.runner_utils import tidy_to_wide
                            wide2_recon = tidy_to_wide(df_2d, dimension='2d', person_idx=0)
                        except Exception:
                            wide2_recon = pd.DataFrame()
                        if isinstance(wide2_recon, pd.DataFrame) and not wide2_recon.empty:
                            COCO_ORDER = [
                                'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
                            ]
                            ske_rows = []
                            for rec in wide2_recon.to_dict(orient='records'):
                                lc_map = {str(k).lower(): v for k, v in rec.items()}
                                newr = {}
                                for j in COCO_ORDER:
                                    x_keys = [f"{j}__x", f"{j}_x", f"{j}_X", f"{j}__X"]
                                    y_keys = [f"{j}__y", f"{j}_y", f"{j}_Y", f"{j}__Y"]
                                    c_keys = [f"{j}__c", f"{j}_c", f"{j}_conf", f"{j}_score"]
                                    x_val = next((lc_map[k.lower()] for k in x_keys if k.lower() in lc_map), None)
                                    y_val = next((lc_map[k.lower()] for k in y_keys if k.lower() in lc_map), None)
                                    c_val = next((lc_map[k.lower()] for k in c_keys if k.lower() in lc_map), None)
                                    try:
                                        newr[f"{j}_x"] = float(x_val) if x_val is not None else ''
                                    except Exception:
                                        newr[f"{j}_x"] = ''
                                    try:
                                        newr[f"{j}_y"] = float(y_val) if y_val is not None else ''
                                    except Exception:
                                        newr[f"{j}_y"] = ''
                                    try:
                                        newr[f"{j}_c"] = float(c_val) if c_val is not None else ''
                                    except Exception:
                                        newr[f"{j}_c"] = ''
                                ske_rows.append(newr)
                            ske_df = pd.DataFrame(ske_rows, columns=[f"{j}_{s}" for j in COCO_ORDER for s in ('x','y','c')])
                            ske_csv_parent = ske_csv.parent if 'ske_csv' in locals() else ske_dir
                            ensure_dir = lambda p: p.mkdir(parents=True, exist_ok=True)
                            ensure_dir(ske_csv_parent)
                            ske_df.to_csv(ske_csv_parent / 'skeleton2d.csv', index=False)
                            response_payload.setdefault('debug', {})['mmaction_input_csv_written'] = str(ske_csv_parent / 'skeleton2d.csv')
            except Exception:
                # non-fatal
                pass
        except Exception:
            # non-fatal; continue
            pass

        # --- FINAL WAIT: conservatively wait for metrics + MMACTION/STGCN responses before writing final JSON ---
        try:
            try:
                mmaction_final_wait = int(os.environ.get('MMACTION_FINAL_WAIT_SECONDS', '60'))
            except Exception:
                mmaction_final_wait = 60
            try:
                metrics_final_wait = int(os.environ.get('METRICS_FINAL_WAIT_SECONDS', '60'))
            except Exception:
                metrics_final_wait = 60

            # candidate paths for STGCN response
            resp_pref = Path(dest_dir) / f"{result_basename}_stgcn_response.json"
            resp_legacy = Path(dest_dir) / f"{job_id}_stgcn_response.json"

            # metrics readiness: prefer combined metric_result file, otherwise per-metric artifacts
            metrics_ready = False
            stgcn_ready = False

            import time
            start = time.time()
            interval = 0.5

            def check_metrics_ready():
                # prefer combined metric_result json
                for p in Path(dest_dir).iterdir():
                    if p.is_file() and p.name.endswith(f"_{job_id}_metric_result.json"):
                        return True
                if (Path(dest_dir) / f"{job_id}_metric_result.json").exists():
                    return True
                # fallback: check that per-metric CSVs exist for known metrics
                csv_dir = Path(dest_dir) / 'csv'
                if not csv_dir.exists():
                    return False
                needed = ['com_speed', 'swing_speed']
                for m in needed:
                    found = False
                    for p in csv_dir.iterdir():
                        if p.is_file() and (p.name.endswith(f"_{job_id}_{m}_metrics.csv") or p.name.endswith(f"{m}_metrics.csv") or p.name.endswith(f"{job_id}_{m}_metrics.csv")):
                            found = True
                            break
                    if not found:
                        return False
                return True

            def check_stgcn_ready():
                if resp_pref.exists() or resp_legacy.exists():
                    return True
                return False

            # Wait loop: require both metrics_ready and stgcn_ready or timeouts
            while True:
                now = time.time()
                elapsed = now - start

                # update flags
                if not metrics_ready:
                    metrics_ready = check_metrics_ready()
                if not stgcn_ready:
                    stgcn_ready = check_stgcn_ready()

                # break conditions:
                # - both ready
                if metrics_ready and stgcn_ready:
                    break
                # - metric timeout exceeded and stgcn ready: allow stgcn to proceed
                if stgcn_ready and elapsed >= metrics_final_wait:
                    break
                # - mmaction timeout exceeded and metrics ready
                if metrics_ready and elapsed >= mmaction_final_wait:
                    break
                # - total timeout exceeded (max of both waits)
                if elapsed >= max(metrics_final_wait, mmaction_final_wait):
                    break

                time.sleep(interval)

            # merge STGCN response if present
            try:
                chosen = resp_pref if resp_pref.exists() else (resp_legacy if resp_legacy.exists() else None)
                if chosen:
                    try:
                        resp_txt = chosen.read_text(encoding='utf-8')
                        resp_obj = json.loads(resp_txt)
                    except Exception:
                        resp_obj = None
                    if resp_obj:
                        rj = resp_obj.get('resp_json') if isinstance(resp_obj, dict) else None
                        if isinstance(rj, dict) and 'result' in rj:
                            response_payload['stgcn_inference'] = rj['result']
                        else:
                            response_payload['stgcn_inference'] = resp_obj
            except Exception:
                pass

            # merge metric_result if present (reuse existing logic above) - re-run merge block to gather late files
            try:
                # prefer combined metric_result file
                metrics_result_path = None
                for p in Path(dest_dir).iterdir():
                    if p.is_file() and p.name.endswith(f"_{job_id}_metric_result.json"):
                        metrics_result_path = p
                        break
                if metrics_result_path is None:
                    legacy_mr = Path(dest_dir) / f"{job_id}_metric_result.json"
                    if legacy_mr.exists():
                        metrics_result_path = legacy_mr
                if metrics_result_path and metrics_result_path.exists():
                    try:
                        mr = json.loads(metrics_result_path.read_text(encoding='utf-8'))
                        if isinstance(mr, dict):
                            for mname, mval in (mr.get('metrics') or mr).items():
                                try:
                                    # overwrite or set metric entries with mr content (prefer mr)
                                    response_payload.setdefault('metrics', {})[mname] = mval
                                except Exception:
                                    pass
                            response_payload['metrics_path'] = str(metrics_result_path)
                    except Exception:
                        pass
            except Exception:
                pass

            # Finally write the atomic final JSON (filter top-level keys to the allowed set)
            try:
                final_path = Path(dest_dir) / f"{result_basename}.json"
                if not final_path.exists():
                    final_path = Path(dest_dir) / f"{job_id}.json"
                allowed = ('frame_count', 'dimension', 'skeleton_rows', 'skeleton_columns', 'debug', 'metrics', 'stgcn_inference')
                filtered = {k: response_payload[k] for k in allowed if k in response_payload}
                _safe_write_json(final_path, filtered)
                # remove partial if final written
                try:
                    if partial_out_path.exists():
                        partial_out_path.unlink()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        return response_payload

        # cleanup tmp if desired (do not remove in case debugging needed)
        # shutil.rmtree(tmp_dir)
    except Exception as e:
        # Save error info to dest_dir for debugging
        try:
            err_path = Path(dest_dir) / 'result_error.txt'
            with err_path.open('w', encoding='utf-8') as ef:
                ef.write(traceback.format_exc())
        except Exception:
            pass
        return {'message': 'ERROR', 'detail': str(e)}


def upload_result_to_s3(dest_dir: Path, job_id: str, s3_key: Optional[str] = None, result_bucket: Optional[str] = None):
    """Upload only the canonical job JSON and MP4s under dest_dir/mp4/ to the result S3 bucket.

    Behavior:
    - Upload dest_dir/{job_id}.json (or raise if missing)
    - Upload files under dest_dir/mp4/*.mp4 only
    - When s3_key is provided and looks like '<user>/<dimension>/...': upload under
      '{user}/{dimension}/{job_id}/' preserving filenames (JSON at prefix root, MP4s under prefix/mp4/)
    - Otherwise upload under 'results/' with same structure (results/{job_id}.json and results/mp4/{fname})
    - Default bucket if not provided: 'golf-result-s3'
    """
    try:
        bucket = result_bucket or os.environ.get('S3_RESULT_BUCKET_NAME') or os.environ.get('RESULT_S3_BUCKET') or 'golf-result-s3'
        s3 = boto3.client('s3')

        target_prefix = None
        if s3_key:
            try:
                k = s3_key.lstrip('/')
                parts = k.split('/')
                if len(parts) >= 2:
                    # user/dimension/job_id
                    target_prefix = f"{parts[0]}/{parts[1]}/{job_id}"
            except Exception:
                target_prefix = None

        # canonical job json only
        local_json = Path(dest_dir) / f"{job_id}.json"
        if not local_json.exists():
            raise FileNotFoundError(f'Result file not found: {local_json}')

        uploaded = []
        if target_prefix:
            job_key = f"{target_prefix}/{local_json.name}"
        else:
            job_key = f"results/{local_json.name}"
        s3.upload_file(str(local_json), bucket, job_key)
        uploaded.append({'local': str(local_json), 'bucket': bucket, 'key': job_key})

        # only mp4s under dest_dir/mp4/
        mp4_dir = Path(dest_dir) / 'mp4'
        if mp4_dir.exists() and mp4_dir.is_dir():
            for mp in sorted(mp4_dir.glob('*.mp4')):
                fname = mp.name
                if target_prefix:
                    key = f"{target_prefix}/mp4/{fname}"
                else:
                    key = f"results/mp4/{fname}"
                s3.upload_file(str(mp), bucket, key)
                uploaded.append({'local': str(mp), 'bucket': bucket, 'key': key})

        return {'message': 'UPLOADED', 'files': uploaded}
    except Exception as e:
        try:
            err_path = Path(dest_dir) / 'upload_error.txt'
            with err_path.open('w', encoding='utf-8') as ef:
                ef.write(traceback.format_exc())
        except Exception:
            pass
        return {'message': 'ERROR', 'detail': str(e)}
