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

from openpose.skeleton_interpolate import interpolate_sequence
from openpose.openpose import run_openpose_on_video, run_openpose_on_dir, _sanitize_person_list


def run_metrics_in_process(dimension: str, ctx: dict):
    """Placeholder hook to run in-process metric calculations.

    `ctx` contains local variables including df_2d or df_3d (if produced) and people_sequence.
    This function intentionally does not raise; it's a no-op extension point for downstream algorithms.
    """
    try:
        # Attempt to call the packaged metric runner which expects the local context
        # (df_2d/df_3d, people_sequence, etc.), plus dest_dir and job_id if present.
        # We intentionally guard this call so metric failures don't break processing.
        try:
            from metric_algorithm import run_metrics_from_context
        except Exception:
            # metric package not available
            traceback.print_exc()
            return None

        # ctx is typically locals() from process_and_save; try to locate dest_dir and job_id
        dest_dir = ctx.get('dest_dir') or ctx.get('dest') or None
        job_id = ctx.get('job_id') or ctx.get('job') or None

        # Ensure strings
        if dest_dir is None or job_id is None:
            # If missing, don't fail - just call runner with placeholders
            try:
                res = run_metrics_from_context(ctx, dest_dir=str(dest_dir) if dest_dir is not None else '.', job_id=str(job_id) if job_id is not None else 'unknown', dimension=dimension)
            except Exception:
                traceback.print_exc()
                return None
        else:
            try:
                res = run_metrics_from_context(ctx, dest_dir=str(dest_dir), job_id=str(job_id), dimension=dimension)
            except Exception:
                traceback.print_exc()
                return None

        return res
    except Exception:
        traceback.print_exc()
        return None


def process_and_save(s3_key: str, dimension: str, job_id: str, turbo_without_skeleton: bool, dest_dir: Path):
    """Download input from S3 using s3_key, run OpenPose (or parse files) depending on dimension (2d/3d),
    interpolate sequence and save results to dest_dir/<job_id>.json.

    Expects an environment variable `S3_BUCKET` or `AWS_S3_BUCKET` to be set. This function is intended to run in a
    background worker (non-blocking for API).
    """
    try:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # determine input bucket from environment
        bucket = os.environ.get('S3_VIDEO_BUCKET_NAME') or os.environ.get('S3_BUCKET') or os.environ.get('AWS_S3_BUCKET')
        if not bucket:
            raise RuntimeError('S3_VIDEO_BUCKET_NAME (or S3_BUCKET/AWS_S3_BUCKET) environment variable is not set')

        s3 = boto3.client('s3')
        key = s3_key.lstrip('/')

        # Use /tmp for temporary work (Docker-friendly).
        # For debugging, create a persistent tmp dir (mkdtemp) so files are not auto-deleted.
        # NOTE: this is intentionally persistent for debugging; remember to clean up manually.
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

        if dimension == '2d':
            # download mp4 and run openpose on video
            local_video = tmp_dir / 'input.mp4'
            s3.download_file(bucket, key, str(local_video))
            run_openpose_on_video(str(local_video), str(output_json_dir), str(output_img_dir))

            # parse JSON outputs (sorted by filename) and build an in-memory pandas DataFrame (long/tidy)
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
        elif dimension == '3d':
                # download zip and extract expected color/ and depth/ folders
                local_zip = tmp_dir / 'input.zip'
                s3.download_file(bucket, key, str(local_zip))

                # Secure extraction (prevent zip-slip) and simple limits
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

                # Expect color/ and depth/ directories
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

                # Try to discover intrinsics recursively in the extracted tree so we can
                # use any recorded meta (e.g., depth_scale or color width/height)
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

                # Early validation: sample up to 10 depth files evenly to check dtype/shape and valid pixels
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

                    # Infer a depth scale factor (raw units -> meters). Heuristic:
                    # - if intrinsics file contained meta.depth_scale, prefer that
                    # - else if samples show integer dtype or median_valid > 1000, assume mm -> m
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

                # Run OpenPose on color images directory
                # Note: openpose.openpose expects an "output_img_path" whose dirname() is used
                # for the --write_images flag. To ensure images go into output_img_dir, pass
                # a file path under that directory (the dirname will then be output_img_dir).
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

                # Persist OpenPose rendered images into dest_dir BEFORE tmpdir is removed so operators
                # and metric modules can access them. This avoids images disappearing when TemporaryDirectory
                # context manager removes the tmpdir at the end of the with block.
                try:
                    dest_img_dir = Path(dest_dir) / 'img'
                    dest_img_dir.mkdir(parents=True, exist_ok=True)
                    if output_img_dir.exists():
                        for p in sorted(output_img_dir.iterdir()):
                            try:
                                if p.is_file():
                                    shutil.copy2(str(p), str(dest_img_dir / p.name))
                            except Exception:
                                pass
                except Exception:
                    pass

                # If no overlay mp4 yet, create a simple mp4 from copied images (written into dest_dir)
                try:
                    imgs = sorted([p for p in dest_img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]) if dest_img_dir.exists() else []
                    if imgs:
                        try:
                            import cv2
                            h, w = None, None
                            for p in imgs:
                                im = cv2.imread(str(p))
                                if im is None:
                                    continue
                                h, w = im.shape[:2]
                                break
                            if h is not None and w is not None:
                                out_mp4 = Path(dest_dir) / f"{job_id}_overlay.mp4"
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                vw = cv2.VideoWriter(str(out_mp4), fourcc, 30.0, (w, h))
                                for p in imgs:
                                    im = cv2.imread(str(p))
                                    if im is None:
                                        continue
                                    if im.shape[1] != w or im.shape[0] != h:
                                        im = cv2.resize(im, (w, h))
                                    vw.write(im)
                                vw.release()
                        except Exception:
                            # ignore if OpenCV not available or fails
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
            'pose_id': None,
            'people_sequence': people_sequence,
            'frame_count': len(people_sequence)
        }

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

        out_path = dest_dir / f"{job_id}.json"
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(response_payload, f, ensure_ascii=False, indent=2)

        # Copy rendered images into the persistent dest_dir so metric modules can access them
        try:
            # output_img_dir contains OpenPose rendered images inside tmpdir; copy a reasonable sample set
            dest_img_dir = Path(dest_dir) / 'img'
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            # copy images (preserve names)
            for p in sorted(output_img_dir.iterdir() if output_img_dir.exists() else []):
                try:
                    if p.is_file():
                        shutil.copy2(str(p), str(dest_img_dir / p.name))
                except Exception:
                    pass
        except Exception:
            pass

        # Optionally run local metrics using the in-memory DataFrames; run_metrics_in_process will write CSVs into dest_dir
        try:
            run_metrics_in_process(dimension, locals())
        except Exception:
            # metrics failures shouldn't take down processing
            traceback.print_exc()

        # If metrics did not create an overlay mp4, create a minimal overlay from copied images
        try:
            # look for any mp4 in dest_dir
            mp4s = list(Path(dest_dir).glob(f"{job_id}*.mp4"))
            if not mp4s:
                img_dir_check = Path(dest_dir) / 'img'
                imgs = sorted([p for p in img_dir_check.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]) if img_dir_check.exists() else []
                if imgs:
                    # create a simple mp4 using OpenCV
                    try:
                        import cv2
                        h, w = None, None
                        for p in imgs:
                            im = cv2.imread(str(p))
                            if im is None:
                                continue
                            h, w = im.shape[:2]
                            break
                        if h is not None and w is not None:
                            out_mp4 = Path(dest_dir) / f"{job_id}_overlay.mp4"
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            vw = cv2.VideoWriter(str(out_mp4), fourcc, 30.0, (w, h))
                            for p in imgs:
                                im = cv2.imread(str(p))
                                if im is None:
                                    continue
                                if im.shape[1] != w or im.shape[0] != h:
                                    im = cv2.resize(im, (w, h))
                                vw.write(im)
                            vw.release()
                    except Exception:
                        # ignore if OpenCV not available
                        pass
        except Exception:
            pass

        return response_payload

        # cleanup tmp if desired (do not remove in case debugging needed)
        # shutil.rmtree(tmp_dir)

        return response_payload
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
    """Upload the generated <job_id>.json and any overlay videos to the result S3 bucket.

    If `s3_key` is provided and has the form '<user_id>/<dimension>/.../<job_id_or_filename>',
    we upload to the same prefix '<user_id>/<dimension>/' with filenames like '<job_id>.json' and
    overlay files named '<job_id><overlay_suffix>.mp4'. If `s3_key` is not provided we fall back
    to key 'results/<job_id>.json'.
    """
    try:
        # determine result bucket from environment
        bucket = result_bucket or os.environ.get('S3_RESULT_BUCKET_NAME') or os.environ.get('RESULT_S3_BUCKET')
        if not bucket:
            raise RuntimeError('S3_RESULT_BUCKET_NAME (or RESULT_S3_BUCKET) not configured')

        s3 = boto3.client('s3')

        # Determine S3 prefix from s3_key when possible
        prefix = None
        if s3_key:
            try:
                k = s3_key.lstrip('/')
                parts = k.split('/')
                if len(parts) >= 2:
                    # user_id/dimension prefix
                    prefix = f"{parts[0]}/{parts[1]}"
            except Exception:
                prefix = None

        local_json = Path(dest_dir) / f"{job_id}.json"
        if not local_json.exists():
            raise FileNotFoundError(f'Result file not found: {local_json}')

        uploaded = []
        # upload main json
        if prefix:
            json_key = f"{prefix}/{job_id}.json"
        else:
            json_key = f"results/{job_id}.json"
        s3.upload_file(str(local_json), bucket, json_key)
        uploaded.append({'local': str(local_json), 'bucket': bucket, 'key': json_key})

        # upload any overlay mp4 files found in dest_dir
        # overlay filenames convention: either '{job_id}*.mp4' or '*.mp4' in dest_dir
        mp4_candidates = list(Path(dest_dir).glob(f"{job_id}*.mp4"))
        if not mp4_candidates:
            # fallback: all mp4s in dest_dir
            mp4_candidates = list(Path(dest_dir).glob("*.mp4"))

        for mp in mp4_candidates:
            fname = mp.name
            if prefix:
                key = f"{prefix}/{fname}"
            else:
                key = f"results/{fname}"
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
