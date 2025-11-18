"""
video_crop.py

Detect people in videos using a YOLO model and produce per-person cropped mp4 files.

Behavior:
- For each input mp4 under a dataset tree (see find_all_mp4s), run person detection on sampled frames.
- Cluster detections across frames to find distinct persons (simple spatial clustering by centroid).
- For each detected person track, compute union box, apply padding (10-20% default), clamp to video size.
- Crop the original video using ffmpeg and write to a sibling `crop_video` directory with filenames
  suffixed by `_person01_crop.mp4`, `_person02_crop.mp4`, etc.

Dependencies (install if needed):
- pip install opencv-python numpy tqdm ultralytics
  - `ultralytics` is recommended (YOLOv8) and is used if available. If not available, the script
    will try to import a local/yolov5-style model if present on PATH.

This script focuses on robust, easy-to-run cropping and intentionally samples frames for speed.
"""

from pathlib import Path
import os
import subprocess
import math
import json
from typing import List, Tuple
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
from tqdm import tqdm
import sqlite3
from datetime import datetime

# Padding ratio applied to union box (10% default). Can be overridden by CLI.
DEFAULT_PAD_RATIO = 0.1
# Separate defaults for horizontal (x) and vertical (y) padding. By default both use DEFAULT_PAD_RATIO.
DEFAULT_PAD_X = DEFAULT_PAD_Y = DEFAULT_PAD_RATIO

# Minimum crop dimensions (pixels). Clusters producing smaller crops will be skipped by default.
DEFAULT_MIN_WIDTH = 80
DEFAULT_MIN_HEIGHT = 80



def load_yolo_model():
    """Try to load an object-detection model (prefer ultralytics/yolov8). Returns a callable detect(img)
    that returns a list of detections in xyxy, conf, class format.
    """
    # default wrapper; device will be provided per-call via outer closure
    try:
        from ultralytics import YOLO

        def _make_detector(device: str = 'cpu'):
            model = YOLO('yolov8n.pt')  # small model; will auto-download if not present

            def detect(img):
                # img is a BGR numpy array
                results = model.predict(img, imgsz=640, device=device, verbose=False)
                out = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue
                    for b in boxes:
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        conf = float(b.conf[0].cpu().numpy())
                        cls = int(b.cls[0].cpu().numpy())
                        out.append((xyxy, conf, cls))
                return out

            return detect

        return _make_detector
    except Exception:
        # Fallback: try to import a yolov5 style detect function if repo available
        try:
            # This is intentionally best-effort; user should install ultralytics for reliability.
            from yolov5 import detect as y5_detect  # type: ignore

            def _make_detector(device: str = 'cpu'):
                def detect(img):
                    raise RuntimeError('yolov5 wrapper not implemented. Install `ultralytics` instead.')
                return detect

            return _make_detector
        except Exception:
            raise RuntimeError(
                'No YOLO model available. Install ultralytics with `pip install ultralytics`')


def sample_uniform_indices(frame_count: int, target_samples: int) -> List[int]:
    """Return up to target_samples indices uniformly spaced across the frame range.

    Guarantees inclusion of first (0) and last (frame_count-1) frames when possible.
    """
    if frame_count <= 0 or target_samples <= 0:
        return []
    import numpy as _np
    n = min(frame_count, int(target_samples))
    idxs = _np.linspace(0, frame_count - 1, num=n, dtype=int)
    return _np.unique(idxs).tolist()


def detect_people_in_video(video_path: Path, detect_fn, sample_fraction: float = 0.25) -> List[List[Tuple[float,float,float,float]]]:
    """Run detection on sampled frames and return list per frame of person boxes (xyxy).

    Two sampling modes supported:
    - sample_fraction: if provided (0<sample_fraction<=1) will sample approx frame_count * sample_fraction frames
      uniformly across the video (recommended default behavior).
    - sample_rate: legacy behaviour (every Nth frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = max(1, int(frame_count * sample_fraction)) if sample_fraction is not None else max(1, int(frame_count * 0.25))
    indices = sample_uniform_indices(frame_count, target)
    boxes_by_frame = []
    for idx in tqdm(indices, desc=f"Detect {video_path.name}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            boxes_by_frame.append([])
            continue
        dets = detect_fn(frame)
        ppl = []
        for (xyxy, conf, cls) in dets:
            # class 0 is usually person for COCO models
            if cls != 0:
                continue
            x1, y1, x2, y2 = map(float, xyxy)
            ppl.append((x1, y1, x2, y2))
        boxes_by_frame.append(ppl)
    cap.release()
    return boxes_by_frame


def cluster_boxes_across_frames(boxes_by_frame: List[List[Tuple[float,float,float,float]]], eps=100, min_samples=3):
    """Cluster detections across frames by centroid using DBSCAN to separate distinct people.
    Returns list of clusters, where each cluster is a list of boxes (x1,y1,x2,y2).
    """
    from sklearn.cluster import DBSCAN

    centroids = []
    mapping = []  # maps centroid index -> (frame_idx, box_idx)
    for fi, boxes in enumerate(boxes_by_frame):
        for bi, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centroids.append([cx, cy])
            mapping.append((fi, bi))

    if not centroids:
        return []
    X = np.array(centroids)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    clusters = {}
    for i, lbl in enumerate(labels):
        if lbl == -1:
            continue
        clusters.setdefault(lbl, []).append(mapping[i])

    # For each cluster, collect all corresponding boxes
    cluster_boxes = []
    for lbl, entries in clusters.items():
        boxes = []
        for (fi, bi) in entries:
            boxes.append(boxes_by_frame[fi][bi])
        cluster_boxes.append(boxes)
    return cluster_boxes


def union_box(boxes: List[Tuple[float,float,float,float]], pad_x=DEFAULT_PAD_X, pad_y=DEFAULT_PAD_Y) -> Tuple[int,int,int,int]:
    arr = np.array(boxes)
    x1 = float(arr[:, 0].min())
    y1 = float(arr[:, 1].min())
    x2 = float(arr[:, 2].max())
    y2 = float(arr[:, 3].max())
    w = x2 - x1
    h = y2 - y1
    pad_w = w * pad_x
    pad_h = h * pad_y
    return int(x1 - pad_w), int(y1 - pad_h), int(w + 2 * pad_w), int(h + 2 * pad_h)


def clamp_box_to_frame(bbox: Tuple[int,int,int,int], frame_w: int, frame_h: int) -> Tuple[int,int,int,int]:
    x, y, w, h = bbox
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > frame_w: w = frame_w - x
    if y + h > frame_h: h = frame_h - y
    if w <= 0 or h <= 0:
        raise ValueError(f'Invalid crop box after clamp: {(x,y,w,h)}')
    return x, y, w, h


def crop_with_ffmpeg(src: Path, dst: Path, bbox: Tuple[int,int,int,int]):
    x, y, w, h = bbox
    cmd = [
        'ffmpeg', '-y', '-i', str(src),
        '-filter:v', f'crop={w}:{h}:{x}:{y}',
        '-pix_fmt', 'yuv420p', str(dst)
    ]
    subprocess.run(cmd, check=True)


def _open_index_db(db_path: Path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    cur = conn.cursor()
    # WAL mode helps concurrency
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("CREATE TABLE IF NOT EXISTS videos (video_path TEXT PRIMARY KEY, status TEXT, outputs TEXT, ts TEXT, error TEXT);")
    return conn


def _is_video_done(conn: sqlite3.Connection, video_path: str) -> bool:
    cur = conn.cursor()
    cur.execute('SELECT status FROM videos WHERE video_path = ?', (video_path,))
    row = cur.fetchone()
    return bool(row and row[0] == 'done')


def _mark_video_done(conn: sqlite3.Connection, video_path: str, outputs: List[str]):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute('INSERT OR REPLACE INTO videos (video_path, status, outputs, ts, error) VALUES (?, ?, ?, ?, ?)', (video_path, 'done', json.dumps(outputs), ts, None))


def _mark_video_failed(conn: sqlite3.Connection, video_path: str, error_msg: str = None):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute('INSERT OR REPLACE INTO videos (video_path, status, outputs, ts, error) VALUES (?, ?, ?, ?, ?)', (video_path, 'failed', json.dumps([]), ts, error_msg))


def process_one_mp4(mp4_path: Path, pad_x=DEFAULT_PAD_X, pad_y=DEFAULT_PAD_Y, sample_fraction: float = 0.25, eps=100, min_samples=3, min_width=DEFAULT_MIN_WIDTH, min_height=DEFAULT_MIN_HEIGHT, resume: bool = True, index_db: str = None, device: str = 'cpu'):
    print(f'Processing {mp4_path}')
    # determine crop output directory early
    video_dir = None
    for anc in mp4_path.parents:
        if anc.name.lower() == 'video':
            video_dir = anc
            break
    if video_dir is not None:
        crop_video_dir = video_dir.parent / 'video_crop'
    else:
        crop_video_dir = mp4_path.parent.parent / 'video_crop'
    base_name = mp4_path.stem
    crop_video_dir.mkdir(parents=True, exist_ok=True)

    conn = None
    # open index DB early (each worker/process should open its own connection)
    if index_db:
        conn = _open_index_db(Path(index_db))
        if resume and _is_video_done(conn, str(mp4_path)):
            print(f'Skipping {mp4_path.name} because index shows done')
            # return list of existing crop files for this video
            existing = sorted(crop_video_dir.glob(f'{base_name}_person*_crop.mp4'))
            try:
                conn.close()
            except Exception:
                pass
            return existing

    try:
        # load detector factory and create detector bound to device
        detector_factory = load_yolo_model()
        if callable(detector_factory):
            detect_fn = detector_factory(device)
        else:
            # legacy: load_yolo_model returned a detect function directly
            detect_fn = detector_factory

        boxes_by_frame = detect_people_in_video(mp4_path, detect_fn, sample_fraction=sample_fraction)
        clusters = cluster_boxes_across_frames(boxes_by_frame, eps=eps, min_samples=min_samples)
        if not clusters:
            print(f'No people detected in {mp4_path.name}')
            return []

        # get video size
        cap = cv2.VideoCapture(str(mp4_path))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        out_files = []
        # crop_video_dir and base_name were computed earlier (for resume support)
        for i, boxes in enumerate(clusters, start=1):
            try:
                bbox = union_box(boxes, pad_x=pad_x, pad_y=pad_y)
                bbox = clamp_box_to_frame(bbox, frame_w, frame_h)
                # Ensure width/height are integers
                x, y, w, h = bbox
                w = int(w); h = int(h); x = int(x); y = int(y)
                # User requested: ignore (skip) crops that are <= min thresholds
                if w <= min_width or h <= min_height:
                    print(f'Skipping cluster {i} for {mp4_path.name}: crop too small {(w,h)} <= ({min_width},{min_height})')
                    continue
                # make w/h even (libx264 prefers even dims)
                if w % 2 == 1:
                    if x + w + 1 <= frame_w:
                        w += 1
                    else:
                        x = max(0, x - 1); w += 1
                if h % 2 == 1:
                    if y + h + 1 <= frame_h:
                        h += 1
                    else:
                        y = max(0, y - 1); h += 1
                bbox = (x, y, w, h)
            except Exception as e:
                print(f'Skipping cluster {i} for {mp4_path.name}: {e}')
                continue
            suffix = f'_person{i:02d}_crop.mp4'
            out_path = crop_video_dir / f'{base_name}{suffix}'
            crop_with_ffmpeg(mp4_path, out_path, bbox)
            out_files.append(out_path)
    except Exception as e:
        # mark failed in index DB if available
        if conn is not None:
            try:
                _mark_video_failed(conn, str(mp4_path), str(e))
            except Exception:
                pass
        raise
    finally:
        # On success, mark done and close connection; on failure the except above already marked failed
        if conn is not None:
            try:
                _mark_video_done(conn, str(mp4_path), [str(p) for p in out_files])
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    return out_files
    


def find_all_mp4s(base_path: Path):
    """Find mp4 files under a dataset root.

    Behavior:
    - If base_path contains 'video' directories in arbitrary places, this will find them recursively
      and return all .mp4 files (used when no specific class is provided).
    - For compatibility with older scripts, callers can still pass a specific target directory
      (e.g., root/true/class or root/class) and this will find mp4s beneath it.
    """
    mp4s = []
    # If base_path itself is the intended class root, check for video dirs beneath it
    if base_path.exists():
        for vdir in base_path.rglob('video'):
            try:
                if not vdir.is_dir():
                    continue
            except Exception:
                continue
            for p in vdir.rglob('*.mp4'):
                mp4s.append(p)
    return mp4s


def main(dataset_base_path: str, pad_ratio=DEFAULT_PAD_RATIO, sample_rate=5, max_videos=None):
    base = Path(dataset_base_path)
    mp4s = find_all_mp4s(base)
    if max_videos:
        mp4s = mp4s[:max_videos]
    print(f'Found {len(mp4s)} mp4s')
    for mp4 in mp4s:
        try:
            process_one_mp4(mp4, pad_ratio=pad_ratio, sample_rate=sample_rate)
        except Exception as e:
            print(f'Error processing {mp4}: {e}')


if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=r'E:\golfDataset_dlc\dataset\train', help='Top-level dataset root (matches mp4_meta_processing scripts)')
    parser.add_argument('--class', dest='cls', choices=['good','best','normal','worst','bad','test'], default=None, help='Optional class to process. One of: good,best,normal,worst,bad. good/best/normal -> <root>/true/<class>, worst/bad -> <root>/false/<class>')
    parser.add_argument('--pad_ratio', type=float, default=DEFAULT_PAD_RATIO, help='(legacy) uniform padding ratio for both axes')
    parser.add_argument('--pad_x', type=float, default=DEFAULT_PAD_X, help='Horizontal padding ratio (relative to width)')
    parser.add_argument('--pad_y', type=float, default=DEFAULT_PAD_Y, help='Vertical padding ratio (relative to height)')
    parser.add_argument('--sample_fraction', type=float, default=0.25, help='(recommended) fraction of frames to sample uniformly (0..1). Default 0.25')
    parser.add_argument('--max_videos', type=int, default=None)
    parser.add_argument('--index-db', type=str, default=r'E:\data_processing\openpose_processing\processed.db', help='Path to SQLite index DB file to record processing state (default under openpose_processing)')
    parser.add_argument('--batch_size', type=int, default=50, help='Process videos in batches of this size to limit memory (default 50)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel worker processes to use per batch (default 1). Uses ProcessPoolExecutor when >1')
    parser.add_argument('--device', type=str, default='auto', help="Device to run detector on: 'auto'|'cpu'|'cuda:0' etc. Default 'auto' (prefers GPU if available)")
    parser.add_argument('--min_width', type=int, default=DEFAULT_MIN_WIDTH, help='Minimum crop width in pixels (default 160). Crops smaller than this will be expanded or skipped.')
    parser.add_argument('--min_height', type=int, default=DEFAULT_MIN_HEIGHT, help='Minimum crop height in pixels (default 160). Crops smaller than this will be expanded or skipped.')
    parser.add_argument('--no-resume', dest='no_resume', action='store_true', help='Disable resume behavior; reprocess all videos even if .done marker exists')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Resolve device: if auto, prefer CUDA when available
    if args.device == 'auto':
        try:
            import torch
            args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        except Exception:
            args.device = 'cpu'

    # simple logging control
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)-5s %(message)s')

    root = Path(args.root)
    if not root.exists():
        logging.error(f'Root does not exist: {root}')
        raise SystemExit(2)

    target_roots = []
    if args.cls:
        # support layouts: <root>/<class> or under boolean folders
        cand_direct = root / args.cls
        cand_under_true = root / 'true' / args.cls
        cand_under_false = root / 'false' / args.cls
        # Map class names to boolean subtrees
        true_classes = {'good', 'best', 'normal','test'}
        false_classes = {'worst', 'bad'}

        if args.cls in false_classes:
            # prefer root/false/<class>
            if cand_under_false.exists() and cand_under_false.is_dir():
                target_roots.append(cand_under_false)
                logging.info(f'Processing class folder: {cand_under_false} (under false)')
            elif cand_direct.exists() and cand_direct.is_dir():
                target_roots.append(cand_direct)
                logging.info(f'Processing class folder: {cand_direct} (direct)')
            else:
                logging.error(f'Requested class folder not found. Tried: {cand_under_false} and {cand_direct}')
                raise SystemExit(2)
        else:
            # default behavior: prefer root/true/<class> for common positive classes, else fall back to direct
            if cand_under_true.exists() and cand_under_true.is_dir():
                target_roots.append(cand_under_true)
                logging.info(f'Processing class folder: {cand_under_true} (under true)')
            elif cand_direct.exists() and cand_direct.is_dir():
                target_roots.append(cand_direct)
                logging.info(f'Processing class folder: {cand_direct} (direct)')
            else:
                logging.error(f'Requested class folder not found. Tried: {cand_under_true} and {cand_direct}')
                raise SystemExit(2)
    else:
        # No class provided: process entire root by finding all video dirs beneath it
        target_roots.append(root)

    # collect mp4s from all target_roots
    all_mp4s = []
    for tr in target_roots:
        all_mp4s.extend(find_all_mp4s(tr))

    print(f'Found {len(all_mp4s)} mp4s')
    # Backwards-compat: if user passed --pad_ratio but not --pad_x/--pad_y, apply pad_ratio to both axes
    import sys
    if ('--pad_ratio' in sys.argv) and ('--pad_x' not in sys.argv) and ('--pad_y' not in sys.argv):
        args.pad_x = args.pad_y = args.pad_ratio

    # default behaviour: uniform sampling fraction 1/4
    videos = all_mp4s[:args.max_videos] if args.max_videos else all_mp4s
    # process in batches to limit memory/handles
    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i+size]

    # If class is 'test' we don't write to the index DB (debug mode)
    index_db_to_use = None if args.cls == 'test' else args.index_db

    for batch_idx, batch in enumerate(chunked(videos, args.batch_size), start=1):
        logging.info(f'Processing batch {batch_idx}: {len(batch)} videos')
        if args.workers and args.workers > 1:
            # Use process pool so each worker loads its own model (safer for large models)
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(process_one_mp4, mp4, args.pad_x, args.pad_y, args.sample_fraction, 100, 3, args.min_width, args.min_height, not args.no_resume, index_db_to_use, args.device): mp4 for mp4 in batch}
                for fut in as_completed(futures):
                    mp4 = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        logging.error(f'Error processing {mp4} in batch {batch_idx}: {e}')
        else:
            for mp4 in batch:
                try:
                    process_one_mp4(mp4, pad_x=args.pad_x, pad_y=args.pad_y, sample_fraction=args.sample_fraction, eps=100, min_samples=3, min_width=args.min_width, min_height=args.min_height, resume=not args.no_resume, index_db=index_db_to_use, device=args.device)
                except Exception as e:
                    logging.error(f'Error processing {mp4} in batch {batch_idx}: {e}')

        # free memory between batches
        gc.collect()
