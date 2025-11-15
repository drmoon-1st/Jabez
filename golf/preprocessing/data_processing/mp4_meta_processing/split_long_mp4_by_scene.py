#!/usr/bin/env python3
"""
split_long_mp4_by_scene.py

Scans 'video' directories under <root>/<class> (class is 'good' or 'best' and must be
specified) and for every mp4 longer than --min-length (default 10s) detects abrupt scene
changes using ffmpeg's scene detection. The script splits the input into consecutive
segments at scene-change timestamps, then removes any segments shorter than --min-segment
(default 1.5s) by merging them into neighbors.

Outputs: by default writes segments next to the original file with suffix _partXX.mp4.
Dry-run mode reports actions only.

Dependencies: ffmpeg and ffprobe on PATH or provide --ffmpeg-path / --ffprobe-path.
"""

from pathlib import Path
import argparse
import subprocess
import logging
import sys
import re
import math
import tempfile


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s %(levelname)-5s %(message)s'
    logging.basicConfig(level=level, format=fmt)


def ffprobe_duration(path: Path, ffprobe_cmd='ffprobe') -> float:
    cmd = [ffprobe_cmd, '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        s = out.decode().strip()
        if not s:
            return 0.0
        return float(s)
    except Exception as e:
        logging.warning(f"ffprobe failed for {path}: {e}")
        return 0.0


def ffprobe_fps(path: Path, ffprobe_cmd='ffprobe') -> float:
    """Return video FPS (avg_frame_rate) or 0.0 on error."""
    cmd = [ffprobe_cmd, '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=avg_frame_rate',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        s = out.decode().strip()
        if not s:
            return 0.0
        if '/' in s:
            num, den = s.split('/')
            num = float(num)
            den = float(den) if den and float(den) != 0 else 0.0
            return (num / den) if den else 0.0
        return float(s)
    except Exception as e:
        logging.warning(f"ffprobe (fps) failed for {path}: {e}")
        return 0.0


def detect_scene_changes(path: Path, ffmpeg_cmd='ffmpeg', scene_thresh=0.4):
    """Run ffmpeg select=gt(scene,T),showinfo and parse pts_time values from stderr."""
    # Build filter expression. We wrap the select expression in single quotes; when passing
    # as a single argv element this works for ffmpeg.
    vf = f"select='gt(scene,{scene_thresh})',showinfo"
    cmd = [ffmpeg_cmd, '-hide_banner', '-loglevel', 'info', '-i', str(path), '-vf', vf, '-an', '-f', 'null', '-']
    logging.debug('Running ffmpeg for scene detection: %s', ' '.join(cmd))
    try:
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logging.warning(f"ffmpeg scene detection timed out for {path}")
        return []

    stderr = proc.stderr
    # Parse pts_time occurrences from showinfo lines
    times = []
    for line in stderr.splitlines():
        if 'pts_time:' in line:
            m = re.search(r'pts_time:([0-9]+\.?[0-9]*)', line)
            if m:
                try:
                    t = float(m.group(1))
                    times.append(t)
                except Exception:
                    pass
    times = sorted(set(times))
    logging.debug(f"Detected scene times: {times}")
    return times


def merge_short_segments(segments, min_seg):
    """Given list of (start,end) segments, merge segments shorter than min_seg into neighbors.
    Strategy: iterate and if a segment is shorter than min_seg, merge it to the next segment if exists,
    otherwise merge to previous.
    """
    if not segments:
        return []
    out = []
    i = 0
    while i < len(segments):
        s, e = segments[i]
        dur = e - s
        if dur >= min_seg:
            out.append((s,e))
            i += 1
            continue
        # too short -> try to merge with next
        if i + 1 < len(segments):
            ns, ne = segments[i+1]
            # merge current and next
            out.append((s, ne))
            i += 2
        else:
            # last segment, merge with previous if exists
            if out:
                ps, pe = out[-1]
                out[-1] = (ps, e)
            else:
                # single short segment: keep it (edge-case)
                out.append((s,e))
            i += 1
    # After merging, it's possible some segments still shorter than min_seg (rare); repeat until stable
    changed = True
    while changed:
        changed = False
        new_out = []
        j = 0
        while j < len(out):
            s,e = out[j]
            if (e - s) >= min_seg:
                new_out.append((s,e))
                j += 1
            else:
                changed = True
                if j + 1 < len(out):
                    ns, ne = out[j+1]
                    new_out.append((s, ne))
                    j += 2
                elif new_out:
                    ps, pe = new_out[-1]
                    new_out[-1] = (ps, e)
                    j += 1
                else:
                    new_out.append((s,e))
                    j += 1
        out = new_out
    return out


def write_segment(in_path: Path, start: float, end: float, out_path: Path, ffmpeg_cmd='ffmpeg') -> bool:
    # use re-encode for safe accurate cutting
    # Build ffmpeg command: -ss start -to end -i in -c:v libx264 -crf 20 -preset veryfast -c:a copy out
    duration = max(0.001, end - start)
    # Use -i then -ss -t for accurate re-encoding-based trimming
    cmd = [ffmpeg_cmd, '-hide_banner', '-loglevel', 'error', '-i', str(in_path), '-ss', f"{start}", '-t', f"{duration}",
           '-avoid_negative_ts', 'make_zero', '-fflags', '+genpts', '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-c:a', 'copy', str(out_path)]
    logging.debug('Writing segment: %s', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed to write segment {out_path}: {e}")
        return False


def verify_segment_duration(path: Path, expected: float, ffprobe_cmd='ffprobe') -> float:
    """Return actual duration (seconds) or 0.0 on error"""
    actual = ffprobe_duration(path, ffprobe_cmd=ffprobe_cmd)
    if actual <= 0:
        logging.warning(f"Could not verify duration for written segment {path}")
        return 0.0
    # allow small tolerance
    if abs(actual - expected) > max(0.1, expected * 0.02):
        logging.warning(f"Segment duration mismatch for {path}: expected {expected:.2f}s, got {actual:.2f}s")
    return actual


def find_video_dirs(root: Path):
    try:
        if root.is_dir() and root.name.lower() == 'video':
            yield root
    except Exception:
        pass
    for p in root.rglob('video'):
        try:
            if p.is_dir() and (not (root.is_dir() and p.samefile(root))):
                yield p
        except Exception:
            if p.is_dir() and str(p) != str(root):
                yield p


def main(argv=None):
    parser = argparse.ArgumentParser(description='Split long mp4s into swings using scene-change detection')
    parser.add_argument('--root', default='D:\\golfDataset\\dataset\\test', help='Top-level dataset root')
    parser.add_argument('--class', dest='cls', required=True, choices=['good','best', 'test'], help='Which class folder to process (good or best)')
    parser.add_argument('--min-length', type=float, default=10.0, help='Process files longer than this (seconds)')
    parser.add_argument('--min-segment', type=float, default=1.5, help='Minimum segment length to keep (seconds)')
    parser.add_argument('--scene-thresh', type=float, default=0.05, help='ffmpeg scene detection threshold (0..1)')
    parser.add_argument('--ffmpeg-path', default='ffmpeg', help='ffmpeg executable')
    parser.add_argument('--ffprobe-path', default='ffprobe', help='ffprobe executable')
    parser.add_argument('--dry-run', action='store_true', help='Report actions only')
    parser.add_argument('--backup-dir', default=None, help='If set, copy original mp4 to this folder (maintains relative path) before splitting')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--merge-if-shorter-than', default='60f',
                        help="If a produced segment is shorter than this threshold, merge it into the previous segment.\n"
                             "Specify seconds (e.g. 2.5) or frames with 'f' suffix (e.g. 60f). Default: 60f")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    root = Path(args.root)
    if not root.exists():
        logging.error(f"Root does not exist: {root}")
        sys.exit(2)

    # Support two possible layouts:
    # 1) <root>/<class>           (e.g. train/good)
    # 2) <root>/true/<class>      (e.g. train/true/good)
    candidate_direct = root / args.cls
    candidate_under_true = root / 'true' / args.cls
    if candidate_direct.exists() and candidate_direct.is_dir():
        target_root = candidate_direct
        logging.info(f"Processing class folder: {target_root} (direct)")
    elif candidate_under_true.exists() and candidate_under_true.is_dir():
        target_root = candidate_under_true
        logging.info(f"Processing class folder: {target_root} (under 'true')")
    else:
        logging.error(f"Requested class folder not found. Tried: {candidate_direct} and {candidate_under_true}")
        sys.exit(2)

    # find all mp4s recursively under video directories within target_root
    files = []
    for vdir in find_video_dirs(target_root):
        logging.info(f"Scanning video dir: {vdir}")
        for p in vdir.rglob('*.mp4'):
            files.append(p)

    logging.info(f"Found {len(files)} mp4 files under {target_root}")

    for p in files:
        dur = ffprobe_duration(p, ffprobe_cmd=args.ffprobe_path)
        if dur <= 0:
            logging.warning(f"Cannot determine duration for {p}, skipping")
            continue
        if dur <= args.min_length:
            logging.debug(f"Skipping {p} (duration {dur:.2f}s <= min-length)")
            continue
        logging.info(f"Analyzing {p} (duration {dur:.2f}s)")

        # detect scene change times
        scene_times = detect_scene_changes(p, ffmpeg_cmd=args.ffmpeg_path, scene_thresh=args.scene_thresh)
        # Build cut points: include 0 and end
        cut_points = [0.0] + scene_times + [dur]
        # construct initial segments
        segments = []
        for a,b in zip(cut_points[:-1], cut_points[1:]):
            # ensure numeric rounding
            s = max(0.0, float(a))
            e = min(float(dur), float(b))
            if e - s > 0.01:
                segments.append((s,e))
        logging.debug(f"Initial segments: {segments}")

        # merge segments shorter than min_segment
        final_segments = merge_short_segments(segments, args.min_segment)
        logging.info(f"Final segments for {p.name}: {final_segments}")

        # optionally merge very short segments into the previous one (to avoid tiny finish-only parts)
        # parse threshold: if value endswith 'f' interpret as frames and convert using file fps
        merge_threshold = None
        thr_raw = args.merge_if_shorter_than
        # get fps for this file (fallback 30)
        try:
            fps = ffprobe_fps(p, ffprobe_cmd=args.ffprobe_path)
            if fps <= 0:
                fps = 30.0
        except Exception:
            fps = 30.0
        try:
            if isinstance(thr_raw, str) and thr_raw.lower().endswith('f'):
                frames = int(thr_raw[:-1])
                merge_threshold = float(frames) / float(fps)
            else:
                merge_threshold = float(thr_raw)
        except Exception:
            logging.warning(f"Invalid --merge-if-shorter-than value '{thr_raw}', ignoring")
            merge_threshold = None

        if merge_threshold is not None and merge_threshold > 0:
            def merge_into_previous_if_short(segments, threshold):
                if not segments:
                    return []
                segs = [s for s in segments]
                out = []
                i = 0
                while i < len(segs):
                    s,e = segs[i]
                    dur_i = e - s
                    if dur_i <= threshold:
                        if out:
                            # merge into previous
                            ps,pe = out[-1]
                            out[-1] = (ps, e)
                        else:
                            # no previous -> merge into next if exists by extending next's start
                            if i + 1 < len(segs):
                                ns, ne = segs[i+1]
                                segs[i+1] = (s, ne)
                            else:
                                # single short segment, keep it
                                out.append((s,e))
                        i += 1
                    else:
                        out.append((s,e))
                        i += 1
                return out

            merged_prev = merge_into_previous_if_short(final_segments, merge_threshold)
            if merged_prev != final_segments:
                logging.info(f"Merged very short segments for {p.name}: from {final_segments} -> {merged_prev} (threshold {merge_threshold:.2f}s)")
                final_segments = merged_prev

        if args.dry_run:
            logging.info(f"DRY RUN: would write {len(final_segments)} segments for {p}")
            continue

        # create output files
        # if backup requested, copy original there before writing segments
        if args.backup_dir:
            try:
                backup_root = Path(args.backup_dir)
                rel = p.relative_to(target_root)
                dest = backup_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    import shutil
                    shutil.copy2(p, dest)
                    logging.info(f"Backed up original to {dest}")
                else:
                    logging.debug(f"Backup already exists: {dest}")
            except Exception as e:
                logging.error(f"Failed to backup original {p} to {args.backup_dir}: {e}")
        # end backup
        for idx, (s,e) in enumerate(final_segments, start=1):
            out_name = f"{p.stem}_part{idx:02d}.mp4"
            out_path = p.parent / out_name
            logging.info(f"Writing segment {idx}: {s:.2f}s -> {e:.2f}s -> {out_path}")
            ok = write_segment(p, s, e, out_path, ffmpeg_cmd=args.ffmpeg_path)
            if not ok:
                logging.error(f"Failed to write segment {out_path}")
                continue
            # verify duration
            expected = e - s
            actual = verify_segment_duration(out_path, expected, ffprobe_cmd=args.ffprobe_path)
            if actual <= 0 or actual < 0.5:
                logging.error(f"Suspicious segment written (too short or unreadable): {out_path} ({actual:.2f}s)")


if __name__ == '__main__':
    main()
