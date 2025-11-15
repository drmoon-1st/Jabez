#!/usr/bin/env python3
"""
remove_short_mp4s.py

Recursively finds 'video' directories under a root and removes any .mp4 files
whose duration is less than a given threshold (seconds).

Defaults to dry-run (report only). Use --apply to actually delete files.

Usage:
  python remove_short_mp4s.py --root "E:\\golfDataset\\dataset\\train" --threshold 2.0 --apply

Dependencies: ffprobe must be on PATH or provide --ffprobe-path.
"""

from pathlib import Path
import argparse
import subprocess
import logging
import sys


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s %(levelname)-5s %(message)s'
    logging.basicConfig(level=level, format=fmt)


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


def list_mp4_files(video_dir: Path):
    return sorted([p for p in video_dir.rglob('*.mp4') if p.is_file()])


def ffprobe_duration(path: Path, ffprobe_cmd='ffprobe') -> float:
    cmd = [ffprobe_cmd, '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=8)
        s = out.decode().strip()
        if not s:
            return 0.0
        return float(s)
    except Exception as e:
        logging.debug(f"ffprobe failed for {path}: {e}")
        return 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Remove MP4 files shorter than threshold seconds')
    parser.add_argument('--root', default='E:\\golfDataset\\dataset\\train', help='Root folder to search')
    parser.add_argument('--only-class', dest='only_class', choices=['true', 'false'],
                        help='If set, search only under <root>/<only-class> (e.g. train\\true)')
    parser.add_argument('--threshold', type=float, default=1.5, help='Duration threshold in seconds (default 2.0)')
    parser.add_argument('--ffprobe-path', default='ffprobe', help='ffprobe executable path')
    parser.add_argument('--apply', action='store_true', help='Actually delete files (default is dry-run)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    root = Path(args.root)
    # If user requested only a specific class subfolder, validate and adjust root
    if args.only_class:
        candidate = root / args.only_class
        if not candidate.exists():
            logging.error(f"Requested class subfolder does not exist: {candidate}")
            sys.exit(2)
        logging.info(f"Restricting search to: {candidate}")
        root = candidate

    # If root looks like a train folder and has a 'true' subfolder, suggest using it
    else:
        try:
            train_true = root / 'true'
            if train_true.exists() and train_true.is_dir():
                logging.info(f"Note: found '{train_true}'. To restrict to true-only, run with --only-class true or set --root to '{train_true}'")
        except Exception:
            pass
    if not root.exists():
        logging.error(f"Root path does not exist: {root}")
        sys.exit(2)

    to_delete = []

    for vdir in find_video_dirs(root):
        logging.info(f"Scanning video dir: {vdir}")
        files = list_mp4_files(vdir)
        for p in files:
            dur = ffprobe_duration(p, ffprobe_cmd=args.ffprobe_path)
            # if ffprobe failed to get duration, note and skip
            if dur <= 0:
                if args.verbose:
                    logging.info(f"  {p.name}: could not determine duration (skipping)")
                else:
                    logging.debug(f"Could not determine duration for {p}, skipping")
                continue

            # show per-file duration when verbose for visibility
            if args.verbose:
                logging.info(f"  {p.name}: {dur:.2f}s")

            if dur < args.threshold:
                to_delete.append((p, dur))

    logging.info(f"Found {len(to_delete)} files shorter than {args.threshold}s")

    for p, dur in to_delete:
        if args.apply:
            try:
                p.unlink()
                logging.info(f"Deleted {p} ({dur:.2f}s)")
            except Exception as e:
                logging.error(f"Failed to delete {p}: {e}")
        else:
            logging.info(f"DRY RUN: would delete {p} ({dur:.2f}s)")


if __name__ == '__main__':
    main()
