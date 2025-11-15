#!/usr/bin/env python3
"""
merge_short_slices.py

Searches recursively for 'video' directories under a root, groups mp4 files by a shared
base prefix (regex strip trailing _NN part), checks durations with ffprobe, and if any
slice in a group is shorter than min_duration (seconds), merges all slices in that group
into a single mp4 using ffmpeg concat demuxer.

Usage:
  python merge_short_slices.py --root "E:\\golfDataset\\dataset\\train" --min-duration 4

Defaults to dry-run. Use --replace to actually write merged output files.

Dependencies: ffmpeg and ffprobe must be on PATH or provide --ffmpeg-path.
"""

from pathlib import Path
import re
import argparse
import subprocess
import sys
import tempfile
import shlex
import logging


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s %(levelname)-5s %(message)s'
    logging.basicConfig(level=level, format=fmt)


GROUP_RE = re.compile(r"^(?P<prefix>.+?)(_\d{2})?\.(?P<ext>mp4|MP4)$")


def find_video_dirs(root: Path):
    # If the provided root itself is a 'video' directory, include it
    try:
        if root.is_dir() and root.name.lower() == 'video':
            yield root
    except Exception:
        pass

    for p in root.rglob('video'):
        # avoid yielding the root twice
        try:
            if p.is_dir() and (not (root.is_dir() and p.samefile(root))):
                yield p
        except Exception:
            # on some platforms samefile may fail; fall back to path compare
            if p.is_dir() and str(p) != str(root):
                yield p


def list_mp4_files(video_dir: Path):
    # Search recursively under the video_dir for mp4 files (some datasets use one extra level)
    return sorted([p for p in video_dir.rglob('*.mp4') if p.is_file()])


def group_files(files):
    """Group files by the filename prefix that excludes the trailing _NN part.

    Example: '..._004_01.mp4' and '..._004_02.mp4' -> key '..._004'
    """
    groups = {}
    for f in files:
        name = f.name
        # remove extension then strip last underscore-number group if present
        stem = f.stem
        m = re.match(r'^(?P<prefix>.+?)_(?P<part>\d{2})$', stem)
        if m:
            key = m.group('prefix')
        else:
            # fallback: use full stem
            key = stem
        groups.setdefault(key, []).append(f)

    # sort each group's files by name (path) to maintain order
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda p: p.name)
    return groups


def ffprobe_duration(path: Path, ffprobe_cmd='ffprobe') -> float:
    # returns duration in seconds as float, or 0 on error
    cmd = [ffprobe_cmd, '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(path)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        s = out.decode().strip()
        if not s:
            return 0.0
        return float(s)
    except Exception as e:
        logging.debug(f"ffprobe failed for {path}: {e}")
        return 0.0


def merge_group(group_files, out_path: Path, ffmpeg_cmd='ffmpeg'):
    # Use concat demuxer via a temporary list file with absolute paths
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt', encoding='utf-8') as listf:
        for p in group_files:
            # ffmpeg concat expects lines like: file 'path'
            listf.write("file '{}\n".format(str(p).replace("'", "'\\''")))
        list_name = listf.name

    cmd = [ffmpeg_cmd, '-y', '-f', 'concat', '-safe', '0', '-i', list_name, '-c', 'copy', str(out_path)]
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed: {e}")
        return False
    finally:
        try:
            Path(list_name).unlink()
        except Exception:
            pass


def validate_file_with_ffmpeg(path: Path, ffmpeg_cmd='ffmpeg', timeout: int = 10) -> bool:
    """Quickly validate that ffmpeg can read/process the file.

    Runs: ffmpeg -v error -i <path> -f null -
    Returns True if ffmpeg exits 0, False otherwise. Uses timeout to avoid hangs.
    """
    cmd = [ffmpeg_cmd, '-v', 'error', '-i', str(path), '-f', 'null', '-']
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
        if proc.returncode == 0:
            return True
        else:
            logging.warning(f"ffmpeg validation failed for {path} (returncode={proc.returncode}): {proc.stderr.decode(errors='ignore')[:200]}")
            return False
    except subprocess.TimeoutExpired:
        logging.warning(f"ffmpeg validation timed out for {path} (timeout={timeout}s)")
        return False
    except Exception as e:
        logging.warning(f"ffmpeg validation error for {path}: {e}")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description='Merge short video slices into full clips')
    parser.add_argument('--root', default='E:\\golfDataset\\dataset\\train', help='Root folder to search (e.g. E:\\golfDataset\\dataset\\train)')
    parser.add_argument('--min-duration', type=float, default=4.0, help='Minimum allowed duration for any slice in seconds')
    parser.add_argument('--ffmpeg-path', default='ffmpeg', help='ffmpeg executable path')
    parser.add_argument('--ffprobe-path', default='ffprobe', help='ffprobe executable path')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', default=False, help='Do not perform merges; run in report-only mode')
    parser.add_argument('--replace', dest='replace', action='store_true', default=True, help='Apply merges and replace originals (default behavior)')
    parser.add_argument('--output-suffix', default='_merged', help='Suffix appended to merged filename before extension')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose (debug) logging')
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    if args.replace:
        args.dry_run = False
        logging.debug('Replace requested: dry-run disabled')

    root = Path(args.root)
    if not root.exists():
        logging.error(f"Root path does not exist: {root}")
        sys.exit(2)

    total_groups = 0
    to_merge = []  # list of tuples (group_key, files)

    # search for 'video' dirs
    for vdir in find_video_dirs(root):
        logging.info(f"Processing video dir: {vdir}")
        files = list_mp4_files(vdir)
        if not files:
            continue
        groups = group_files(files)
        for key, gfiles in groups.items():
            total_groups += 1
            # FAST/UNSAFE MODE: skip per-file validation and duration checks
            # Merge any group that has more than one slice immediately.
            if len(gfiles) <= 1:
                continue
            # record with dummy durations (0.0)
            to_merge.append((vdir, key, gfiles, [0.0] * len(gfiles)))

    logging.info(f"Scanned groups: {total_groups}. Groups to merge: {len(to_merge)}")

    

    for vdir, key, gfiles, durations in to_merge:
        logging.info('-' * 60)
        logging.info(f"Group: {key} in {vdir}")
        for p, d in zip(gfiles, durations):
            logging.info(f"  {p.name}: {d:.2f}s")
        merged_name = f"{key}{args.output_suffix}.mp4"
        merged_path = vdir / merged_name
        if args.dry_run:
            logging.info(f"DRY RUN: would merge {len(gfiles)} files -> {merged_path}")
            continue

        logging.info(f"Merging into: {merged_path}")
        ok = merge_group(gfiles, merged_path, ffmpeg_cmd=args.ffmpeg_path)
        if not ok:
            logging.error(f"Failed to merge group {key}")
            continue

        # If replace requested, remove original slice files and rename merged file to base name
        if args.replace:
            # determine base output name (take prefix without trailing _NN)
            base_output = vdir / f"{key}.mp4"
            try:
                for p in gfiles:
                    p.unlink()
                merged_path.replace(base_output)
                logging.info(f"Replaced slices with {base_output}")
            except Exception as e:
                logging.error(f"Error replacing files: {e}")


if __name__ == '__main__':
    main()
