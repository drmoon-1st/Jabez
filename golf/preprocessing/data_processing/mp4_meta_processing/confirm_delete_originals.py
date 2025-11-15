#!/usr/bin/env python3
"""
confirm_delete_originals.py

Given a backup directory created by split_long_mp4_by_scene.py (the --backup-dir argument),
this script lists backed-up original mp4s and asks the user for confirmation. If the user
answers 'yes', it deletes the original files from the dataset (not from the backup).

Usage:
  python confirm_delete_originals.py --backup-dir "E:\\some_backup" --dataset-root "E:\\golfDataset\\dataset\\train" --class good

This script is interactive and will prompt before deleting.
"""

from pathlib import Path
import argparse
import logging
import sys


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s %(levelname)-5s %(message)s'
    logging.basicConfig(level=level, format=fmt)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Confirm and delete originals after splitting')
    parser.add_argument('--backup-dir', required=True, help='Backup dir used during splitting')
    parser.add_argument('--dataset-root', required=True, help='Top-level dataset root (e.g. E:\\golfDataset\\dataset\\train)')
    parser.add_argument('--class', dest='cls', required=True, choices=['good','best'], help='class folder')
    parser.add_argument('--dry-run', action='store_true', help='Report only')
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    backup_root = Path(args.backup_dir)
    if not backup_root.exists():
        logging.error(f"Backup dir not found: {backup_root}")
        sys.exit(2)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        logging.error(f"Dataset root not found: {dataset_root}")
        sys.exit(2)

    # Collect all backed up originals under backup_root
    backed = list(backup_root.rglob('*.mp4'))
    logging.info(f"Found {len(backed)} backed-up original mp4 files in {backup_root}")

    # Map backups to dataset originals (try dataset_root/<class> and dataset_root/true/<class>)
    candidates = []
    for b in backed:
        rel = b.relative_to(backup_root)
        # candidate paths to delete
        cand1 = dataset_root / args.cls / rel
        cand2 = dataset_root / 'true' / args.cls / rel
        if cand1.exists():
            candidates.append(cand1)
        elif cand2.exists():
            candidates.append(cand2)
        else:
            logging.warning(f"Original not found for backup {b} (tried {cand1} and {cand2})")

    if not candidates:
        logging.info('No originals found to delete.')
        return

    logging.info(f"Will consider deleting {len(candidates)} original files. Sample:")
    for c in candidates[:10]:
        logging.info(f"  {c}")

    if args.dry_run:
        logging.info('Dry run - no files will be deleted')
        return

    ans = input('Delete these originals from dataset? Type yes to confirm: ')
    if ans.strip().lower() != 'yes':
        logging.info('Aborted by user')
        return

    # proceed to delete
    deleted = 0
    for c in candidates:
        try:
            c.unlink()
            deleted += 1
            logging.info(f"Deleted {c}")
        except Exception as e:
            logging.error(f"Failed to delete {c}: {e}")

    logging.info(f"Deleted {deleted}/{len(candidates)} originals")


if __name__ == '__main__':
    main()
