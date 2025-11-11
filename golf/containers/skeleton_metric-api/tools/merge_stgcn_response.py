"""Utility to merge an existing <jobid>_stgcn_response.json into <jobid>.json in-place.

Usage: python tools/merge_stgcn_response.py <dest_dir> <job_id>

This reads the response file and writes back the job json atomically with stgcn_inference merged.
"""
import sys
import json
from pathlib import Path


def _safe_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        tmp.replace(path)
    except Exception:
        import os
        os.replace(str(tmp), str(path))


def main():
    if len(sys.argv) < 3:
        print('Usage: python merge_stgcn_response.py <dest_dir> <job_id>')
        return
    dest_dir = Path(sys.argv[1])
    job_id = sys.argv[2]
    resp = dest_dir / f"{job_id}_stgcn_response.json"
    j = dest_dir / f"{job_id}.json"
    if not resp.exists():
        print('No response file:', resp)
        return
    if not j.exists():
        print('No job json:', j)
        return
    try:
        resp_obj = json.loads(resp.read_text(encoding='utf-8'))
    except Exception as e:
        print('Failed to read response:', e)
        return
    try:
        job = json.loads(j.read_text(encoding='utf-8'))
    except Exception as e:
        print('Failed to read job json:', e)
        return
    rj = resp_obj.get('resp_json') if isinstance(resp_obj, dict) else None
    if isinstance(rj, dict) and 'result' in rj:
        job['stgcn_inference'] = rj['result']
    else:
        job['stgcn_inference'] = resp_obj
    _safe_write_json(j, job)
    print('Merged response into', j)


if __name__ == '__main__':
    main()
