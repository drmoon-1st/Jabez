"""
pack_upload.py

ZIP 파일을 만들고 백엔드에 presigned URL 요청 후 업로드 처리
"""
import zipfile
import os
from pathlib import Path
import json
import requests


def zip_dir(src_dir: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                full = Path(root) / f
                rel = full.relative_to(src_dir)
                zf.write(full, arcname=str(rel))


def write_manifest(out_dir: Path, object_name: str, frame_count: int):
    manifest = {
        "object_name": object_name,
        "frame_count": int(frame_count)
    }
    try:
        p = Path(out_dir) / 'manifest.json'
        p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def request_presigned_and_upload(token: str, backend_base: str, s3_object_name: str, zip_path: Path, timeout=30):
    """POST /api/upload with Authorization header, then PUT to presigned URL."""
    backend_base = backend_base.rstrip('/')
    url = f"{backend_base}/api/upload"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    payload = {
        'upload_source': '3D',
        'original_filename': zip_path.name,
        'file_type': 'application/zip',
        'file_size_bytes': zip_path.stat().st_size
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    presigned_url = j.get('presigned_url') or j.get('url')
    job_id = j.get('job_id')
    if not presigned_url:
        raise RuntimeError('presigned_url missing in response')

    # upload via PUT
    with open(zip_path, 'rb') as fh:
        put_hdr = {'Content-Type': 'application/zip'}
        putr = requests.put(presigned_url, data=fh, headers=put_hdr, timeout=120)
        if not (200 <= putr.status_code < 300):
            raise RuntimeError(f'Upload failed: {putr.status_code} {putr.text}')

    return {'job_id': job_id, 'presigned_url': presigned_url}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('token')
    p.add_argument('backend')
    p.add_argument('zip')
    args = p.parse_args()
    print(request_presigned_and_upload(args.token, args.backend, 'realsense_pkg', Path(args.zip)))
