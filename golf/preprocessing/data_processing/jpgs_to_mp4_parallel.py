# C:/Users/{User}/AppData/Local/Programs/Python/Python38/python.exe -m pip install ffmpeg-python 해야함
# C:/Users/조규찬/AppData/Local/Programs/Python/Python38/python.exe -m pip install imageio[ffmpeg]


from pathlib import Path
from collections import defaultdict
import os, re
import concurrent.futures
# ffmpeg 바이너리 직접 지정 (pip만 쓸 경우 반드시 필요)
# https://www.gyan.dev/ffmpeg/builds/ 에서 ffmpeg.exe 다운로드 후 아래 경로로 수정
FFMPEG_EXE_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"  # 본인 ffmpeg.exe 경로로 수정
if not os.path.exists(FFMPEG_EXE_PATH):
    raise RuntimeError(f"ffmpeg.exe not found at: {FFMPEG_EXE_PATH}\n경로를 본인 환경에 맞게 수정하세요.")
os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE_PATH

import imageio_ffmpeg
import ffmpeg  # ffmpeg-python

ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
print("ffmpeg_bin:", ffmpeg_bin)
print("ffmpeg_bin exists:", os.path.exists(ffmpeg_bin))
os.environ["PATH"] = os.path.dirname(ffmpeg_bin) + os.pathsep + os.environ["PATH"]
print("PATH:", os.environ["PATH"])

# ✏️ 경로 설정 (test, train 등 원하는 루트로 변경)
DATASET_BASE_PATH = Path(r"E:\golfDataset_dlc\dataset\train")
FPS = 30  # 출력 비디오 FPS
MAX_WORKERS = 8

def find_all_leaf_jpg_dirs(base_path: Path):
    """
    true/best/jpg/label/source, false/bad/jpg/label/source 등 모든 leaf jpg 폴더 반환
    """
    jpg_dirs = []
    for tf in ["true", "false"]:
        tf_dir = base_path / tf
        if not tf_dir.exists():
            continue
        for eval_dir in tf_dir.iterdir():
            if not eval_dir.is_dir():
                continue
            jpg_root = eval_dir / "jpg"
            if not jpg_root.exists():
                continue
            for label_dir in jpg_root.iterdir():
                if not label_dir.is_dir():
                    continue
                for source_dir in label_dir.iterdir():
                    if source_dir.is_dir():
                        jpg_dirs.append(source_dir)
    return jpg_dirs


def images_to_video_in_dir(img_dir: Path, fps: int = FPS):
    """
    한 폴더 내 <prefix>_0000.jpg 묶음을 MP4로 변환, 변환된 JPG 삭제, video 폴더에 저장
    (ffmpeg-python 사용)
    """
    # video 폴더 루트 (실제 하위 경로는 label/source 구조로 만듭니다)
    video_root = img_dir.parent.parent.parent / "video"


    jpgs = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
    # support .jpg/.jpeg (any case) and variable-length frame numbers
    jpgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    pat = re.compile(r"(.+)_([0-9]+)\.jpe?g", re.IGNORECASE)
    groups = defaultdict(list)  # prefix -> list of (filename, frame_num)
    for f in jpgs:
        m = pat.match(f)
        if m:
            prefix = m.group(1)
            num = int(m.group(2))
            groups[prefix].append((f, num))

    skip_log = []

    for prefix, files in groups.items():
        # files: list of (filename, frame_num)
        files.sort(key=lambda x: x[1])
        if not files:
            continue
        filenames = [fn for fn, _ in files]
        nums_sorted = [n for _, n in files]
        min_num, max_num = nums_sorted[0], nums_sorted[-1]
        expected = list(range(min_num, max_num + 1))
        gap = len(expected) - len(nums_sorted)
        # 프레임 개수 50 이하인 경우 JPG들 삭제하고 로그에 기록
        if len(files) <= 50:
            deleted = []
            for fn, _ in files:
                try:
                    (img_dir / fn).unlink()
                    deleted.append(str((img_dir / fn).resolve()))
                except Exception as e:
                    print(f"⚠️ 삭제 실패: {(img_dir / fn)}: {e}")
            if deleted:
                dl_log = img_dir / "deleted_small_sequences.log"
                try:
                    with open(dl_log, "a", encoding="utf-8") as lf:
                        for p in deleted:
                            lf.write(p + "\n")
                except Exception as e:
                    print(f"⚠️ 로그 기록 실패: {dl_log}: {e}")
                print(f"🗑️  Deleted {len(deleted)} small JPGs for '{prefix}' in {img_dir} (<=50 frames)")
            continue
        # 시퀀스 gap이 5 초과면 스킵 (log 남김)
        if gap > 5:
            # files is list of (filename, frame_num) tuples; use filename only
            skip_log.extend([str(img_dir / fn) for fn, _ in files])
            continue
        # gap이 5 이하인 경우는 log에 남기지 않고 실제 jpg만 합침
        # gap이 5 이하여도 빈 프레임 복제 없이, 존재하는 jpg만 시퀀스에 포함
        # ffmpeg-python에서 concat demuxer를 사용해 실제 파일만 합침
        import tempfile
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=str(img_dir)) as listfile:
            for fn in filenames:
                # use forward slashes for ffmpeg concat list on Windows
                p = (img_dir / fn).resolve().as_posix()
                listfile.write(f"file '{p}'\n")
            list_txt_path = listfile.name
        # preserve label folder to avoid collisions: img_dir.parent is label
        video_dir = video_root / img_dir.parent.name / img_dir.name
        video_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = video_dir / f"{prefix}.mp4"
        cmd = [
            ffmpeg_bin,
            '-y',
            '-f', 'concat', '-safe', '0',
            '-i', list_txt_path,
            '-r', str(fps),
            '-pix_fmt', 'yuv420p',
            str(out_mp4)
        ]
        import subprocess
        try:
            proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0:
                print(f"▶ {out_mp4.name}  ({len(filenames)} frames, gap {gap}) in {img_dir}")
                deleted_count = 0
                for fn, _ in files:
                    try:
                        (img_dir / fn).unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"⚠️ 삭제 실패: {(img_dir / fn)}: {e}")
                print(f"🗑️  Deleted {deleted_count} JPGs for '{prefix}' in {img_dir}")
            else:
                print(f"❌ ffmpeg failed for '{prefix}' in {img_dir}. rc={proc.returncode}")
                print("ffmpeg stderr:", proc.stderr.strip())
                # 기록: 실패한 ffmpeg 호출을 로그로 남김
                try:
                    err_log = img_dir / "ffmpeg_failed.log"
                    with open(err_log, "a", encoding="utf-8") as ef:
                        ef.write(f"prefix={prefix}, out={out_mp4}, rc={proc.returncode}\n")
                        ef.write(proc.stderr.strip() + "\n")
                except Exception as e:
                    print(f"⚠️ ffmpeg 실패 로그 기록 실패: {err_log}: {e}")
        except subprocess.CalledProcessError as e:
            # keep for backward compatibility, but prefer handled proc above
            print(f"❌ ffmpeg failed for '{prefix}' in {img_dir}: {e}")
        finally:
            os.remove(list_txt_path)

    # 스킵된 jpg 로그 저장
    if skip_log:
        log_path = img_dir / "skipped_jpg.log"
        with open(log_path, "w", encoding="utf-8") as logf:
            for f in skip_log:
                logf.write(f + "\n")

if __name__ == "__main__":
    print("=== JPG→MP4 변환 시작 ===")
    jpg_dirs = find_all_leaf_jpg_dirs(DATASET_BASE_PATH)
    print(f"총 {len(jpg_dirs)}개 jpg 폴더 발견")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(images_to_video_in_dir, jpg_dirs))
    print("\n모든 분류 폴더의 비디오 변환 및 JPG 삭제 완료.")

    # 모든 작업 후 jpg 폴더 통째로 삭제 (비어있지 않아도 강제 삭제)
    # jpg 폴더 삭제 기능 제거됨
