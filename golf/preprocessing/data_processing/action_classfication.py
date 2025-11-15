

import os
import json
import shutil
import gc
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 경로 설정
# Training/Validation, Public/Assosiation/Amature, male/female 로 경로 나뉨
training_base_path = r"E:\스포츠 사람 동작 영상(골프)\Training"
dataset_base_path = r"E:\golfDataset_dlc\dataset\train"   # 저장은 다른 접근이 쉬운 dataset 폴더에

# 처리할 하위 폴더들 정의
categories = ["Public", "Amateur", "Association"]  # Amateur 오타 수정
genders = ["male", "female"]

# 배치 처리 설정
BATCH_SIZE = 1000  # 한 번에 처리할 파일 수
PROGRESS_INTERVAL = 50  # 진행 상황 출력 간격
MAX_WORKERS = 8  # 너무 많은 동시 작업은 I/O 병목 유발, 8~12 추천

# 기준
true_evaluations = {"best", "good", "normal"}
false_evaluations = {"bad", "worst"}
all_evaluations = true_evaluations | false_evaluations

# 평가값별 폴더 구조 생성
evaluation_folders = {
    "true": ["best", "good", "normal"],
    "false": ["bad", "worst"]
}
for dataset, evals in evaluation_folders.items():
    for eval_name in evals:
        for ext in ["json", "jpg"]:
            os.makedirs(os.path.join(dataset_base_path, dataset, eval_name, ext), exist_ok=True)
# unknown, other 폴더 생성
for ext in ["json", "jpg"]:
    os.makedirs(os.path.join(dataset_base_path, "unknown", ext), exist_ok=True)
    os.makedirs(os.path.join(dataset_base_path, "other", ext), exist_ok=True)


def process_one_file(args):
    json_path, base_path = args
    try:
        relative_parts = os.path.relpath(json_path, base_path).split(os.sep)
        relative_parts[0] = relative_parts[0].replace("[라벨]", "[원천]")
        jpg_filename = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"
        jpg_path = os.path.join(base_path, *relative_parts[:-1], jpg_filename)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            evaluation = data.get("image", {}).get("evaluation", "").lower()

        if not evaluation:
            unknown_json_dir = os.path.join(dataset_base_path, "unknown", "json")
            unknown_jpg_dir = os.path.join(dataset_base_path, "unknown", "jpg")
            os.makedirs(unknown_json_dir, exist_ok=True)
            os.makedirs(unknown_jpg_dir, exist_ok=True)
            dest_json_path = os.path.join(unknown_json_dir, os.path.basename(json_path))
            shutil.copy2(json_path, dest_json_path)
            try: os.remove(json_path)
            except Exception as e: print(f"원본 JSON 삭제 실패: {json_path} - {e}")
            if os.path.exists(jpg_path):
                dest_jpg_path = os.path.join(unknown_jpg_dir, jpg_filename)
                shutil.copy2(jpg_path, dest_jpg_path)
                try: os.remove(jpg_path)
                except Exception as e: print(f"원본 JPG 삭제 실패: {jpg_path} - {e}")
            return 1, 0

        if evaluation not in all_evaluations:
            other_json_dir = os.path.join(dataset_base_path, "other", "json")
            other_jpg_dir = os.path.join(dataset_base_path, "other", "jpg")
            os.makedirs(other_json_dir, exist_ok=True)
            os.makedirs(other_jpg_dir, exist_ok=True)
            dest_json_path = os.path.join(other_json_dir, os.path.basename(json_path))
            shutil.copy2(json_path, dest_json_path)
            try: os.remove(json_path)
            except Exception as e: print(f"원본 JSON 삭제 실패: {json_path} - {e}")
            if os.path.exists(jpg_path):
                dest_jpg_path = os.path.join(other_jpg_dir, jpg_filename)
                shutil.copy2(jpg_path, dest_jpg_path)
                try: os.remove(jpg_path)
                except Exception as e: print(f"원본 JPG 삭제 실패: {jpg_path} - {e}")
            return 1, 0

        if evaluation in true_evaluations:
            tf_label = "true"
            eval_label = evaluation
        elif evaluation in false_evaluations:
            tf_label = "false"
            eval_label = evaluation

        relative_path = os.path.relpath(json_path, base_path)
        path_parts = relative_path.split(os.sep)
        label_folder = path_parts[0].replace("[라벨]", "")
        source_folder = path_parts[1]

        json_dest_dir = os.path.join(dataset_base_path, tf_label, eval_label, "json", label_folder, source_folder)
        jpg_dest_dir = os.path.join(dataset_base_path, tf_label, eval_label, "jpg", label_folder, source_folder)
        os.makedirs(json_dest_dir, exist_ok=True)
        os.makedirs(jpg_dest_dir, exist_ok=True)

        dest_json_path = os.path.join(json_dest_dir, os.path.basename(json_path))
        shutil.copy2(json_path, dest_json_path)
        try: os.remove(json_path)
        except Exception as e: print(f"원본 JSON 삭제 실패: {json_path} - {e}")

        if os.path.exists(jpg_path):
            dest_jpg_path = os.path.join(jpg_dest_dir, jpg_filename)
            shutil.copy2(jpg_path, dest_jpg_path)
            try: os.remove(jpg_path)
            except Exception as e: print(f"원본 JPG 삭제 실패: {jpg_path} - {e}")
        return 1, 0

    except Exception as e:
        print(f"파일 처리 중 예외: {args[0]} - {e}")
        return 0, 1

def collect_json_files():
    if collect_json_files.has_run:
        return collect_json_files.cached_result
    print("JSON 파일 탐색 중...")
    json_files = []
    for category in categories:
        for gender in genders:
            base_path = os.path.join(training_base_path, category, gender)
            if not os.path.exists(base_path):
                print(f"경로가 존재하지 않음: {base_path}")
                continue
            print(f"탐색 중: {category}/{gender}")
            for root, dirs, files in os.walk(base_path):
                if "[라벨]" not in root:
                    continue
                for file in files:
                    if file.endswith(".json"):
                        json_path = os.path.join(root, file)
                        json_files.append((json_path, base_path))
    print(f"총 {len(json_files)}개의 JSON 파일 발견")
    collect_json_files.has_run = True
    collect_json_files.cached_result = json_files
    return json_files
collect_json_files.has_run = False
collect_json_files.cached_result = None


if __name__ == "__main__":
    print("=== 골프 데이터 분류 시작 ===")
    start_time = time.time()

    all_json_files = collect_json_files()

    if not all_json_files:
        print("처리할 JSON 파일이 없습니다.")
    else:
        total_files = len(all_json_files)
        total_processed = 0
        total_errors = 0
        BATCH_SIZE_THREAD = 1000
        for batch_start in range(0, total_files, BATCH_SIZE_THREAD):
            batch_end = min(batch_start + BATCH_SIZE_THREAD, total_files)
            batch = all_json_files[batch_start:batch_end]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_one_file, args) for args in batch]
                for idx, future in enumerate(as_completed(futures), 1):
                    processed, errors = future.result()
                    total_processed += processed
                    total_errors += errors
                    if (idx % 100 == 0) or (batch_start + idx == total_files):
                        print(f"진행: {batch_start + idx}/{total_files} 파일 완료 (누적 처리: {total_processed}개, 오류: {total_errors}개)")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n=== 처리 완료 ===")
        print(f"총 처리 시간: {elapsed_time:.2f}초")
        print(f"성공적으로 처리된 파일: {total_processed}개")
        print(f"오류가 발생한 파일: {total_errors}개")
        print(f"처리 속도: {total_processed/elapsed_time:.2f}파일/초")
