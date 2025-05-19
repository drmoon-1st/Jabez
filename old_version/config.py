import os

# 기본 경로 설정
ROOT_PATH = os.getcwd()
VIDEO_NAME = "test.mp4"

# 모델 파일 경로
MODEL_PATH = f"{ROOT_PATH}/pose_landmarker_heavy.task"

# 입력 및 출력 경로 설정
VIDEO_PATH = f"{ROOT_PATH}/markerless_video/{VIDEO_NAME}"
OUTPUT_VIDEO_PATH = f"{ROOT_PATH}/output_video/output_processed.mp4"
CSV_OUTPUT_DIR = f"{ROOT_PATH}/markerless_raw_csv"
CSV_FILE_PATH = os.path.join(CSV_OUTPUT_DIR, "pose_landmarks.csv")

# CSV 폴더 생성
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
