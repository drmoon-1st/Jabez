import pandas as pd
import matplotlib.pyplot as plt
from config import VIDEO_PATH, OUTPUT_VIDEO_PATH, CSV_FILE_PATH

# CSV 파일 읽기
df = pd.read_csv(CSV_FILE_PATH)

# 랜드마크 번호와 신체 부위 매핑
landmark_mapping = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",       # 손가락 끝 (새끼손가락)
    18: "right_pinky",      # 손가락 끝 (새끼손가락)
    19: "left_index",       # 손가락 끝 (검지손가락)
    20: "right_index",      # 손가락 끝 (검지손가락)
    21: "left_thumb",       # 엄지손가락
    22: "right_thumb",      # 엄지손가락
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",        # 발뒤꿈치
    30: "right_heel",       # 발뒤꿈치
    31: "left_foot_index",  # 발끝 (검지발가락)
    32: "right_foot_index"  # 발끝 (검지발가락)
}

# 열 이름 변경
new_columns = []
for col in df.columns:
    if col.startswith("landmark_"):
        parts = col.split("_")   # 예를 들어, 'landmark_11_x'를 분리
        landmark_idx = int(parts[1])   # '11' 추출
        axis = parts[2]               # 'x', 'y', 'z' 추출
        new_name = f"{landmark_mapping[landmark_idx]}_{axis}"
        new_columns.append(new_name)
    else:
        new_columns.append(col)       # frame_idx 등 다른 열은 그대로 유지

df.columns = new_columns

# 변경된 데이터프레임 저장 (선택 사항)
# output_csv_path = "renamed_pose_landmarks.csv"
# df.to_csv(output_csv_path, index=False)

count_landmarks = df[[
        'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',  # 왼쪽 어깨
        'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',  # 오른쪽 어깨
        'left_elbow_x', 'left_elbow_y', 'left_elbow_z',  # 왼쪽 팔꿈치
        'right_elbow_x', 'right_elbow_y', 'right_elbow_z'   # 오른쪽 팔꿈치
    ]]

def graph():
    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(count_landmarks["left_elbow_y"], marker='o', label="Left Shoulder Y")
    plt.title("Left Shoulder Y Values Over Time", fontsize=16)
    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Left Shoulder Y Value", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def count_rep():
    count_landmarks = df[[
            'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',  # 왼쪽 어깨
            'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',  # 오른쪽 어깨
            'left_elbow_x', 'left_elbow_y', 'left_elbow_z',  # 왼쪽 팔꿈치
            'right_elbow_x', 'right_elbow_y', 'right_elbow_z'   # 오른쪽 팔꿈치
        ]]
    threshold = min(count_landmarks["left_shoulder_y"])  # 어깨의 y 좌표 기준값

    count = 0
    above = False  # 현재 상태 (어깨보다 높은지 여부)
    state = "below"

    for y in count_landmarks["left_elbow_y"]:
        if state == "below" and y > threshold:
            state = "above"  # 높아짐
        elif state == "above" and y < threshold:
            state = "below"  # 낮아짐 → 한 사이클 완료
            count += 1

    print("Count:", count)