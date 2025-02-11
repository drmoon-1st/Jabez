import cv2
import csv
import os
from config import VIDEO_PATH, OUTPUT_VIDEO_PATH, CSV_FILE_PATH
from pose_detection import detect_pose, draw_landmarks_on_frame

def process_video():
    """비디오를 처리하여 포즈 랜드마크를 감지하고 CSV 및 비디오 파일로 저장."""
    
    frame_idx = 0

    cap = cv2.VideoCapture(VIDEO_PATH)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    with open(CSV_FILE_PATH, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["frame_idx"] + [f"landmark_{i}_{axis}" for i in range(33) for axis in ["x", "y", "z"]]
        csv_writer.writerow(header)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detection_result = detect_pose(frame, frame_idx, fps)
            # 랜드마크 시각화 및 CSV 저장 (수정된 부분)
            if detection_result.pose_landmarks:  # 랜드마크가 감지된 경우에만 처리
                for pose_landmark in detection_result.pose_landmarks:
                    row = [frame_idx]
                    for landmark in pose_landmark:  # 각 랜드마크의 x, y, z 좌표 추출
                        row.extend([landmark.x, landmark.y, landmark.z])  # x, y, z 추가
                    csv_writer.writerow(row)
            else:
                print(f"[경고] 프레임 {frame_idx}에서 랜드마크를 감지하지 못했습니다.")

            annotated_frame = draw_landmarks_on_frame(frame, detection_result)
            out.write(annotated_frame)
            frame_idx += 1  #  프레임 인덱스 증가 (다시 실행해도 정상 동작)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"영상 처리 완료: {OUTPUT_VIDEO_PATH}")
    print(f"포즈 랜드마크 저장 완료: {CSV_FILE_PATH}")
