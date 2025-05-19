import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from config import MODEL_PATH

# MediaPipe Pose 설정
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

def detect_pose(frame, frame_idx, fps):
    """비디오 프레임에서 포즈 랜드마크를 감지하는 함수."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return pose_landmarker.detect_for_video(mp_image, timestamp_ms=frame_idx * (1000 // fps))

def draw_landmarks_on_frame(frame, detection_result):
    """포즈 랜드마크를 프레임에 시각화하는 함수."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_frame = np.copy(frame)
    
    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_frame
