import cv2
import mediapipe as mp
import numpy as np
import os
from typing import List, Dict, Optional

# MediaPipe Tasks API Imports (Compatible with Python 3.14/MediaPipe 0.10.x)
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

class FaceTracker:
    """
    Handles face landmark detection using the modern MediaPipe Tasks API.
    """
    def __init__(self, model_path: str = "face_landmarker.task"):
        """
        Initializes the Face Landmarker task.
        """
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found.")

        # Configure Face Landmarker options
        base_options = BaseOptions(model_asset_path=self.model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Legacy landmark indices mapping
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_BROW = [70, 63, 105, 66, 107]
        self.RIGHT_BROW = [336, 296, 334, 293, 300]
        self.LIPS = [13, 14, 78, 308]
        self.JAW = [172, 397]
        self.NOSE_TIP = 1

    def process_frame(self, frame: np.ndarray) -> Optional[List]:
        """
        Processes a BGR image and returns detected landmarks.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run detection
        result = self.landmarker.detect(mp_image)
        
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def get_coords(self, landmarks: List, indices: List[int], img_w: int, img_h: int) -> np.ndarray:
        """
        Maps normalized landmarks to pixel coordinates.
        """
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coords.append(np.array([int(lm.x * img_w), int(lm.y * img_h)]))
        return np.array(coords)

    def get_all_regions(self, landmarks: List, img_w: int, img_h: int) -> Dict[str, np.ndarray]:
        """
        Extracts coordinate groups for all monitored facial regions.
        """
        return {
            "left_eye": self.get_coords(landmarks, self.LEFT_EYE, img_w, img_h),
            "right_eye": self.get_coords(landmarks, self.RIGHT_EYE, img_w, img_h),
            "left_brow": self.get_coords(landmarks, self.LEFT_BROW, img_w, img_h),
            "right_brow": self.get_coords(landmarks, self.RIGHT_BROW, img_w, img_h),
            "lips": self.get_coords(landmarks, self.LIPS, img_w, img_h),
            "jaw": self.get_coords(landmarks, self.JAW, img_w, img_h),
            "nose_tip": self.get_coords(landmarks, [self.NOSE_TIP], img_w, img_h)[0]
        }

    def close(self):
        """Releases the landmarker resources."""
        self.landmarker.close()
