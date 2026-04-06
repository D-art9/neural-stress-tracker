import cv2
import time
import numpy as np
from typing import Tuple, Optional, List, Dict
from face_tracker import FaceTracker
from feature_extractor import eye_aspect_ratio, brow_tension, mouth_aspect_ratio, facial_asymmetry
from scoring_engine import ScoringEngine

class StressDetectorController:
    """
    Coordinates between MediaPipe landmarks, signal feature extraction, 
    and the scoring engine.
    
    This class manages the webcam capture lifecycle, per-frame landmark
    tracking and regional processing.
    """
    def __init__(self, calibration_frames: int = 300, verbose: bool = False, use_webcam: bool = False):
        """
        Initializes the detection pipeline.

        Parameters:
            calibration_frames (int): Length of initial data gathering in frames.
            verbose (bool): If True, logs feature values for debugging.
            use_webcam (bool): If True, initializes internal webcam capture.
        """
        self.tracker = FaceTracker()
        self.scorer = ScoringEngine(calibration_frames, verbose=verbose)
        self.verbose = verbose
        self.cap = None
        
        if use_webcam:
            # Webcam fallback: try 0, then 1, then 2
            for idx in [0, 1, 2]:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    break
            else:
                raise Exception("Failed to open any webcam device.")

        self.current_score = 0.0
        self.status = "CALIBRATING"
        self.last_second_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()
        self.blink_per_min = 0.0
        self.z_scores = {}
        self.history = []

    def run_on_frame(self) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """
        Captures a frame from internal webcam and processes it.
        """
        if self.cap is None:
            return None, None
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        return self.process_external_frame(frame)

    def process_external_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Runs signal analysis on a provided frame.
        """
        h, w, _ = frame.shape
        landmarks = self.tracker.process_frame(frame)
        
        if landmarks:
            # Phase 1: Feature Extraction
            self.blink_per_min = eye_aspect_ratio(landmarks, self.tracker.LEFT_EYE)
            avg_bt = brow_tension(landmarks)
            mar = mouth_aspect_ratio(landmarks, self.tracker.LIPS)
            asym = facial_asymmetry(landmarks)
            
            current_feats = {'blink': self.blink_per_min, 'brow': avg_bt, 'lip': mar, 'asymmetry': asym}
            
            # Phase 2: Scoring
            if not self.scorer.is_calibrated:
                self.scorer.calibrate(current_feats)
                self.status = "CALIBRATING"
                self.current_score = 0.0
            else:
                self.current_score, self.status, self.z_scores = self.scorer.calculate_score(current_feats)
                
            self.history.append(self.current_score)
            if len(self.history) > 600:
                self.history.pop(0)

            # Performance monitoring
            self.frame_count += 1
            if time.time() - self.fps_timer >= 1.0:
                self.fps = self.frame_count / (time.time() - self.fps_timer)
                self.frame_count = 0
                self.fps_timer = time.time()

            # Apply overlays
            self.draw_overlay(frame, landmarks, w, h)
            
        return frame, landmarks

    def draw_overlay(self, frame: np.ndarray, landmarks: List, w: int, h: int) -> None:
        """
        Draws anatomical landmark points on the frame.
        """
        color = (0, 255, 0)
        if self.status == "HIGH": color = (0, 0, 255)
        elif self.status == "MODERATE": color = (0, 255, 255)
            
        # Draw regions
        regions = self.tracker.get_all_regions(landmarks, w, h)
        for r_name in ['left_eye', 'right_eye', 'left_brow', 'right_brow', 'lips']:
            if r_name in regions:
                for p in regions[r_name]:
                    cv2.circle(frame, tuple(p), 2, color, -1)
                
        cv2.putText(frame, f"STRESS: {self.status if self.status != 'CALIBRATING' else '...'}", 
                   (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 150, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

    def close(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    detector = StressDetectorController(use_webcam=True, verbose=True)
    try:
        while True:
            frame, _ = detector.run_on_frame()
            if frame is not None:
                cv2.imshow("Stress Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        detector.close()
        cv2.destroyAllWindows()
