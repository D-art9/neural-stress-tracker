import cv2
import time
import numpy as np
from typing import Tuple, Optional, List, Dict
from face_tracker import FaceTracker
from scoring_engine import ScoringEngine

class StressDetectorController:
    """
    Coordinates between MediaPipe landmarks, signal feature extraction, 
    and the weighted scoring engine.
    """
    def __init__(self, calibration_frames: int = 150, verbose: bool = False, use_webcam: bool = False):
        self.tracker = FaceTracker()
        self.scorer = ScoringEngine(calibration_frames, verbose=verbose)
        self.verbose = verbose
        self.cap = None
        self.landmark_history = []
        
        if use_webcam:
            for idx in [0, 1, 2]:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
            else:
                raise Exception("Failed to open any webcam device.")

        self.current_score = 0.0
        self.status = "CALIBRATING"
        self.z_scores = {}
        self.history = []
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()

    def process_external_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Runs signal analysis using the latest weighted scoring engine.
        """
        h, w, _ = frame.shape
        landmarks = self.tracker.process_frame(frame)
        
        if landmarks:
            # Maintain landmark history for entropy/jitter analysis
            self.landmark_history.append(landmarks)
            if len(self.landmark_history) > 10:
                self.landmark_history.pop(0)
            
            # Phase 1: Scoring with weighted biometrics
            self.current_score = self.scorer.update(landmarks, self.landmark_history)
            self.z_scores = self.scorer.last_z_scores if hasattr(self.scorer, 'last_z_scores') else {}
            
            # Phase 2: Status Mapping
            if not self.scorer.calibrated:
                self.status = f"CALIBRATING ({self.scorer.frame_count}/{self.scorer.calibration_frames})"
            else:
                if self.current_score > 70: self.status = "HIGH"
                elif self.current_score > 35: self.status = "MODERATE"
                else: self.status = "LOW"
                
            self.history.append(self.current_score)
            if len(self.history) > 600: self.history.pop(0)

            # Performance monitoring
            self.frame_count += 1
            if time.time() - self.fps_timer >= 1.0:
                self.fps = self.frame_count / (time.time() - self.fps_timer)
                self.frame_count = 0
                self.fps_timer = time.time()

            # Apply anatomical overlays
            self.draw_overlay(frame, landmarks, w, h)
            
        return frame, landmarks

    def draw_overlay(self, frame: np.ndarray, landmarks: List, w: int, h: int) -> None:
        color = (0, 200, 0)
        if "HIGH" in self.status: color = (0, 0, 255)
        elif "MODERATE" in self.status: color = (0, 255, 255)
            
        # Draw regional landmarks
        regions = self.tracker.get_all_regions(landmarks, w, h)
        for r_name in ['left_eye', 'right_eye', 'left_brow', 'right_brow', 'lips']:
            if r_name in regions:
                for p in regions[r_name]:
                    cv2.circle(frame, tuple(p), 1, color, -1)
                
        cv2.putText(frame, f"STRESS: {self.status}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)

    def close(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
