import numpy as np
import time
from collections import deque
from feature_extractor import (
    eye_aspect_ratio, 
    brow_tension, 
    mouth_aspect_ratio, 
    facial_entropy, 
    head_stability
)

class ScoringEngine:
    """
    Calculates a holistic stress score using statistical deviations (Z-scores)
    from a baseline. Incorporates a weighted smoothing formula for stability.
    """
    def __init__(self, calibration_frames: int = 150, verbose: bool = False):
        self.calibration_frames = calibration_frames
        self.calibrated = False
        self.verbose = verbose
        
        # Calibration baselines [mean, std]
        self.baselines = {
            'blink': [0.0, 1.0],
            'brow': [0.0, 1.0],
            'lip': [0.0, 1.0],
            'jitter': [0.0, 1.0],
            'posture': [0.0, 1.0]
        }
        
        # Rolling storage for calibration phase
        self.calib_data = {k: [] for k in self.baselines.keys()}
        self.frame_count = 0
        
        # Emotional Smoothing (Exponential Moving Average)
        self.stress_ema = 0.0
        self.alpha = 0.15 # Smoothing factor
        
        # Feature Weights (Brow and Jitter are pro-level indicators)
        self.weights = {
            'blink': 0.15,
            'brow': 0.35,
            'lip': 0.10,
            'jitter': 0.25,
            'posture': 0.15
        }

    def update(self, landmarks, landmark_history) -> float:
        """
        Calculates and returns current normalized stress score (0-100).
        """
        # 1. Extract Current Features
        raw_features = {
            'blink': eye_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144]),
            'brow': brow_tension(landmarks),
            'lip': mouth_aspect_ratio(landmarks, [13, 14, 61, 291]),
            'jitter': facial_entropy(landmarks, landmark_history),
            'posture': head_stability(landmarks)
        }
        
        # 2. Calibration Phase
        if not self.calibrated:
            for k, v in raw_features.items():
                self.calib_data[k].append(v)
            
            self.frame_count += 1
            if self.frame_count >= self.calibration_frames:
                for k in self.baselines.keys():
                    self.baselines[k] = [np.mean(self.calib_data[k]), np.std(self.calib_data[k]) + 1e-6]
                self.calibrated = True
            return 0.0 # Return 0 during calibration
            
        # 3. Calculate Z-Scores (Standard Deviations from Normal)
        z_scores = {}
        for k, v in raw_features.items():
            mean, std = self.baselines[k]
            z = (v - mean) / std
            # Clip outlier extremes for stability
            z_scores[k] = np.clip(abs(z), 0, 5)
            
        # 4. Weighted Integration
        instant_stress = sum(z_scores[k] * self.weights[k] for k in z_scores.keys())
        
        # Convert to 0-100 scale (Assuming Z=3 is very high stress)
        normalized_stress = (instant_stress / 3.0) * 100
        normalized_stress = np.clip(normalized_stress, 0, 100)
        
        # 5. Apply EMA Smoothing (prevents jumpy charts)
        self.stress_ema = (self.alpha * normalized_stress) + (1 - self.alpha) * self.stress_ema
        
        self.last_z_scores = z_scores
        return float(self.stress_ema)
