import numpy as np
from typing import Dict, List, Tuple

class ScoringEngine:
    """
    Handles calibration and real-time stress scoring using Z-score normalization.
    
    This engine establishes a baseline from initial data frames and calculates
    a weighted stress score based on deviations (Z-scores) from that baseline.
    """
    def __init__(self, calibration_frames: int = 300, verbose: bool = False):
        """
        Initializes the engine for calibration.

        Parameters:
            calibration_frames (int): Number of frames for the baseline calibration. Default is 300.
            verbose (bool): If True, prints extra debugging info to the console. Default is False.
        """
        self.calibration_frames = calibration_frames
        self.verbose = verbose
        self.frames_collected = 0
        self.is_calibrated = False
        
        # Internal storage
        self.calibration_data = {'blink': [], 'brow': [], 'lip': [], 'asymmetry': []}
        self.baseline = {}
        self.rolling_scores = []
        
        # Weighted stress contribution factors
        self.weights = {'blink': 0.30, 'brow': 0.25, 'lip': 0.25, 'asymmetry': 0.20}
        self.raw_z_scores = {}

    def calibrate(self, features: Dict[str, float]) -> None:
        """
        Collects feature data for the baseline during the calibration phase.

        Parameters:
            features (Dict[str, float]): Dictionary containing current values for all features.
        """
        if self.is_calibrated:
            return

        for k, v in features.items():
            if k in self.calibration_data:
                self.calibration_data[k].append(v)
        
        self.frames_collected += 1
        
        # Check if calibration is complete
        if self.frames_collected >= self.calibration_frames:
            for k, v in self.calibration_data.items():
                mean_val = np.mean(v)
                std_val = np.std(v)
                # Avoid division by zero: if std is too low, use a small constant
                self.baseline[k] = {
                    'mean': mean_val,
                    'std': std_val if std_val > 0.0001 else 0.0001
                }
            self.is_calibrated = True
            
            if self.verbose:
                print(f"[Engine] Calibration Complete. Baseline: {self.baseline}")

    def calculate_score(self, current_features: Dict[str, float]) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculates a smoothed stress score (0-100) using current feature deviations.
        Formula: Z = (current_value - baseline_mean) / baseline_std

        Parameters:
            current_features (Dict[str, float]): The current calculated values for all signals.

        Returns:
            Tuple[float, str, Dict[str, float]]: (Final Score, Stress Level Band, Feature Z-Scores).
        """
        if not self.is_calibrated:
            return 0.0, "CALIBRATING", {}

        z_scores = {}
        for k, v in current_features.items():
            mean = self.baseline[k]['mean']
            std = self.baseline[k]['std']
            # Z-score normalization
            z = (v - mean) / std
            z_scores[k] = z
        
        self.raw_z_scores = z_scores
        
        # Calculate raw weighted score using absolute Z-score (deviation indicates stress)
        raw_weighted_score = sum(abs(z_scores[k]) * self.weights[k] for k in self.weights)
        
        # Map raw weighted score to 0-100% scale
        # Typically a total Z-score of 5.0 across features represents very high stress
        final_mapped_score = min(100, max(0, raw_weighted_score * 20))
        
        # Smooth using a 30-frame rolling average for stability
        self.rolling_scores.append(final_mapped_score)
        if len(self.rolling_scores) > 30:
            self.rolling_scores.pop(0)
        
        smoothed_score = float(np.mean(self.rolling_scores))
        
        # Categorize into stress level bands
        status = "LOW"
        if 30 <= smoothed_score < 60:
            status = "MODERATE"
        elif smoothed_score >= 60:
            status = "HIGH"
            
        return smoothed_score, status, z_scores
