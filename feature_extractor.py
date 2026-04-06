import numpy as np
import time
from typing import List

# Internal module-level state for stateful counting of blinks over time
_blink_times = []
_last_ear = 1.0

def eye_aspect_ratio(landmarks: List, eye_indices: List[int]) -> float:
    """
    Calculates the Eye Aspect Ratio (EAR) and returns the current blinks per minute (BPM).
    Formula: EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    Blinks are recorded when EAR drops below the threshold of 0.20.

    Parameters:
        landmarks (List): All face mesh landmarks.
        eye_indices (List[int]): Indices of the landmarks that define the eye region.

    Returns:
        float: The current count of blinks detected within the last 60 seconds (Blink Rate).
    """
    global _blink_times, _last_ear
    
    # Extract coordinate points
    p = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in eye_indices]
    
    # Vertical distances (P2-P6, P3-P5)
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    # Horizontal distance (P1-P4)
    h = np.linalg.norm(p[0] - p[3])
    
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
    
    # Blink Detection Logic
    # Transition from above threshold to below threshold is 1 blink event
    if ear < 0.20 and _last_ear >= 0.20:
        _blink_times.append(time.time())
    _last_ear = ear
    
    # Cleanup history: remove timestamps older than 60 seconds
    now = time.time()
    _blink_times = [t for t in _blink_times if now - t < 60]
    
    return float(len(_blink_times))

def brow_tension(landmarks: List) -> float:
    """
    Calculates numerical brow tension based on vertical brow-eyelid distance 
    and horizontal brow furrowing.

    Parameters:
        landmarks (List): All face mesh landmarks.

    Returns:
        float: A composite tension score where lower values often signify higher tension.
    """
    # Vertical distance: distance from brow points (70, 336) to upper eyelid points (160, 385)
    v_l = np.linalg.norm(np.array([landmarks[70].x, landmarks[70].y]) - np.array([landmarks[160].x, landmarks[160].y]))
    v_r = np.linalg.norm(np.array([landmarks[336].x, landmarks[336].y]) - np.array([landmarks[385].x, landmarks[385].y]))
    
    # Horizontal furrowing: distance between inner brow points (107 and 336 are inner brow corners)
    h_dist = np.linalg.norm(np.array([landmarks[107].x, landmarks[107].y]) - np.array([landmarks[336].x, landmarks[336].y]))
    
    # Simple average score for the baseline calculation
    return (v_l + v_r + h_dist) / 3.0

def mouth_aspect_ratio(landmarks: List, lip_indices: List[int]) -> float:
    """
    Calculates the Mouth Aspect Ratio (MAR) to detect lip compression or yawning.
    Formula: MAR = vertical_lip_distance / horizontal_lip_distance

    Parameters:
        landmarks (List): All face mesh landmarks.
        lip_indices (List[int]): Indices of the lip region landmarks [Upper, Lower, Left, Right].

    Returns:
        float: Calculated MAR value.
    """
    p = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in lip_indices]
    vertical = np.linalg.norm(p[0] - p[1])
    horizontal = np.linalg.norm(p[2] - p[3])
    
    return vertical / horizontal if horizontal > 0 else 0.0

def facial_asymmetry(landmarks: List) -> float:
    """
    Calculates a facial asymmetry score based on mirrored left/right landmark pairs 
    referenced against the nose tip midline.

    Parameters:
        landmarks (List): All face mesh landmarks.

    Returns:
        float: Average asymmetry score (higher values signify more deviation from midline symmetry).
    """
    nose_x = landmarks[1].x
    # Representative mirrored pairs [Left_Idx, Right_Idx]
    pairs = [(33, 263), (70, 300), (78, 308), (172, 397), (63, 293)]
    
    asym_scores = []
    for l_idx, r_idx in pairs:
        l_p = landmarks[l_idx]
        r_p = landmarks[r_idx]
        
        # Horizontal deviation from midline (nose tip)
        d_l = abs(l_p.x - nose_x)
        d_r = abs(r_p.x - nose_x)
        # Vertical alignment difference
        v_diff = abs(l_p.y - r_p.y)
        
        asym_scores.append(abs(d_l - d_r) + v_diff)
        
    return float(np.mean(asym_scores))
