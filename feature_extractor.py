import numpy as np
import time
from typing import List

# Internal module-level state for stateful counting of blinks over time
_blink_times = []
_last_ear = 1.0

def eye_aspect_ratio(landmarks: List, eye_indices: List[int]) -> float:
    """
    Calculates the Eye Aspect Ratio (EAR) and returns the current blinks per minute (BPM).
    """
    global _blink_times, _last_ear
    p = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in eye_indices]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
    if ear < 0.20 and _last_ear >= 0.20:
        _blink_times.append(time.time())
    _last_ear = ear
    now = time.time()
    _blink_times = [t for t in _blink_times if now - t < 60]
    return float(len(_blink_times))

def brow_tension(landmarks: List) -> float:
    """
    Calculates numerical brow tension based on vertical brow-eyelid distance.
    """
    v_l = np.linalg.norm(np.array([landmarks[70].x, landmarks[70].y]) - np.array([landmarks[160].x, landmarks[160].y]))
    v_r = np.linalg.norm(np.array([landmarks[317].x, landmarks[317].y]) - np.array([landmarks[385].x, landmarks[385].y]))
    h_dist = np.linalg.norm(np.array([landmarks[107].x, landmarks[107].y]) - np.array([landmarks[336].x, landmarks[336].y]))
    return (v_l + v_r + h_dist) / 3.0

def mouth_aspect_ratio(landmarks: List, lip_indices: List[int]) -> float:
    """
    Calculates the Mouth Aspect Ratio (MAR) to detect lip compression or yawning.
    """
    p = [np.array([landmarks[idx].x, landmarks[idx].y]) for idx in lip_indices]
    vertical = np.linalg.norm(p[0] - p[1])
    horizontal = np.linalg.norm(p[2] - p[3])
    return vertical / horizontal if horizontal > 0 else 0.0

def facial_asymmetry(landmarks: List) -> float:
    """
    Calculates a facial asymmetry score based on mirrored left/right landmark pairs.
    """
    nose_x = landmarks[1].x
    pairs = [(33, 263), (70, 300), (78, 308), (172, 397), (63, 293)]
    asym_scores = []
    for l_idx, r_idx in pairs:
        l_p, r_p = landmarks[l_idx], landmarks[r_idx]
        d_l, d_r = abs(l_p.x - nose_x), abs(r_p.x - nose_x)
        v_diff = abs(l_p.y - r_p.y)
        asym_scores.append(abs(d_l - d_r) + v_diff)
    return float(np.mean(asym_scores))

def facial_entropy(landmarks: List, history: List[List]) -> float:
    """
    Calculates the micro-jitter (entropy) of facial landmarks over time.
    """
    if len(history) < 5: return 0.0
    tracked_indices = [1, 33, 263, 61, 291]
    variances = []
    for land_idx in tracked_indices:
        pts = [h[land_idx] for h in history if land_idx < len(h)]
        if not pts: continue
        x_vals, y_vals = [p.x for p in pts], [p.y for p in pts]
        variances.append(np.std(x_vals) + np.std(y_vals))
    return float(np.mean(variances) * 1000) if variances else 0.0

def head_stability(landmarks: List) -> float:
    """
    Measures head tilt and stability relative to the vertical axis.
    """
    nose, chin = landmarks[1], landmarks[152]
    dx, dy = chin.x - nose.x, chin.y - nose.y
    return float(abs(np.arctan2(dx, dy)))
