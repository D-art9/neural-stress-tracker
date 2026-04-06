# Neural Facial Micro-Expression Detection System

**Real-time, non-invasive stress analysis powered by computer vision.**

## 🔍 Problem Statement
High-stress environments (trading, competitive gaming, deep work) often lead to behavioral signals that go unnoticed by the individual. This project provides a **non-invasive** way to detect and score these micro-expressions using nothing but a standard webcam. By analyzing facial landmarks, the system identifies stress-related behavioral signals such as increased blink rates, brow furrowing, lip compression, and facial asymmetry.

## ⚙️ How It Works
The system captures human facial geometry using **MediaPipe FaceMesh** (468 landmarks) and quantifies stress across 4 primary behavioral vectors:

1.  **Blink Rate (EAR)**: Eye Aspect Ratio tracking. A drop in EAR below 0.20 signals a blink. The system tracks "Blinks Per Minute" (BPM) against your baseline.
2.  **Brow Tension**: Combines the vertical distance between brows and eyelids with horizontal furrowing (inner brow points).
3.  **Lip Compression (MAR)**: Mouth Aspect Ratio captures tightening of the lips often associated with intense focus or agitation.
4.  **Facial Asymmetry**: Compares 5 mirrored landmark pairs relative to the nose-tip midline to detect involuntary micro-asymmetries.

### 🧠 Scoring Engine
- **Calibration**: The first 10 seconds establish your unique "neutral" baseline (mean and standard deviation).
- **Z-Score Normalization**: Each signal is scored based on its standard deviation from the baseline ($Z = \frac{x - \mu}{\sigma}$).
- **Weighted Aggregation**: Signals are weighted (Blink: 30%, Brow: 25%, Lip: 25%, Asymmetry: 20%) to produce a final 0-100 Stress Index.
- **Smoothing**: A 30-frame rolling average ensures stable, meaningful readings without frame-by-frame jitter.

## 🛠️ Tech Stack
- **Languages**: Python 3.9 / 3.10
- **Frameworks**: OpenCV (Vision), MediaPipe (Tracking), Streamlit (UI)
- **Visualization**: Plotly (Real-time analytics), Pandas (Data structures), NumPy (Math)

## 📂 Project Structure
- `main.py`: Coordinates the sensor pipeline, webcam state, and sensor aggregation.
- `face_tracker.py`: High-fidelity MediaPipe FaceMesh wrapper for anatomical landmark extraction.
- `feature_extractor.py`: Quantitative signal calculators for EAR, MAR, brow tension, and asymmetry.
- `scoring_engine.py`: Baseline calibration and statistical Z-score scoring logic.
- `dashboard.py`: Thread-safe Streamlit UI for real-time visualization and system control.

## 🚀 Setup & Installation
1.  Navigate to the project root: `cd stress_detector`
2.  Install all dependencies: `pip install -r requirements.txt`
3.  Launch the application: `streamlit run dashboard.py`

## 🕹️ Use & Instructions
1.  **Start Detection**: Click "Start AI Analysis" in the sidebar.
2.  **Calibration**: Look neutrally at the camera for the first 10 seconds. The status bar will show "CALIBRATING". Do not talk or express during this time.
3.  **Live Monitoring**: 
    - **Left Panel**: View the live stream with tracking overlays. Color changes based on stress levels (Green -> Yellow -> Red).
    - **Center Panel**: Real-time Gauge showing the normalized 0-100 Stress Score.
    - **Right Panel**: Detailed Z-score deviations for each individual biometric signal.
    - **Bottom Panel**: Historical timeline showing stress volatility over the session.

## ⚠️ Limitations
- **Medical Disclaimer**: This is an experimental biometric project, not a medical or lie-detection device.
- **Lighting**: Works best in well-lit environments where facial shadows are minimized.
- **Occlusion**: Glasses or heavy facial hair may slightly decrease landmark accuracy.
- **Motion**: Frequent head tilting or rapid movement can introduce noise into the asymmetry and EAR signals.

---
**Author**: Devang Aswani
