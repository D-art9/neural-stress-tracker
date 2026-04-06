import streamlit as st
import cv2
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import threading
from typing import List, Optional
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av

from main import StressDetectorController

# Page configuration
st.set_page_config(page_title="AI Stress Detection (Cloud Ready)", layout="wide")

# RTC Configuration for STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302", "stun:stun3.l.google.com:19302"]},
        ]
    }
)

class StressVideoProcessor(VideoProcessorBase):
    """
    Processes incoming WebRTC video frames using the StressDetectorController.
    """
    def __init__(self) -> None:
        self.detector = None
        self.lock = threading.Lock()
        self.latest_data = {
            'score': 0.0,
            'status': "INITIALIZING",
            'z_scores': {},
            'history': []
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Lazy load the detector to avoid WebRTC connection timeouts
        try:
            if self.detector is None:
                with self.lock:
                    self.detector = StressDetectorController(calibration_frames=150)
            
            # Process the frame
            processed_img, _ = self.detector.process_external_frame(img)
            
            # Share data with UI thread
            with self.lock:
                self.latest_data['score'] = self.detector.current_score
                self.latest_data['status'] = self.detector.status
                self.latest_data['z_scores'] = self.detector.z_scores
                self.latest_data['history'] = list(self.detector.history)
                
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            # Fallback for errors during processing
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# Initialize Session State
if "processor" not in st.session_state:
    st.session_state.processor = None

st.title("🧠 Neural Facial Stress Detection")
st.markdown("Cloud-deployed version using browser-native camera access via WebRTC.")
st.markdown("---")

# Layout
col_feed, col_score, col_features = st.columns([2.5, 1, 1], gap="medium")

with col_feed:
    st.subheader("📸 Browser Camera Stream")
    ctx = webrtc_streamer(
        key="stress-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=StressVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

with col_score:
    st.subheader("⚡ Stress Metrics")
    gauge_placeholder = st.empty()
    status_placeholder = st.empty()

with col_features:
    st.subheader("📊 Signal Deviations")
    bar_chart_placeholder = st.empty()

st.subheader("📉 Holistic Stress Timeline")
timeline_placeholder = st.empty()

# UI Update Loop
if ctx.video_processor:
    while ctx.state.playing:
        if ctx.video_processor is None:
            break
            
        with ctx.video_processor.lock:
            data = ctx.video_processor.latest_data.copy()
        
        # Skip if initialization isn't done or data is empty
        if not data or 'status' not in data:
            time.sleep(0.1)
            continue
            
        # Update Gauge
        score = data.get('score', 0.0)
        status = data.get('status', 'INITIALIZING')
        color_map = {"HIGH": "#FF0000", "MODERATE": "#FFCC00", "LOW": "#00CC00", "CALIBRATING": "#99DDFF"}
        gauge_color = color_map.get(status, "#CCCCCC")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 204, 0, 0.1)"},
                    {'range': [30, 60], 'color': "rgba(255, 204, 0, 0.1)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.1)"}
                ]
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
        gauge_placeholder.plotly_chart(fig_gauge, use_container_width=True)
        status_placeholder.markdown(f"<h3 style='text-align: center; color: {gauge_color};'>Level: {status}</h3>", unsafe_allow_html=True)

        # Update Bar Chart
        z_data = data.get('z_scores', {})
        if z_data:
            pretty_names = {
                "blink": "Blink Rate", 
                "brow": "Brow Tension", 
                "lip": "Lip Tightness", 
                "asymmetry": "Asymmetry",
                "jitter": "Micro-Jitter",
                "posture": "Head Tilt"
            }
            df_z = pd.DataFrame({
                'Signal': [pretty_names.get(k, k) for k in z_data.keys()],
                'Deviation (Z)': list(z_data.values())
            })
            fig_bar = px.bar(df_z, x='Signal', y='Deviation (Z)', color='Deviation (Z)',
                           color_continuous_scale=['#00CC00', '#FF0000'], range_y=[-5, 5])
            fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            bar_chart_placeholder.plotly_chart(fig_bar, use_container_width=True)

        # Update Timeline
        history = data.get('history', [])
        if len(history) > 1:
            fig_line = px.line(x=np.arange(len(history)), y=history)
            fig_line.update_layout(yaxis_range=[0, 105], height=300, xaxis_title="Time", yaxis_title="Stress Index")
            timeline_placeholder.plotly_chart(fig_line, use_container_width=True)

        time.sleep(0.5) # Throttle UI updates for stability
else:
    st.info("Please allow camera access and press 'Start' to begin analysis.")
