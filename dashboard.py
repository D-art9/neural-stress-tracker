import streamlit as st
import cv2
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import threading
from typing import List, Optional
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av

from main import StressDetectorController

# Page configuration
st.set_page_config(page_title="AI Stress Detection (Cloud Ready)", layout="wide")

# Enhanced RTC Configuration with fallback stun servers
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]},
            {"urls": ["stun:stun.services.mozilla.com:3478"]}
        ]
    }
)

class StressVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.detector = None
        self.lock = threading.Lock()
        self.landmark_history = []
        self.latest_data = {
            'score': 0.0, 'status': "INITIALIZING",
            'z_scores': {}, 'history': []
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            if self.detector is None:
                with self.lock:
                    self.detector = StressDetectorController(calibration_frames=100)
            
            # Process & Update
            processed_img, landmarks = self.detector.process_external_frame(img)
            
            with self.lock:
                self.latest_data['score'] = self.detector.current_score
                self.latest_data['status'] = self.detector.status
                self.latest_data['z_scores'] = self.detector.z_scores
                self.latest_data['history'] = list(self.detector.history)
                
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🧠 Neural Facial Stress Detection")
st.markdown("Cloud-deployed suite utilizing multi-provider STUN fallbacks for stability.")

# Layout
col_feed, col_metrics = st.columns([2, 1], gap="small")

with col_feed:
    ctx = webrtc_streamer(
        key="stress-tracker-v3",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=StressVideoProcessor,
        # Reduced constraints for firewall stability
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "frameRate": {"ideal": 15}},
            "audio": False
        },
        async_processing=False, # Must stay False for Python 3.11 stability
    )

with col_metrics:
    gauge_placeholder = st.empty()
    status_placeholder = st.empty()
    bar_chart_placeholder = st.empty()

st.markdown("---")
timeline_placeholder = st.empty()

# UI Update Loop
if ctx.video_processor:
    while ctx.state.playing:
        if ctx.video_processor is None: break
            
        with ctx.video_processor.lock:
            data = ctx.video_processor.latest_data.copy()
        
        if not data or 'status' not in data:
            time.sleep(0.1)
            continue
            
        # Update UI Elements
        score = data.get('score', 0.0)
        status = data.get('status', 'INITIALIZING')
        color_map = {"HIGH": "#FF0000", "MODERATE": "#FFCC00", "LOW": "#00CC00"}
        gauge_color = color_map.get(status, "#99DDFF")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color}}
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
        gauge_placeholder.plotly_chart(fig_gauge, use_container_width=True)
        status_placeholder.markdown(f"<h3 style='text-align: center; color: {gauge_color};'>{status}</h3>", unsafe_allow_html=True)

        z_data = data.get('z_scores', {})
        if z_data:
            pretty_names = {"blink": "Blink", "brow": "Brow", "lip": "Lip", "jitter": "Jitter", "posture": "Tilt"}
            df_z = pd.DataFrame({
                'Signal': [pretty_names.get(k, k) for k in z_data.keys()],
                'Deviation': list(z_data.values())
            })
            fig_bar = px.bar(df_z, x='Signal', y='Deviation', color='Deviation',
                           color_continuous_scale=['#00CC00', '#FF0000'], range_y=[-3, 3])
            fig_bar.update_layout(height=250, margin=dict(l=5, r=5, t=5, b=5), showlegend=False)
            bar_chart_placeholder.plotly_chart(fig_bar, use_container_width=True)

        history = data.get('history', [])
        if len(history) > 1:
            fig_line = px.line(x=np.arange(len(history)), y=history)
            fig_line.update_layout(yaxis_range=[0, 105], height=200, xaxis_title="Frames", yaxis_title="Stress Index")
            timeline_placeholder.plotly_chart(fig_line, use_container_width=True)

        time.sleep(0.5)
else:
    st.info("👋 Welcome! Click 'Start' and allow camera access to begin.")
