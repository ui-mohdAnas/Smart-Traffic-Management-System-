"""
app.py -- Streamlit dashboard for the traffic congestion pipeline.

Run locally:    streamlit run app.py
Deploy:         push this repo to GitHub, then deploy on
                Streamlit Community Cloud (share.streamlit.io) or
                Hugging Face Spaces (SDK: Streamlit) -- both are free.
"""

import time
import tempfile

import cv2
import pandas as pd
import streamlit as st

from detector import CongestionDetector

st.set_page_config(page_title="Traffic Congestion Prediction", layout="wide")
st.title("🚦 Real-Time Traffic Congestion Prediction")
st.caption("YOLOv8 vehicle detection + centroid tracking + density/speed based congestion scoring")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Source")
source_type = st.sidebar.radio("Input type", ["Upload video", "Webcam", "Stream URL"])

video_source = None
if source_type == "Upload video":
    uploaded = st.sidebar.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_source = tfile.name
elif source_type == "Webcam":
    video_source = 0  # default local webcam device index
else:
    video_source = st.sidebar.text_input(
        "Stream URL (RTSP / HLS / MJPEG / YouTube-extracted URL)",
        placeholder="https://...",
    )

st.sidebar.header("Detection settings")
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
roi_top_frac = st.sidebar.slider("Ignore above this fraction of frame (sky/buildings)", 0.0, 0.8, 0.3, 0.05)
frame_skip = st.sidebar.slider("Process every Nth frame (higher = faster, less smooth)", 1, 10, 3)
proc_width = st.sidebar.select_slider(
    "Processing resolution width px (lower = faster, less accurate)",
    options=[320, 416, 480, 640], value=416,
)

st.sidebar.header("Congestion thresholds")
max_vehicles = st.sidebar.slider("Vehicle count = fully jammed", 5, 50, 20)
max_speed = st.sidebar.slider("Pixel displacement/frame = free flow", 5, 60, 25)

run = st.sidebar.button("▶ Start processing", type="primary")

# ---------------- Main layout ----------------
col_video, col_metrics = st.columns([2, 1])
frame_placeholder = col_video.empty()
count_metric = col_metrics.empty()
speed_metric = col_metrics.empty()
level_metric = col_metrics.empty()
chart_placeholder = col_metrics.empty()

if run:
    if not video_source:
        st.warning("Please provide a video source first.")
        st.stop()

    detector = CongestionDetector(
        conf_threshold=conf_threshold,
        max_vehicles=max_vehicles,
        max_speed=max_speed,
    )
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Could not open this video source. Check the file/URL/webcam index.")
        st.stop()

    history = pd.DataFrame(columns=["vehicle_count"])
    frame_idx = 0
    stop_button = col_video.button("⏹ Stop")
    fps_metric = col_metrics.empty()

    while cap.isOpened():
        # Lightweight skip: grab() pulls the next frame from the decoder
        # buffer WITHOUT decoding it -- much cheaper than read(), which
        # does grab()+decode(). We only pay the decode cost on frames we
        # actually process. This is the fix for "processing falls behind
        # the video" -- the old loop decoded every single frame even when
        # skipping it, wasting most of the CPU budget on frames we threw away.
        grabbed = cap.grab()
        if not grabbed:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

        t_start = time.time()
        scale = proc_width / frame.shape[1]
        frame = cv2.resize(frame, (proc_width, int(frame.shape[0] * scale)))
        annotated, metrics = detector.process_frame(frame, roi_top_frac=roi_top_frac)
        elapsed = time.time() - t_start

        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        count_metric.metric("Vehicle count (ROI)", metrics["vehicle_count"])
        speed_metric.metric("Avg speed proxy (px/frame)", metrics["avg_speed_px"])
        level_metric.metric("Congestion level", metrics["congestion_level"],
                             delta=f"score {metrics['congestion_score']}")
        fps_metric.caption(f"Processing speed: {1/elapsed:.1f} fps "
                            f"({elapsed*1000:.0f} ms/frame on this server's CPU)")

        history.loc[len(history)] = [metrics["vehicle_count"]]
        if len(history) > 200:
            history = history.iloc[-200:]
        chart_placeholder.line_chart(history)

    cap.release()
    st.success("Stream ended.")
else:
    st.info("Choose a source on the left, then click **Start processing**.")
