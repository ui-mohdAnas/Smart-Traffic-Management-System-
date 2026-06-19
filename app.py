"""
app.py -- Streamlit dashboard for the traffic congestion pipeline.

Run locally:    streamlit run app.py
Deploy:         push this repo to GitHub, then deploy on
                Streamlit Community Cloud (share.streamlit.io) or
                Hugging Face Spaces (SDK: Streamlit) -- both are free.

Requires: pip install yt-dlp   (for automatic YouTube live stream resolution)
"""

import os
import time
import tempfile

import cv2
import pandas as pd
import streamlit as st

from detector import CongestionDetector

# Give FFMPEG (used internally by OpenCV) connection/read timeouts so it
# fails fast instead of hanging forever on a bad/expired/slow stream URL.
# Values are in microseconds. Works for RTSP, HLS (.m3u8), MJPEG, etc.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;10000000|rw_timeout;15000000"
)


def resolve_stream_url(url: str, format_str: str = "best[height<=480][ext=mp4]/best[height<=480]") -> str:
    """
    If the given URL is a normal YouTube page/live link, use yt-dlp
    (as a Python library, not subprocess) to resolve it to a direct,
    fresh HLS (.m3u8) URL right now -- at the exact moment it's needed.

    This removes the need to manually run yt-dlp in a terminal and
    copy-paste a huge expiring URL (error-prone), and it also avoids
    the URL expiring between extraction and use, since it's fetched
    fresh every time "Start processing" is clicked.

    If the URL is already a direct stream (rtsp/m3u8/mjpeg/etc.), it
    is returned unchanged.
    """
    if "youtube.com" in url or "youtu.be" in url:
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": format_str,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info["url"]
    return url


st.set_page_config(page_title="Traffic Congestion Prediction", layout="wide")
st.title("🚦 Real-Time Traffic Congestion Prediction")
st.caption("YOLOv8 vehicle detection + centroid tracking + density/speed based congestion scoring")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Source")
source_type = st.sidebar.radio("Input type", ["Upload video", "Webcam", "Stream URL"])

video_source = None
raw_stream_input = None
if source_type == "Upload video":
    uploaded = st.sidebar.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_source = tfile.name
elif source_type == "Webcam":
    video_source = 0  # default local webcam device index
else:
    raw_stream_input = st.sidebar.text_input(
        "Stream URL (RTSP / HLS / MJPEG / YouTube link)",
        placeholder="Paste a YouTube watch/live URL, or a direct RTSP/HLS/MJPEG URL",
    )
    st.sidebar.caption(
        "Tip: you can paste a normal YouTube link directly (e.g. "
        "youtube.com/watch?v=... or youtube.com/live/...) -- it will be "
        "auto-resolved to a fresh stream URL each time you click Start."
    )
    resolution_choice = st.sidebar.selectbox(
        "YouTube stream quality (lower = more stable real-time reading)",
        ["480p (recommended)", "360p", "720p", "Best available"],
        index=0,
    )
    _res_map = {
        "480p (recommended)": "best[height<=480][ext=mp4]/best[height<=480]",
        "360p": "best[height<=360][ext=mp4]/best[height<=360]",
        "720p": "best[height<=720][ext=mp4]/best[height<=720]",
        "Best available": "best[ext=mp4]/best",
    }
    yt_format_str = _res_map[resolution_choice]

st.sidebar.header("Detection settings")
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
roi_top_frac = st.sidebar.slider("Ignore above this fraction of frame (sky/buildings)", 0.0, 0.8, 0.3, 0.05)
frame_skip = st.sidebar.slider("Process every Nth frame (higher = faster, less smooth)", 1, 5, 1)

st.sidebar.header("Congestion thresholds")
max_vehicles = st.sidebar.slider("Vehicle count = fully jammed", 5, 50, 20)
max_speed = st.sidebar.slider("Pixel displacement/frame = free flow", 5, 60, 25)

run = st.sidebar.button("▶ Start processing", type="primary")

# ---------------- Main layout ----------------
col_video, col_metrics = st.columns([2, 1])
frame_placeholder = col_video.empty()
status_placeholder = col_video.empty()
count_metric = col_metrics.empty()
speed_metric = col_metrics.empty()
level_metric = col_metrics.empty()
chart_placeholder = col_metrics.empty()

MAX_CONSECUTIVE_FAILED_READS = 30  # ~ a few seconds of dropped frames before giving up

if run:
    if source_type == "Stream URL":
        if not raw_stream_input:
            st.warning("Please provide a video source first.")
            st.stop()
        with st.spinner("Resolving stream URL..."):
            try:
                video_source = resolve_stream_url(raw_stream_input, yt_format_str)
            except Exception as e:
                st.error(
                    f"Could not resolve this URL via yt-dlp: {e}\n\n"
                    "If this is a YouTube link, the video may be private, "
                    "region-locked, or not actually live. If it's a direct "
                    "RTSP/HLS/MJPEG URL, double check it's correct and reachable."
                )
                st.stop()
        st.sidebar.caption(f"Resolved URL:\n{video_source[:90]}...")
    elif not video_source:
        st.warning("Please provide a video source first.")
        st.stop()

    detector = CongestionDetector(
        conf_threshold=conf_threshold,
        max_vehicles=max_vehicles,
        max_speed=max_speed,
    )

    with st.spinner("Connecting to source..."):
        if isinstance(video_source, str) and video_source.startswith(("http://", "https://", "rtsp://")):
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        st.error(
            "Could not open this video source. Check the file/URL/webcam index.\n\n"
            "If this is a stream URL, try clicking Start again (a fresh URL "
            "is resolved every time for YouTube links)."
        )
        st.stop()

    history = pd.DataFrame(columns=["vehicle_count"])
    frame_idx = 0
    failed_reads = 0
    reconnect_attempts = 0
    MAX_RECONNECT_ATTEMPTS = 5
    is_youtube_source = source_type == "Stream URL" and (
        "youtube.com" in (raw_stream_input or "") or "youtu.be" in (raw_stream_input or "")
    )
    stop_button = col_video.button("⏹ Stop")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            failed_reads += 1
            status_placeholder.warning(
                f"No frame received ({failed_reads}/{MAX_CONSECUTIVE_FAILED_READS}) -- retrying..."
            )
            if failed_reads >= MAX_CONSECUTIVE_FAILED_READS:
                # For YouTube sources, try a full reconnect: re-resolve a
                # fresh URL (gets us back to the live edge) and reopen
                # the capture, instead of giving up immediately.
                if is_youtube_source and reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
                    reconnect_attempts += 1
                    status_placeholder.warning(
                        f"Stream stalled -- reconnecting ({reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})..."
                    )
                    cap.release()
                    try:
                        fresh_url = resolve_stream_url(raw_stream_input, yt_format_str)
                        cap = cv2.VideoCapture(fresh_url, cv2.CAP_FFMPEG)
                        failed_reads = 0
                        time.sleep(1)
                        continue
                    except Exception:
                        pass  # fall through to giving up below if reopen fails

                st.error(
                    "Lost connection to the stream (no frames after repeated attempts"
                    + (" and reconnects" if is_youtube_source else "") + "). "
                    "Live YouTube streams can stall if processing falls behind the "
                    "live edge -- try a lower resolution setting, or test with a "
                    "direct MJPEG/RTSP traffic-cam feed instead."
                )
                break
            time.sleep(0.3)
            continue

        # Reset failure counter once a frame succeeds
        failed_reads = 0
        status_placeholder.empty()

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 360))
        annotated, metrics = detector.process_frame(frame, roi_top_frac=roi_top_frac)

        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        count_metric.metric("Vehicle count (ROI)", metrics["vehicle_count"])
        speed_metric.metric("Avg speed proxy (px/frame)", metrics["avg_speed_px"])
        level_metric.metric("Congestion level", metrics["congestion_level"],
                             delta=f"score {metrics['congestion_score']}")

        history.loc[len(history)] = [metrics["vehicle_count"]]
        if len(history) > 200:
            history = history.iloc[-200:]
        chart_placeholder.line_chart(history)

        time.sleep(0.01)  # tiny yield so Streamlit can refresh the UI

    cap.release()
    st.success("Stream ended.")
else:
    st.info("Choose a source on the left, then click **Start processing**.")
