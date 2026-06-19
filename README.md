# 🚦 Real-Time Traffic Congestion Prediction

Computer-vision pipeline that ingests traffic video (file, webcam, or live
stream URL), detects vehicles frame-by-frame with YOLOv8, tracks them across
frames, and scores congestion in real time based on vehicle density and
movement speed.

## How it works

```
video source (file / webcam / RTSP-HLS-MJPEG URL)
        │  cv2.VideoCapture
        ▼
   frame (resized 640x360)
        │
        ▼
  YOLOv8n detection  ──► filter to car/bus/truck/motorcycle
        │
        ▼
  Centroid tracker   ──► assigns persistent IDs, computes per-vehicle
        │                displacement between frames (speed proxy)
        ▼
  Congestion scoring ──► score = 0.6·density + 0.4·(1 - speed)
        │                Low / Medium / High, color-coded
        ▼
  Streamlit dashboard ──► live annotated video + metrics + history chart
```

- **Detection**: `yolov8n.pt`, pretrained on COCO. No training required —
  it already recognizes the 4 vehicle classes we need.
- **Tracking**: a from-scratch centroid tracker (`tracker.py`) — nearest-
  neighbor matching across frames. This is what turns "count" into "speed".
- **Congestion score**: rule-based, not a black box — you can explain every
  number in an interview. `max_vehicles` and `max_speed` are tunable
  thresholds (sidebar sliders in the app).

## Setup (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```
First run auto-downloads the YOLOv8n weights (~6 MB) — needs internet once.

## Getting test video (do this first — don't start with a flaky live feed)

Pick one:
- Kaggle: "Highway Traffic Videos Dataset", "Real-Time Traffic Dataset - 500 Videos",
  or "Road Traffic Video Monitoring" — search those names on kaggle.com/datasets.
- Record 1–2 minutes of traffic from your window/phone.
- Any dashcam/highway clip from Pexels (free stock video, no attribution needed).

Upload it in the app's "Upload video" mode. Get this working and looking good
**before** touching a live source — this is your fallback if the demo Wi-Fi
fails in front of a recruiter.

## Adding a genuinely live source (the resume-justifying part)

**Option A — TfL JamCam API (London, free, no approval wait)**
Used by published OpenCV vehicle-counting projects already. Sign up for a
free key at the Transport for London API portal, then pull a camera's MJPEG/
clip URL and pass it straight to `cv2.VideoCapture(url)`.

**Option B — YouTube live traffic cameras**
```bash
pip install yt-dlp
yt-dlp -g "https://www.youtube.com/watch?v=<live-traffic-cam-id>"
```
This prints a direct stream URL — paste it into the app's "Stream URL" mode.
Note: the URL expires every few hours, so re-run `yt-dlp -g` if it goes stale.
Good for a live demo recording; not something to hardcode into deployment.

**Option C — Any public DOT/city camera with an RTSP/HLS/MJPEG endpoint**
`cv2.VideoCapture()` doesn't care about the source as long as it's a stream
OpenCV can decode — same code path as a file.

## Deploying it

1. Push this folder to a GitHub repo (use `opencv-python-headless`, already
   in `requirements.txt` — the GUI version of opencv breaks on headless
   cloud servers).
2. **Streamlit Community Cloud** (share.streamlit.io): connect the repo,
   point it at `app.py`, deploy. Free, takes ~3 minutes.
   *or*
   **Hugging Face Spaces**: create a Space, SDK = Streamlit, push the repo.
3. On a free CPU tier, expect a few frames/sec with YOLOv8n — that's normal
   and fine for a congestion *level* (which doesn't change frame-to-frame
   anyway). If you want it visibly snappier, raise "process every Nth frame"
   in the sidebar.
4. For the deployed demo, default to "Upload video" mode — cloud servers
   usually can't reach your local webcam, and outbound live-stream URLs from
   a free-tier server can be unreliable. Keep the live-stream demo as a
   screen recording for your resume/portfolio video if needed.

## Honest framing for your resume

Bullet that matches what's actually built:

> Built a real-time vehicle detection and tracking pipeline (YOLOv8 + OpenCV)
> that ingests traffic video from files, webcams, or live RTSP/HLS streams,
> and computes a density- and speed-based congestion score with a live
> Streamlit dashboard; deployed on Streamlit Community Cloud.

If you add the live YouTube/TfL source and actually run it: it's accurate to
say "live traffic camera streams." If you only demo on recorded video, say
"live and recorded traffic video" — still a strong, true claim, and a
recruiter who asks a follow-up question won't catch you overstating.

## Stretch goals (worth adding if you have extra time — strengthens the "major project" framing)

- Per-lane congestion (split the ROI into 2–3 vertical strips, score each)
- A simple time-series forecast (e.g. moving average or Prophet) predicting
  congestion 10–15 minutes ahead from logged history
- Alerting (Telegram/email webhook) when congestion crosses "High" for >30s
- Swap YOLOv8n → YOLOv8s/m if deploying with GPU access, for better accuracy
