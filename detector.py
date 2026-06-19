"""
detector.py
-----------
Wraps a pretrained YOLOv8n model (trained on COCO, so it already knows how
to find cars/buses/trucks/motorcycles out of the box -- no training data
or GPU needed) and turns raw detections into a congestion reading.

Congestion score = weighted blend of:
  - density_score : how many vehicles are in the region of interest, relative
                     to a configurable "this many vehicles = jam" ceiling
  - speed_score    : how slowly vehicles are moving, relative to a
                     configurable "this is free-flow speed" ceiling
A higher score = more congested. This rule-based approach is the standard
first pass used in real traffic-monitoring literature before anyone reaches
for a trained congestion classifier, and it's defensible in an interview
because you can explain exactly why a number came out the way it did.
"""

from ultralytics import YOLO
import numpy as np
import cv2

from tracker import CentroidTracker

# COCO class ids for vehicles
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class CongestionDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.35,
                 max_vehicles=20, max_speed=25.0, density_weight=0.6):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker = CentroidTracker()
        self.max_vehicles = max_vehicles   # vehicle count considered "fully jammed"
        self.max_speed = max_speed         # px/frame considered "free flowing"
        self.density_weight = density_weight

    def process_frame(self, frame, roi_top_frac=0.0):
        """
        frame: BGR numpy array (one video frame)
        roi_top_frac: ignore detections above this fraction of frame height
                      (e.g. 0.4 excludes sky/buildings, keeps the road)
        Returns: annotated_frame, dict of metrics
        """
        h, w = frame.shape[:2]
        roi_y = int(h * roi_top_frac)

        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]

        centroids, boxes = [], []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if cy < roi_y:
                continue  # outside region of interest
            centroids.append((cx, cy))
            boxes.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls_id]))

        tracked = self.tracker.update(centroids)

        # draw ROI line + boxes
        annotated = frame.copy()
        if roi_top_frac > 0:
            cv2.line(annotated, (0, roi_y), (w, roi_y), (255, 255, 0), 2)
        for x1, y1, x2, y2, label in boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        vehicle_count = len(centroids)
        speeds = [self.tracker.get_speed(oid) for oid in tracked.keys()]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        density_score = min(vehicle_count / max(self.max_vehicles, 1), 1.0)
        speed_score = 1.0 - min(avg_speed / max(self.max_speed, 1e-6), 1.0)
        congestion_score = (self.density_weight * density_score +
                             (1 - self.density_weight) * speed_score)

        if congestion_score < 0.33:
            level, color = "Low", (0, 200, 0)
        elif congestion_score < 0.66:
            level, color = "Medium", (0, 165, 255)
        else:
            level, color = "High", (0, 0, 255)

        cv2.putText(annotated, f"Congestion: {level}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        metrics = {
            "vehicle_count": vehicle_count,
            "avg_speed_px": round(avg_speed, 2),
            "congestion_score": round(congestion_score, 3),
            "congestion_level": level,
        }
        return annotated, metrics
