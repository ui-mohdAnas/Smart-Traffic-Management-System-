"""
CentroidTracker
----------------
A minimal multi-object tracker. YOLO gives us detections *per frame* with
no memory of "this car is the same car as last frame". This tracker fixes
that by matching new detections to existing tracked objects based on
nearest-centroid distance. That's what lets us measure displacement
between frames -> speed -> feed into the congestion score.

This is intentionally simple (no Kalman filter, no deep-learning re-ID).
For a fast-moving road scene at 10-15 processed fps it's accurate enough,
and it's easy to explain in an interview, which matters for a resume project.
"""

from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict()        # object_id -> centroid (x, y)
        self.history = OrderedDict()        # object_id -> list of past centroids
        self.disappeared = OrderedDict()    # object_id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.history[self.next_object_id] = [centroid]
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.history[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        """centroids: list of (x, y) tuples detected in the current frame."""
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array(centroids)

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(tuple(c))
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # distance matrix: existing objects x new detections
        D = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = tuple(input_centroids[col])
            self.history[object_id].append(tuple(input_centroids[col]))
            if len(self.history[object_id]) > 10:
                self.history[object_id].pop(0)
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            self.register(tuple(input_centroids[col]))

        return self.objects

    def get_speed(self, object_id):
        """Average pixel displacement per frame over recent history (proxy for speed)."""
        pts = self.history.get(object_id, [])
        if len(pts) < 2:
            return 0.0
        dists = [np.linalg.norm(np.array(pts[i]) - np.array(pts[i - 1])) for i in range(1, len(pts))]
        return float(np.mean(dists))
