import cv2
import numpy as np
import mediapipe as mp
from collections import deque


# ============================================================
# CENTROID TRACKER
# ============================================================
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(
            np.array(object_centroids)[:, None] - np.array(input_centroids),
            axis=2
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects


# ============================================================
# GAZE DETECTOR
# ============================================================
class GazeDetector:
    def __init__(self, max_faces=3):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.tracker = CentroidTracker()

        self.buffer_size = 5
        self.prediction_buffer = {}

        # Calibration
        self.calibrated = False
        self.calibration_frames = 30
        self.calibration_data = []
        self.iris_center = 0

        # Attention tracking
        self.attention_window_size = 150  # ~5 sec at 30 FPS
        self.attention_stats = {}

    # ---------------------------------------------------------
    # Eye Visibility
    # ---------------------------------------------------------
    def check_eye_visibility(self, frame, landmarks, w, h):
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        def eye_region(points):
            coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in points]
            x_vals = [p[0] for p in coords]
            y_vals = [p[1] for p in coords]
            return frame[min(y_vals):max(y_vals), min(x_vals):max(x_vals)]

        def visible(eye_img):
            if eye_img.size == 0:
                return False
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            return np.var(gray) > 50

        return visible(eye_region(LEFT_EYE)), visible(eye_region(RIGHT_EYE))

    # ---------------------------------------------------------
    # Head Pose
    # ---------------------------------------------------------
    def estimate_head_pose(self, landmarks, w, h):
        nose = landmarks[1]
        left_face = landmarks[234]
        right_face = landmarks[454]

        mid_x = (left_face.x + right_face.x) / 2
        horizontal_offset = nose.x - mid_x

        facing = abs(horizontal_offset) < 0.3
        nose_pt = (int(nose.x * w), int(nose.y * h))

        return facing, nose_pt

    # ---------------------------------------------------------
    # Iris Normalized Position
    # ---------------------------------------------------------
    def analyze_iris_position(self, landmarks):
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        LEFT_EYE_CORNERS = [33, 133]
        RIGHT_EYE_CORNERS = [362, 263]

        def center(points):
            coords = np.array([[landmarks[p].x, landmarks[p].y] for p in points])
            return coords.mean(axis=0)

        left_iris = center(LEFT_IRIS)
        right_iris = center(RIGHT_IRIS)

        left_corner_l = landmarks[LEFT_EYE_CORNERS[0]]
        left_corner_r = landmarks[LEFT_EYE_CORNERS[1]]
        right_corner_l = landmarks[RIGHT_EYE_CORNERS[0]]
        right_corner_r = landmarks[RIGHT_EYE_CORNERS[1]]

        left_eye_width = abs(left_corner_r.x - left_corner_l.x)
        right_eye_width = abs(right_corner_r.x - right_corner_l.x)

        left_ratio = (left_iris[0] - left_corner_l.x) / left_eye_width if left_eye_width > 0 else 0.5
        right_ratio = (right_iris[0] - right_corner_l.x) / right_eye_width if right_eye_width > 0 else 0.5

        return (left_ratio + right_ratio) / 2

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------
    def update_calibration(self, iris_ratio):
        if not self.calibrated:
            self.calibration_data.append(iris_ratio)
            if len(self.calibration_data) >= self.calibration_frames:
                self.iris_center = np.mean(self.calibration_data)
                self.calibrated = True

    # ---------------------------------------------------------
    # Attention Tracking
    # ---------------------------------------------------------
    def update_attention(self, object_id, looking):
        if object_id not in self.attention_stats:
            self.attention_stats[object_id] = {
                "window": deque(maxlen=self.attention_window_size),
                "total": 0,
                "attentive": 0
            }
        stats = self.attention_stats[object_id]
        stats["window"].append(looking)
        stats["total"] += 1
        if looking:
            stats["attentive"] += 1

    def get_attention_metrics(self, object_id):
        stats = self.attention_stats.get(object_id, None)
        if not stats:
            return 0, 0
        rolling = (sum(stats["window"]) / len(stats["window"])) * 100 if stats["window"] else 0
        session = (stats["attentive"] / stats["total"]) * 100 if stats["total"] else 0
        return rolling, session

    def reset_attention(self, object_id):
        """Reset attention stats for a given tracker object_id."""
        if object_id in self.attention_stats:
            del self.attention_stats[object_id]

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------
    def classify_gaze(self, iris_ratio, facing):
        score = 0
        if facing:
            score += 1.0
        if 0.40 < iris_ratio < 0.60:
            score += 2.0
        looking = score >= 2.0
        confidence = score / 3.0
        return looking, confidence

    # ---------------------------------------------------------
    # Main Detection — returns list of result dicts
    # ---------------------------------------------------------
    def detect(self, frame):
        """
        Returns a list of dicts:
          - status: "calibrating" | "detected"
          - object_id: int tracker ID
          - looking: bool
          - confidence: float 0-1
          - rolling_attention: float %
          - session_attention: float %
          - nose_pt: (x, y) pixel coords
          - facing: bool
          - iris_ratio: float
        Or a single dict with status "calibrating" if still warming up.
        Or empty list if no face detected.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        detections = []

        if not result.multi_face_landmarks:
            self.tracker.update([])
            return []

        for face_landmarks in result.multi_face_landmarks[:3]:
            landmarks = face_landmarks.landmark
            left_visible, right_visible = self.check_eye_visibility(frame, landmarks, w, h)
            if not left_visible or not right_visible:
                continue

            facing, nose_pt = self.estimate_head_pose(landmarks, w, h)
            iris_ratio = self.analyze_iris_position(landmarks)
            self.update_calibration(iris_ratio)

            detections.append({
                "nose": nose_pt,
                "facing": facing,
                "iris_ratio": iris_ratio
            })

        centroids = [d["nose"] for d in detections]
        objects = self.tracker.update(centroids)

        if not self.calibrated:
            calib_progress = len(self.calibration_data)
            return [{"status": "calibrating", "progress": calib_progress, "total": self.calibration_frames}]

        outputs = []

        for object_id, centroid in objects.items():
            closest_det = None
            min_dist = float("inf")

            for det in detections:
                dist = np.linalg.norm(np.array(det["nose"]) - np.array(centroid))
                if dist < min_dist:
                    min_dist = dist
                    closest_det = det

            if closest_det is None:
                continue

            looking, confidence = self.classify_gaze(
                closest_det["iris_ratio"],
                closest_det["facing"]
            )

            self.update_attention(object_id, looking)
            rolling_attn, session_attn = self.get_attention_metrics(object_id)

            outputs.append({
                "status": "detected",
                "object_id": object_id,
                "looking": looking,
                "confidence": confidence,
                "rolling_attention": rolling_attn,
                "session_attention": session_attn,
                "nose_pt": closest_det["nose"],
                "facing": closest_det["facing"],
                "iris_ratio": closest_det["iris_ratio"]
            })

        return outputs
