"""Face detection and landmark tracking using MediaPipe or OpenCV fallback."""

import time
import math
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from src.vision.head_pose import HeadPoseEstimator


class FaceTracker:
    """Tracks face detection and facial landmarks."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.face_count_history: list = []

        if MEDIAPIPE_AVAILABLE:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=confidence_threshold
            )
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=confidence_threshold,
                min_tracking_confidence=0.5,
            )
        else:
            import cv2
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        self._pose_estimator = HeadPoseEstimator()

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a video frame and extract vision features."""
        if frame is None:
            return {
                "face_count": 0, "face_present": False,
                "confidence": 0.0, "landmarks": None, "head_pose": None,
            }

        if MEDIAPIPE_AVAILABLE:
            return self._process_mediapipe(frame)
        return self._process_opencv_fallback(frame)

    def _process_mediapipe(self, frame: np.ndarray) -> dict:
        """Process using MediaPipe."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        det_results = self._face_detection.process(rgb)
        face_count = len(det_results.detections) if det_results.detections else 0
        confidence = max((d.score[0] for d in det_results.detections), default=0.0)

        mesh_results = self._face_mesh.process(rgb)
        landmarks = None
        head_pose = None

        if mesh_results.multi_face_landmarks:
            landmarks = self._extract_key_landmarks(mesh_results.multi_face_landmarks[0])
            raw_pose = self._estimate_head_pose(mesh_results.multi_face_landmarks[0])
            head_pose = self._pose_estimator.process(raw_pose)

        result = {
            "face_count": face_count,
            "face_present": face_count > 0,
            "confidence": confidence,
            "landmarks": landmarks,
            "head_pose": head_pose,
        }
        self.face_count_history.append(face_count)
        return result

    def _extract_key_landmarks(self, face_landmarks) -> list:
        """Extract key facial landmark points (normalized)."""
        key_indices = [1, 33, 263, 61, 291, 199]
        return [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                for i in key_indices]

    def _estimate_head_pose(self, face_landmarks) -> dict:
        """Estimate head pose angles from facial landmarks."""
        nose = face_landmarks.landmark[1]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        eye_dx = right_eye.x - left_eye.x
        yaw = math.atan2(nose.y - (left_eye.y + right_eye.y) / 2, eye_dx) * 180 / math.pi if eye_dx != 0 else 0
        pitch = math.atan2(nose.y - face_landmarks.landmark[199].y, 0.1) * 180 / math.pi
        roll = math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x) * 180 / math.pi

        return {"pitch": pitch, "yaw": yaw, "roll": roll}

    def _process_opencv_fallback(self, frame: np.ndarray) -> dict:
        """Fallback using OpenCV Haar cascades."""
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5)
        result = {
            "face_count": len(faces),
            "face_present": len(faces) > 0,
            "confidence": 1.0 if len(faces) > 0 else 0.0,
            "landmarks": None, "head_pose": None,
        }
        self.face_count_history.append(len(faces))
        return result

    def get_face_presence_ratio(self, window: int = 100) -> float:
        """Get ratio of frames with faces present in recent window."""
        if not self.face_count_history:
            return 1.0
        recent = self.face_count_history[-window:]
        return sum(1 for c in recent if c > 0) / len(recent)

    def get_multi_face_events(self, window: int = 100) -> int:
        """Count frames with multiple faces in recent window."""
        recent = self.face_count_history[-window:]
        return sum(1 for c in recent if c > 1)
