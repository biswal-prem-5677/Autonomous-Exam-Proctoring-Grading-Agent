"""Webcam sensor — captures frames and detects faces using MediaPipe."""

import time
import threading
import queue
from typing import Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class WebcamSensor:
    """Captures webcam frames and provides face detection."""

    def __init__(self, camera_index: int = 0, confidence_threshold: float = 0.7):
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self._capture: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._capture_thread: Optional[threading.Thread] = None

        if MEDIAPIPE_AVAILABLE:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=confidence_threshold
            )
        else:
            # Fallback to Haar cascades
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def start(self) -> bool:
        """Start the webcam capture."""
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            return False
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        return True

    def _capture_loop(self) -> None:
        """Background frame capture loop."""
        while self._running:
            ret, frame = self._capture.read()
            if ret:
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Drop frame if queue is full
            time.sleep(0.033)  # ~30fps

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in a frame. Returns list of (x, y, w, h, confidence)."""
        if frame is None:
            return []

        if MEDIAPIPE_AVAILABLE:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_detection.process(rgb_frame)
            faces = []
            if results.detections:
                h, w = frame.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    fw = int(bbox.width * w)
                    fh = int(bbox.height * h)
                    conf = detection.score[0]
                    faces.append((x, y, fw, fh, conf))
            return faces
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            return [(int(x), int(y), int(w), int(h), 1.0)
                    for (x, y, w, h) in detections]

    def stop(self) -> None:
        """Stop capture and release the camera."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._capture:
            self._capture.release()
            self._capture = None
