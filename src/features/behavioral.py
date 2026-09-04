"""Feature engineering — behavioral features from multimodal signals."""

import numpy as np
from typing import Dict, List, Optional


class BehavioralFeatureExtractor:
    """Extracts behavioral features from proctoring signals."""

    def __init__(self):
        self._face_presence_history: List[float] = []
        self._gaze_violation_history: List[float] = []
        self._audio_event_history: List[float] = []
        self._keyboard_events: List[float] = []
        self._mouse_events: List[float] = []

    def extract(self, signals: Dict) -> np.ndarray:
        """Extract a feature vector from all signal channels.

        Args:
            signals: dict with keys from different sensors:
                - face_count, face_confidence
                - head_pose (pitch, yaw, roll)
                - audio_features (rms, db, zcr)
                - keyboard (idle_seconds, events_per_minute)
                - mouse (avg_speed, total_distance, click_count)

        Returns:
            Feature vector as numpy array.
        """
        features = []

        # Face features (4)
        features.append(1.0 if signals.get("face_present", False) else 0.0)
        features.append(signals.get("face_confidence", 0.0))
        features.append(signals.get("face_count", 0) / 2.0)  # Normalize
        features.append(self._face_absence_ratio())

        # Head pose features (3)
        head_pose = signals.get("head_pose", {}) or {}
        features.append(abs(head_pose.get("yaw", 0)) / 90.0)
        features.append(abs(head_pose.get("pitch", 0)) / 90.0)
        features.append(abs(head_pose.get("roll", 0)) / 90.0)

        # Audio features (5)
        audio = signals.get("audio_features", {}) or {}
        features.append(audio.get("rms_energy", 0.0))
        features.append(min(audio.get("volume_db", -100), 0.0) / 100.0)
        features.append(audio.get("zero_crossing_rate", 0.0))
        features.append(1.0 if audio.get("volume_db", -100) > -40 else 0.0)
        features.append(audio.get("spectral_centroid", 0.0) / 8000.0)

        # Keyboard features (3)
        features.append(signals.get("keyboard_idle_seconds", 0) / 120.0)
        features.append(signals.get("keyboard_events_per_minute", 0) / 100.0)
        features.append(1.0 if signals.get("keyboard_idle_seconds", 0) > 120 else 0.0)

        # Mouse features (4)
        features.append(signals.get("mouse_avg_speed", 0) / 500.0)
        features.append(signals.get("mouse_total_distance", 0) / 1000.0)
        features.append(signals.get("mouse_click_count", 0) / 50.0)
        features.append(signals.get("mouse_idle_seconds", 0) / 120.0)

        # Multi-face events (1)
        features.append(signals.get("multi_face_events", 0) / 10.0)

        # Temporal features (2)
        features.append(signals.get("session_elapsed", 0) / 3600.0)
        features.append(signals.get("total_events", 0) / 100.0)

        return np.array(features, dtype=np.float64)

    def _face_absence_ratio(self) -> float:
        """Compute ratio of recent frames without a face."""
        if not self._face_presence_history:
            return 0.0
        recent = self._face_presence_history[-30:]
        return 1.0 - sum(recent) / len(recent)

    def update_history(self, signals: Dict) -> None:
        """Update internal history buffers with new signal data."""
        self._face_presence_history.append(1.0 if signals.get("face_present", False) else 0.0)
        self._face_presence_history = self._face_presence_history[-100:]

        gaze_rate = signals.get("gaze_violation_rate", 0.0)
        self._gaze_violation_history.append(gaze_rate)
        self._gaze_violation_history = self._gaze_violation_history[-100:]


class TemporalFeatureAggregator:
    """Aggregates features over time windows for temporal analysis."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._window: List[np.ndarray] = []

    def add(self, feature_vector: np.ndarray) -> None:
        """Add a feature vector to the rolling window."""
        self._window.append(feature_vector)
        if len(self._window) > self.window_size:
            self._window.pop(0)

    def get_window(self) -> np.ndarray:
        """Get the current window as a 2D array."""
        if not self._window:
            return np.zeros((0, len(self._window[0]) if self._window else 0))
        return np.stack(self._window)

    def aggregate(self) -> Dict:
        """Compute aggregated statistics over the window."""
        window = self.get_window()
        if window.shape[0] == 0:
            return {"mean": np.zeros(0), "std": np.zeros(0), "trend": np.zeros(0)}

        return {
            "mean": np.mean(window, axis=0),
            "std": np.std(window, axis=0),
            "min": np.min(window, axis=0),
            "max": np.max(window, axis=0),
            "trend": self._compute_trend(window),
        }

    def _compute_trend(self, window: np.ndarray) -> np.ndarray:
        """Compute linear trend (slope) for each feature."""
        if window.shape[0] < 3:
            return np.zeros(window.shape[1])
        t = np.arange(window.shape[0])
        slopes = []
        for i in range(window.shape[1]):
            x = t
            y = window[:, i]
            slope = np.polyfit(x, y, 1)[0] if len(set(y)) > 1 else 0.0
            slopes.append(slope)
        return np.array(slopes)
