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

        # 0: face_present — binary [0, 1]
        features.append(1.0 if signals.get("face_present", False) else 0.0)

        # 1: face_confidence — float [0, 1]
        features.append(float(np.clip(signals.get("face_confidence", 0.0), 0.0, 1.0)))

        # 2: face_count — int/float [0, 4]
        features.append(float(np.clip(signals.get("face_count", 1.0 if signals.get("face_present", False) else 0.0), 0.0, 4.0)))

        # 3: absence_ratio — float [0, 1]
        features.append(float(np.clip(self._face_absence_ratio(), 0.0, 1.0)))

        # 4, 5, 6: Head pose angles normalized [-1, 1]
        head_pose = signals.get("head_pose", {}) or {}
        features.append(float(np.clip(head_pose.get("yaw", 0.0) / 90.0, -1.0, 1.0)))
        features.append(float(np.clip(head_pose.get("pitch", 0.0) / 90.0, -1.0, 1.0)))
        features.append(float(np.clip(head_pose.get("roll", 0.0) / 90.0, -1.0, 1.0)))

        # 7: audio_rms — float [0, 1]
        audio = signals.get("audio_features", {}) or {}
        features.append(float(np.clip(audio.get("rms_energy", 0.0), 0.0, 1.0)))

        # 8: audio_db — float [0, 1]
        raw_db = audio.get("volume_db", -100.0)
        features.append(float(np.clip((raw_db + 100.0) / 100.0, 0.0, 1.0)))

        # 9: audio_zcr — float [0, 1]
        features.append(float(np.clip(audio.get("zero_crossing_rate", 0.0), 0.0, 1.0)))

        # 10: audio_threshold_flag — binary [0, 1]
        features.append(1.0 if raw_db > -40.0 or signals.get("loud_audio", False) else 0.0)

        # 11: spectral_centroid — float [0, 1]
        features.append(float(np.clip(audio.get("spectral_centroid", 0.0) / 8000.0, 0.0, 1.0)))

        # 12: keyboard_idle_flag — binary [0, 1]
        kb_idle_sec = signals.get("keyboard_idle_seconds", 0)
        features.append(1.0 if kb_idle_sec > 10.0 or signals.get("idle_keyboard", False) else 0.0)

        # 13: keystroke_rate — float [0, 1]
        kb_events_pm = signals.get("keyboard_events_per_minute", signals.get("typing_wpm", 0.0) * 5.0)
        features.append(float(np.clip(kb_events_pm / 200.0, 0.0, 1.0)))

        # 14: key_hold_mean — float [0, 1] (seconds)
        raw_hold = signals.get("key_hold_mean", signals.get("mean_key_hold_ms", 120.0) / 1000.0)
        features.append(float(np.clip(raw_hold / 0.5, 0.0, 1.0)))

        # 15: key_latency_mean — float [0, 1] (seconds)
        raw_lat = signals.get("key_latency_mean", signals.get("mean_latency_ms", 100.0) / 1000.0)
        features.append(float(np.clip(raw_lat / 1.0, 0.0, 1.0)))

        # 16: mouse_speed — float [0, 1]
        mouse_spd = signals.get("mouse_avg_speed", signals.get("mouse_velocity", 0.0))
        features.append(float(np.clip(mouse_spd / 500.0, 0.0, 1.0)))

        # 17: mouse_distance — float [0, 1]
        mouse_dist = signals.get("mouse_total_distance", 0.0)
        features.append(float(np.clip(mouse_dist / 2000.0, 0.0, 1.0)))

        # 18: click_rate — float [0, 1]
        clicks = signals.get("mouse_click_count", signals.get("mouse_clicks", 0.0))
        features.append(float(np.clip(clicks / 50.0, 0.0, 1.0)))

        # 19: mouse_idle_flag — binary [0, 1]
        mouse_idle_sec = signals.get("mouse_idle_seconds", 0)
        features.append(1.0 if mouse_idle_sec > 10.0 or signals.get("idle_mouse", False) else 0.0)

        # 20: multi_face_events — float [0, 1]
        features.append(float(np.clip(signals.get("multi_face_events", 0) / 5.0, 0.0, 1.0)))

        # 21: session_progress — float [0, 1]
        dur = max(1.0, float(signals.get("session_duration", 3600.0)))
        prog = signals.get("session_progress", signals.get("session_elapsed", 0.0) / dur)
        features.append(float(np.clip(prog, 0.0, 1.0)))

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
