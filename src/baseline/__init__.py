"""Personal behavioral baseline — per-student behavioral fingerprint.

Don't only compare students vs population.
Compare each student vs themselves.

This gives a much stronger mathematical story for anomaly detection:

Student baseline:  typing 46 WPM, mouse 210 px/s, face presence 98.4%
Later behavior:    typing 92 WPM, mouse 650 px/s, face presence 71%

The anomaly detector asks: Is this unusual for THIS student?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BehavioralSample:
    """A single behavioral observation."""
    timestamp: float
    typing_wpm: float = 0.0
    mouse_velocity: float = 0.0
    mouse_clicks: int = 0
    face_present: float = 1.0  # 0.0–1.0
    face_count: int = 1
    gaze_off_screen_seconds: float = 0.0
    pause_duration_seconds: float = 0.0


@dataclass
class StudentBaseline:
    """Personal behavioral baseline for a student.

    Tracks typical behavior using exponential moving statistics,
    enabling per-student anomaly detection.
    """
    student_id: str

    # Typing profile (WPM)
    typing_wpm_mean: float = 0.0
    typing_wpm_std: float = 10.0

    # Mouse profile (pixels per second)
    mouse_velocity_mean: float = 0.0
    mouse_velocity_std: float = 50.0
    mouse_clicks_per_minute: float = 0.0

    # Vision profile
    face_presence_mean: float = 0.98
    face_presence_std: float = 0.03
    gaze_off_screen_mean: float = 0.5
    gaze_off_screen_std: float = 1.0

    # Interaction profile
    avg_pause_seconds: float = 3.0
    pause_std: float = 2.0

    # Samples accumulated
    samples: int = 0
    # Raw samples for re-estimation
    _typing_samples: List[float] = field(default_factory=list)
    _mouse_samples: List[float] = field(default_factory=list)
    _face_samples: List[float] = field(default_factory=list)
    _gaze_samples: List[float] = field(default_factory=list)

    # Bounds for raw sample storage
    _max_raw_samples: int = 200

    def is_ready(self) -> bool:
        """Whether the baseline has enough samples to be reliable."""
        return self.samples >= 5

    def add_sample(self, sample: BehavioralSample) -> None:
        """Add a behavioral observation and update the baseline."""
        self.samples += 1
        alpha = 1.0 / min(self.samples, 100)  # diminishing update rate

        # Update means (exponential moving average)
        if sample.typing_wpm > 0:
            self.typing_wpm_mean = (1 - alpha) * self.typing_wpm_mean + alpha * sample.typing_wpm
            self._typing_samples.append(sample.typing_wpm)
            if len(self._typing_samples) > self._max_raw_samples:
                self._typing_samples.pop(0)
            if len(self._typing_samples) >= 3:
                self.typing_wpm_std = float(np.std(self._typing_samples))

        if sample.mouse_velocity > 0:
            self.mouse_velocity_mean = ((1 - alpha) * self.mouse_velocity_mean +
                                         alpha * sample.mouse_velocity)
            self._mouse_samples.append(sample.mouse_velocity)
            if len(self._mouse_samples) > self._max_raw_samples:
                self._mouse_samples.pop(0)
            if len(self._mouse_samples) >= 3:
                self.mouse_velocity_std = float(np.std(self._mouse_samples))

        if sample.face_present >= 0:
            self.face_presence_mean = ((1 - alpha) * self.face_presence_mean +
                                        alpha * sample.face_present)
            self._face_samples.append(sample.face_present)
            if len(self._face_samples) > self._max_raw_samples:
                self._face_samples.pop(0)
            if len(self._face_samples) >= 3:
                self.face_presence_std = float(np.std(self._face_samples))

        if sample.gaze_off_screen_seconds >= 0:
            self.gaze_off_screen_mean = ((1 - alpha) * self.gaze_off_screen_mean +
                                          alpha * sample.gaze_off_screen_seconds)
            self._gaze_samples.append(sample.gaze_off_screen_seconds)
            if len(self._gaze_samples) > self._max_raw_samples:
                self._gaze_samples.pop(0)
            if len(self._gaze_samples) >= 3:
                self.gaze_off_screen_std = float(np.std(self._gaze_samples))

    def deviation_typing(self, current_wpm: float) -> Tuple[bool, float]:
        """Returns (is_anomalous, deviation_score 0-1)."""
        if not self.is_ready():
            return False, 0.0
        if self.typing_wpm_std < 0.5:
            return False, 0.0
        z = abs(current_wpm - self.typing_wpm_mean) / max(self.typing_wpm_std, 0.5)
        deviation = min(z / 3.0, 1.0)
        return bool(z > 2.5), deviation

    def deviation_face_presence(self, current_rate: float) -> Tuple[bool, float]:
        """Returns (is_anomalous, deviation_score 0-1)."""
        if not self.is_ready():
            return False, 0.0
        if self.face_presence_std < 0.001:
            return False, 0.0
        z = (self.face_presence_mean - current_rate) / max(self.face_presence_std, 0.001)
        deviation = max(0.0, min(z / 3.0, 1.0))
        return bool(z > 1.5), deviation

    def deviation_mouse(self, current_velocity: float) -> Tuple[bool, float]:
        """Returns (is_anomalous, deviation_score 0-1)."""
        if not self.is_ready():
            return False, 0.0
        if self.mouse_velocity_std < 0.5:
            return False, 0.0
        z = abs(current_velocity - self.mouse_velocity_mean) / max(self.mouse_velocity_std, 0.5)
        deviation = min(z / 3.0, 1.0)
        return bool(z > 2.5), deviation

    def deviation_gaze(self, current_off_screen: float) -> Tuple[bool, float]:
        """Returns (is_anomalous, deviation_score 0-1)."""
        if not self.is_ready():
            return False, 0.0
        if self.gaze_off_screen_std < 0.05:
            return False, 0.0
        z = (current_off_screen - self.gaze_off_screen_mean) / max(self.gaze_off_screen_std, 0.05)
        deviation = max(0.0, min(z / 3.0, 1.0))
        return bool(z > 2.5), deviation

    def compute_all_deviations(self, signals: Dict) -> Dict[str, Tuple[float, bool]]:
        """Compute all baseline deviations from a signal dict.

        Args:
            signals: Dict with keys like typing_wpm, mouse_velocity,
                     face_present_rate, gaze_off_screen.

        Returns:
            Dict of {metric: (deviation_0_1, is_anomalous)}.
        """
        return {
            "typing_wpm": self.deviation_typing(signals.get("typing_wpm", 0)),
            "face_presence": self.deviation_face_presence(signals.get("face_present_rate", 1.0)),
            "mouse_velocity": self.deviation_mouse(signals.get("mouse_velocity", 0)),
            "gaze_off_screen": self.deviation_gaze(signals.get("gaze_off_screen", 0)),
        }

    def max_deviation(self, signals: Dict) -> float:
        """Return the maximum deviation across all metrics."""
        deviations = self.compute_all_deviations(signals)
        return max((d[0] for d in deviations.values()), default=0.0)

    def to_dict(self) -> Dict:
        return {
            "student_id": self.student_id,
            "samples": self.samples,
            "ready": self.is_ready(),
            "typing_wpm": {
                "mean": round(self.typing_wpm_mean, 1),
                "std": round(self.typing_wpm_std, 1),
            },
            "mouse_velocity": {
                "mean": round(self.mouse_velocity_mean, 1),
                "std": round(self.mouse_velocity_std, 1),
            },
            "face_presence": {
                "mean": round(self.face_presence_mean, 4),
                "std": round(self.face_presence_std, 4),
            },
            "gaze_off_screen": {
                "mean": round(self.gaze_off_screen_mean, 2),
                "std": round(self.gaze_off_screen_std, 2),
            },
        }


class BaselineManager:
    """Manages per-student baselines across sessions."""

    def __init__(self):
        self._baselines: Dict[str, StudentBaseline] = {}

    def get_or_create(self, student_id: str) -> StudentBaseline:
        """Get existing baseline or create a new one."""
        if student_id not in self._baselines:
            self._baselines[student_id] = StudentBaseline(student_id=student_id)
        return self._baselines[student_id]

    def record_observation(self, student_id: str, sample: BehavioralSample) -> StudentBaseline:
        """Record a behavioral observation and return updated baseline."""
        baseline = self.get_or_create(student_id)
        baseline.add_sample(sample)
        return baseline

    def get(self, student_id: str) -> Optional[StudentBaseline]:
        """Get baseline for a student, or None if not yet built."""
        return self._baselines.get(student_id)

    def all_baselines(self) -> Dict[str, StudentBaseline]:
        """Return all stored baselines."""
        return dict(self._baselines)

    def remove(self, student_id: str) -> None:
        """Remove a student's baseline."""
        self._baselines.pop(student_id, None)
