"""Head pose estimator for gaze and head-turning detection."""

import math
from typing import Optional


class HeadPoseEstimator:
    """Estimates head pose and detects gaze violations."""

    def __init__(self):
        self._history: list = []
        self._gaze_violation_count = 0

    def process(self, head_pose: dict) -> dict:
        """Process head pose data and detect violations.

        Returns dict with normalized angles and violation flag.
        """
        if not head_pose:
            return {"pitch": 0, "yaw": 0, "roll": 0, "violation": False}

        pitch = max(-90, min(90, head_pose.get("pitch", 0)))
        yaw = max(-90, min(90, head_pose.get("yaw", 0)))
        roll = max(-90, min(90, head_pose.get("roll", 0)))

        violation = abs(yaw) > 45 or abs(roll) > 30
        result = {"pitch": pitch, "yaw": yaw, "roll": roll, "violation": violation}

        self._history.append(result)
        if violation:
            self._gaze_violation_count += 1
        return result

    def get_gaze_violation_rate(self, window: int = 50) -> float:
        """Get violation rate in recent window."""
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return sum(1 for r in recent if r["violation"]) / len(recent)
