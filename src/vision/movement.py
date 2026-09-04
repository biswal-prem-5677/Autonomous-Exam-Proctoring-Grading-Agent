"""Body movement detection from MediaPipe pose/holistic landmarks.

Detects:
- Person leaving seat (desk vs standing pose comparison)
- Excessive upper-body movement
- Repeated posture changes

All computation is local — no external APIs.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.vision.head_pose import HeadPoseEstimator


# MediaPipe holistic landmark indices
NOSE = 0
LEFT_EYE_INNER = 7
LEFT_EYE = 33
RIGHT_EYE_INNER = 262
RIGHT_EYE = 263
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

# Key point groups for posture analysis
TORSO_POINTS = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
UPPER_BODY_POINTS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST]


class MovementDetector:
    """Detects unusual body movement from pose landmarks."""

    def __init__(self, threshold_standing: float = 0.15,
                 threshold_excessive: float = 80.0,
                 history_window: int = 60):
        """Initialize movement detector.

        Args:
            threshold_standing: Hip-to-shoulder ratio above which person is "standing"
            threshold_excessive: Movement displacement (pixels) considered excessive per frame
            history_window: Number of recent frames to keep for temporal analysis
        """
        self.threshold_standing = threshold_standing
        self.threshold_excessive = threshold_excessive
        self.history_window = history_window

        self._torso_positions: List[np.ndarray] = []
        self._movement_displacements: List[float] = []
        self._posture_changes: List[str] = []
        self._current_posture = "unknown"
        self._pose_estimator = HeadPoseEstimator()

    def process_landmarks(self, landmarks: List[Tuple[float, float]]) -> Dict:
        """Process detected pose landmarks.

        Args:
            landmarks: List of (x, y) normalized landmark coordinates

        Returns:
            Dict with movement features and flags
        """
        if not landmarks or len(landmarks) < max(TORSO_POINTS) + 1:
            return {
                "movement_detected": False,
                "displacement": 0.0,
                "posture": "unknown",
                "posture_changed": False,
                "excessive_movement": False,
                "standing": False,
            }

        pts = np.array(landmarks)

        # Posture estimation (sitting vs standing)
        posture, standing = self._estimate_posture(pts)
        posture_changed = (posture != self._current_posture and self._current_posture != "unknown")
        self._current_posture = posture

        # Movement displacement
        torso_center = self._torso_center(pts)
        displacement = self._compute_displacement(torso_center)

        result = {
            "movement_detected": displacement > 5.0,
            "displacement": round(displacement, 2),
            "posture": posture,
            "posture_changed": posture_changed,
            "excessive_movement": displacement > self.threshold_excessive,
            "standing": standing,
            "torso_center": torso_center.tolist(),
        }

        # Update history
        self._torso_positions.append(torso_center)
        self._movement_displacements.append(displacement)
        if len(self._torso_positions) > self.history_window:
            self._torso_positions.pop(0)
        if len(self._movement_displacements) > self.history_window:
            self._movement_displacements.pop(0)

        return result

    def process_face_landmarks(self, face_landmarks) -> Dict:
        """Process MediaPipe face mesh landmarks for movement detection.

        Works with face mesh results when full pose is unavailable.

        Args:
            face_landmarks: MediaPipe face landmarks object

        Returns:
            Dict with head movement features
        """
        nose = face_landmarks.landmark[NOSE]
        left_shoulder_fallback = face_landmarks.landmark[10]
        right_shoulder_fallback = face_landmarks.landmark[340]

        # Use approximate head region for movement
        pts = np.array([
            (nose.x, nose.y),
            (left_shoulder_fallback.x, left_shoulder_fallback.y),
            (right_shoulder_fallback.x, right_shoulder_fallback.y),
        ])
        return self.process_landmarks(pts.tolist())

    def get_movement_features(self) -> np.ndarray:
        """Extract statistical features from movement history.

        Returns:
            Feature vector: [mean_disp, std_disp, max_disp, posture_changes,
                             standing_ratio, sudden_movement_count]
        """
        if not self._movement_displacements:
            return np.zeros(6, dtype=np.float64)

        displacements = np.array(self._movement_displacements)
        mean_disp = float(np.mean(displacements))
        std_disp = float(np.std(displacements))
        max_disp = float(np.max(displacements))

        # Posture changes
        changes = len(self._posture_changes)

        # Standing ratio
        standing_ratio = sum(1 for p in self._posture_changes if p == "standing") / max(len(self._posture_changes), 1)

        # Sudden movement count (displacement spikes > 2 std from mean)
        if std_disp > 0 and len(displacements) > 5:
            sudden = np.sum(np.abs(displacements - mean_disp) > 2 * std_disp)
            sudden_count = float(sudden)
        else:
            sudden_count = 0.0

        features = np.array([
            mean_disp, std_disp, max_disp, changes, standing_ratio, sudden_count
        ], dtype=np.float64)

        # Normalize roughly to [0, 1]
        features[0] = min(mean_disp / 100.0, 1.0)
        features[1] = min(std_disp / 50.0, 1.0)
        features[2] = min(max_disp / 200.0, 1.0)
        features[3] = min(changes / 10.0, 1.0)
        # standing_ratio already [0,1]
        features[5] = min(sudden_count / 10.0, 1.0)

        return features

    def reset(self) -> None:
        """Clear all history."""
        self._torso_positions.clear()
        self._movement_displacements.clear()
        self._posture_changes.clear()
        self._current_posture = "unknown"

    # ─── Private helpers ────────────────────────────────────────────────────

    def _estimate_posture(self, pts: np.ndarray) -> Tuple[str, bool]:
        """Estimate sitting vs standing from torso landmarks.

        Uses the ratio of hip-to-shoulder distance (vertical) vs
        shoulder-to-shoulder distance (horizontal). Standing increases
        the vertical torso ratio.
        """
        if pts.shape[0] < max(TORSO_POINTS) + 1:
            return "unknown", False

        ls = pts[LEFT_SHOULDER]
        rs = pts[RIGHT_SHOULDER]
        lh = pts[LEFT_HIP]
        rh = pts[RIGHT_HIP]

        shoulder_width = math.sqrt((rs[0] - ls[0]) ** 2 + (rs[1] - ls[1]) ** 2)
        if shoulder_width < 1e-6:
            return "unknown", False

        hip_y = (lh[1] + rh[1]) / 2
        shoulder_y = (ls[1] + rs[1]) / 2
        torso_height = shoulder_y - hip_y

        ratio = torso_height / shoulder_width
        standing = ratio > self.threshold_standing

        if len(self._posture_changes) > 0:
            current = self._posture_changes[-1]
        else:
            current = "unknown"

        new_posture = "standing" if standing else "sitting"
        if new_posture != current:
            self._posture_changes.append(new_posture)
            if len(self._posture_changes) > self.history_window:
                self._posture_changes.pop(0)

        return new_posture, standing

    def _torso_center(self, pts: np.ndarray) -> np.ndarray:
        """Compute the center of the torso from landmark points."""
        valid_pts = [pts[i] for i in TORSO_POINTS if i < len(pts)]
        if not valid_pts:
            return np.array([0.5, 0.5], dtype=np.float64)
        return np.mean(valid_pts, axis=0)

    def _compute_displacement(self, current_center: np.ndarray) -> float:
        """Compute movement displacement from last frame's torso position."""
        if not self._torso_positions:
            return 0.0
        last = self._torso_positions[-1]
        # Scale by image dimensions factor (normalized to pixel-like units)
        dx = (current_center[0] - last[0]) * 1000
        dy = (current_center[1] - last[1]) * 1000
        return math.sqrt(dx ** 2 + dy ** 2)


class ProctoringMovementAnalyzer:
    """High-level movement analysis for the proctoring system.

    Combines MovementDetector with the HeadPoseEstimator to produce
    a unified movement feature vector for the ML pipeline.
    """

    def __init__(self):
        self.movement_detector = MovementDetector()
        self._event_counts: Dict[str, int] = {
            "leaving_seat": 0,
            "excessive_movement": 0,
            "posture_change": 0,
        }

    def update(self, face_result: Dict, movement_result: Dict = None) -> Dict:
        """Update analysis with new frame data.

        Args:
            face_result: Dict from FaceTracker.process_frame()
            movement_result: Optional dict from MovementDetector.process_landmarks()

        Returns:
            Dict with movement features for ML pipeline
        """
        features = {
            "movement_detected": False,
            "standing": False,
            "posture_changed": False,
            "excessive_movement": False,
            "movement_displacement": 0.0,
            "leaving_seat": False,
        }

        if movement_result:
            features["movement_detected"] = movement_result.get("movement_detected", False)
            features["standing"] = movement_result.get("standing", False)
            features["posture_changed"] = movement_result.get("posture_changed", False)
            features["excessive_movement"] = movement_result.get("excessive_movement", False)
            features["movement_displacement"] = movement_result.get("displacement", 0.0)

            if movement_result.get("standing"):
                features["leaving_seat"] = True
                self._event_counts["leaving_seat"] += 1

            if movement_result.get("excessive_movement"):
                self._event_counts["excessive_movement"] += 1

            if movement_result.get("posture_changed"):
                self._event_counts["posture_change"] += 1

        # Also check head pose from face result
        head_pose = (face_result or {}).get("head_pose") or {}
        if head_pose:
            features["head_yaw"] = head_pose.get("yaw", 0)
            features["head_pitch"] = head_pose.get("pitch", 0)
            features["head_roll"] = head_pose.get("roll", 0)
            features["head_violation"] = head_pose.get("violation", False)

        # Multi-face check
        face_count = (face_result or {}).get("face_count", 0)
        features["face_count"] = face_count
        features["multiple_faces"] = face_count > 1

        return features

    def get_feature_vector(self) -> np.ndarray:
        """Get 6-dim feature vector from movement history for ML pipeline."""
        mv_features = self.movement_detector.get_movement_features()

        # Pad to exactly 6 features if needed
        while len(mv_features) < 6:
            mv_features = np.append(mv_features, 0.0)

        # Add event counts (normalized)
        event_features = np.array([
            min(self._event_counts["leaving_seat"] / 5.0, 1.0),
            min(self._event_counts["excessive_movement"] / 10.0, 1.0),
        ], dtype=np.float64)

        return np.concatenate([mv_features, event_features])

    def get_event_summary(self) -> Dict[str, int]:
        """Get summary of all movement events."""
        return dict(self._event_counts)

    def reset(self) -> None:
        """Reset all state."""
        self.movement_detector.reset()
        self._event_counts = {
            "leaving_seat": 0,
            "excessive_movement": 0,
            "posture_change": 0,
        }
