"""Facial landmark analysis — geometric features from detected face landmarks.

Implements the mathematical geometry described in the architecture:
  - Eye-nose vectors for head orientation
  - Eye aspect ratio for blink / fatigue detection
  - Face boundary analysis for distance estimation
"""

import math
from typing import Dict, List, Optional, Tuple


# Normalized landmark indices (MediaPipe face mesh)
L_LEFT = 33          # Left eye left corner
L_RIGHT = 133        # Left eye right corner
L_TOP = 159          # Left eye top
L_BOTTOM = 145       # Left eye bottom

R_LEFT = 362         # Right eye left corner
R_RIGHT = 263        # Right eye right corner
R_TOP = 386          # Right eye top
R_BOTTOM = 374       # Right eye bottom

NOSE = 1
CHIN = 152
FOREHEAD = 10
L_MOUTH = 61
R_MOUTH = 291

L_EAR = 234          # Left face boundary
R_EAR = 454          # Right face boundary


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _angle(p1: Tuple[float, float], p2: Tuple[float, float],
           p3: Tuple[float, float]) -> float:
    """Compute angle at p2 formed by p1-p2-p3, in degrees."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_angle))


class LandmarkAnalyzer:
    """Extracts geometric features from facial landmarks."""

    def __init__(self):
        self._ear_history: List[float] = []

    def analyze(self, landmarks: List[Tuple[float, float]]) -> Optional[Dict]:
        """Analyze facial landmarks and return geometric features.

        Args:
            landmarks: List of (x, y) normalized landmark coordinates.
                       Should be the full 468-point MediaPipe face mesh.

        Returns:
            Dict with eye aspect ratio, blink detection, head geometry, etc.
        """
        if not landmarks or len(landmarks) < max(R_EAR, NOSE, CHIN) + 1:
            return None

        features = {}

        # ── Eye Aspect Ratio (EAR) ──────────────────────────────────────
        left_ear = self._eye_aspect_ratio(landmarks, left=True)
        right_ear = self._eye_aspect_ratio(landmarks, left=False)
        avg_ear = (left_ear + right_ear) / 2.0
        features["left_ear"] = round(left_ear, 4)
        features["right_ear"] = round(right_ear, 4)
        features["avg_ear"] = round(avg_ear, 4)

        # Blink detection: EAR drops sharply
        self._ear_history.append(avg_ear)
        if len(self._ear_history) > 30:
            self._ear_history.pop(0)
        features["blink_detected"] = self._detect_blink(avg_ear)
        features["blink_rate"] = self._compute_blink_rate()

        # ── Face geometry ───────────────────────────────────────────────
        nose_pt = landmarks[NOSE]
        chin_pt = landmarks[CHIN]
        forehead_pt = landmarks[FOREHEAD]
        left_ear_pt = landmarks[L_EAR]
        right_ear_pt = landmarks[R_EAR]

        face_height = _dist(forehead_pt, chin_pt)
        face_width = _dist(left_ear_pt, right_ear_pt)
        features["face_width"] = round(face_width, 4)
        features["face_height"] = round(face_height, 4)
        features["face_aspect_ratio"] = round(face_height / max(face_width, 1e-8), 4)

        # Nose position relative to face center
        face_center_x = (left_ear_pt[0] + right_ear_pt[0]) / 2
        face_center_y = (forehead_pt[1] + chin_pt[1]) / 2
        h_offset = (nose_pt[0] - face_center_x) / max(face_width, 1e-8)
        v_offset = (nose_pt[1] - face_center_y) / max(face_height, 1e-8)
        features["gaze_horizontal"] = round(h_offset, 4)
        features["gaze_vertical"] = round(v_offset, 4)

        # Looking away thresholds (per architecture spec)
        features["looking_left"] = h_offset < -0.25
        features["looking_right"] = h_offset > 0.25
        features["looking_down"] = v_offset > 0.15
        features["looking_up"] = v_offset < -0.15
        features["looking_away"] = (
            abs(h_offset) > 0.25 or abs(v_offset) > 0.15
        )

        # ── Mouth analysis (fatigue / speaking) ─────────────────────────
        mouth_openness = _dist(landmarks[L_MOUTH], landmarks[R_MOUTH]) / max(face_width, 1e-8)
        features["mouth_openness"] = round(mouth_openness, 4)
        features["mouth_open"] = mouth_openness > 0.35

        # ── Head tilt ───────────────────────────────────────────────────
        eye_angle = _angle(landmarks[L_LEFT], landmarks[R_RIGHT], landmarks[L_RIGHT])
        features["eye_line_angle"] = round(eye_angle, 2)

        return features

    def _eye_aspect_ratio(self, landmarks: List[Tuple[float, float]],
                          left: bool = True) -> float:
        """Compute Eye Aspect Ratio (EAR) for blink detection.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        where p1-p6 are eye contour points.
        """
        if left:
            top, bottom = landmarks[L_TOP], landmarks[L_BOTTOM]
            left_c, right_c = landmarks[L_LEFT], landmarks[L_RIGHT]
        else:
            top, bottom = landmarks[R_TOP], landmarks[R_BOTTOM]
            left_c, right_c = landmarks[R_LEFT], landmarks[R_RIGHT]

        vertical1 = _dist(top, bottom)
        horizontal = _dist(left_c, right_c)

        if horizontal < 1e-8:
            return 0.0
        return (vertical1) / (2.0 * horizontal)

    def _detect_blink(self, current_ear: float, threshold: float = 0.2,
                      window: int = 3) -> bool:
        """Detect a blink from sudden EAR drop.

        A blink is detected when EAR drops below threshold after being above it,
        within a small temporal window.
        """
        if len(self._ear_history) < window + 1:
            return False

        recent = self._ear_history[-window:]
        before = recent[0] if recent[0] > threshold else None
        if before is None:
            return False

        return current_ear < threshold

    def _compute_blink_rate(self, window_seconds: float = 10.0) -> float:
        """Compute blink rate (blinks per minute) over recent history."""
        if len(self._ear_history) < 2:
            return 0.0
        # Assume ~30fps; window of 30 frames ≈ 1 second at 30fps
        # Adjust based on actual frame count
        blinks = sum(
            1 for i in range(1, len(self._ear_history))
            if self._ear_history[i] < 0.2 and self._ear_history[i - 1] >= 0.2
        )
        frames_per_second = 30.0  # Assumed capture rate
        seconds = len(self._ear_history) / frames_per_second
        if seconds < 1e-6:
            return 0.0
        return blinks / seconds * 60.0  # blinks per minute

    def reset(self) -> None:
        """Clear history."""
        self._ear_history.clear()


def get_landmark_features(
    face_mesh_landmarks,
    head_pose: Optional[Dict] = None,
) -> Optional[Dict]:
    """Convenience function: extract all landmark features from a MediaPipe result.

    Args:
        face_mesh_landmarks: MediaPipe face mesh multi_face_landmarks entry
        head_pose: Optional head pose dict from HeadPoseEstimator

    Returns:
        Dict of geometric features, or None if landmarks unavailable.
    """
    if not face_mesh_landmarks:
        return None

    lm_points = [
        (lm.x, lm.y) for lm in face_mesh_landmarks.landmark
    ]
    analyzer = LandmarkAnalyzer()
    features = analyzer.analyze(lm_points)

    if features and head_pose:
        features["head_yaw"] = head_pose.get("yaw", 0)
        features["head_pitch"] = head_pose.get("pitch", 0)
        features["head_roll"] = head_pose.get("roll", 0)

    return features
