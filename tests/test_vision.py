"""Tests for vision landmark and movement detection modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _landmark_list(pts_dict):
    """Convert {index: (x,y)} dict to a list indexed by position."""
    max_idx = max(pts_dict.keys())
    result = [(0.0, 0.0)] * (max_idx + 1)
    for idx, pt in pts_dict.items():
        result[idx] = pt
    return result


# ─── Pre-built explicit landmark fixtures ────────────────────────────────────

# Symmetric face, centered gaze (nose on face center axis)
_CENTERED_FACE = {
    1: (0.50, 0.47),     # nose
    7: (0.46, 0.47),     # left eye inner
    33: (0.44, 0.47),    # left eye left corner
    133:(0.48, 0.47),    # left eye right corner
    159:(0.44, 0.46),    # left eye top
    145:(0.44, 0.48),    # left eye bottom
    262:(0.52, 0.47),    # right eye inner
    263:(0.56, 0.47),    # right eye right corner  (note: index 263 is RIGHT eye RIGHT)
    362:(0.52, 0.47),    # right eye left corner
    386:(0.56, 0.46),    # right eye top
    374:(0.56, 0.48),    # right eye bottom
    10: (0.50, 0.35),    # forehead
    152:(0.50, 0.62),    # chin
    61: (0.46, 0.52),    # left mouth
    291:(0.54, 0.52),    # right mouth
    234:(0.38, 0.47),    # left ear (face edge)
    454:(0.62, 0.47),    # right ear (face edge)
    11: (0.42, 0.58),    # left shoulder
    12: (0.58, 0.58),    # right shoulder
    13: (0.40, 0.70),    # left elbow
    14: (0.60, 0.70),    # right elbow
    15: (0.40, 0.85),    # left wrist
    16: (0.60, 0.85),    # right wrist
    23: (0.43, 0.78),    # left hip
    24: (0.57, 0.78),    # right hip
}

# Same face but nose shifted LEFT → looking left
# h_offset = (0.38 - 0.50) / 0.24 = -0.50 → abs > 0.25 → looking_away
_LOOKING_LEFT = dict(_CENTERED_FACE)
_LOOKING_LEFT[1] = (0.38, 0.47)   # nose shifted left of face center (0.50)

# Same face but nose shifted RIGHT → looking right
# h_offset = (0.63 - 0.50) / 0.24 = 0.54 → abs > 0.25 → looking_away
_LOOKING_RIGHT = dict(_CENTERED_FACE)
_LOOKING_RIGHT[1] = (0.63, 0.47)  # nose shifted right of face center (0.50)

# Standing pose: large vertical torso (shoulders high, hips low)
# shoulder_width=0.2, torso_height=0.35, ratio=1.75 > 0.15 → standing
_STANDING_POSE = {
    1:  (0.50, 0.47),
    10: (0.50, 0.35),
    152:(0.50, 0.62),
    11: (0.42, 0.45),    # left shoulder — high up
    12: (0.58, 0.45),    # right shoulder
    23: (0.43, 0.82),    # left hip — far below shoulders
    24: (0.57, 0.82),    # right hip
    234:(0.38, 0.47),
    454:(0.62, 0.47),
    33: (0.44, 0.47), 263:(0.56, 0.47),
    159:(0.44, 0.46), 145:(0.44, 0.48),
    386:(0.56, 0.46), 374:(0.56, 0.48),
    61: (0.46, 0.52), 291:(0.54, 0.52),
}


# ─── LandmarkAnalyzer tests ──────────────────────────────────────────────────

class TestLandmarkAnalyzer:
    def test_basic_analysis(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(_landmark_list(_CENTERED_FACE))
        assert result is not None
        assert "avg_ear" in result
        assert "face_aspect_ratio" in result

    def test_center_face_not_looking_away(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(_landmark_list(_CENTERED_FACE))
        assert abs(result["gaze_horizontal"]) < 0.1
        assert not result["looking_away"]

    def test_looking_left(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(_landmark_list(_LOOKING_LEFT))
        # h_offset = (0.38 - 0.50) / 0.24 = -0.50 → looking_left = True
        assert result["looking_left"]
        assert result["looking_away"]

    def test_looking_right(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(_landmark_list(_LOOKING_RIGHT))
        # h_offset = (0.63 - 0.50) / 0.24 = 0.54 → looking_right = True
        assert result["looking_right"]
        assert result["looking_away"]

    def test_empty_landmarks_returns_none(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        assert analyzer.analyze([]) is None
        assert analyzer.analyze(None) is None

    def test_ear_computed(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(_landmark_list(_CENTERED_FACE))
        assert 0 < result["avg_ear"] < 1.0

    def test_mouth_openness(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(_landmark_list(_CENTERED_FACE))
        assert "mouth_openness" in result

    def test_blink_rate_starts_zero(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(_landmark_list(_CENTERED_FACE))
        assert result["blink_rate"] == 0.0

    def test_reset_clears_history(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        for _ in range(5):
            analyzer.analyze(_landmark_list(_CENTERED_FACE))
        assert len(analyzer._ear_history) > 0
        analyzer.reset()
        assert len(analyzer._ear_history) == 0

    def test_face_aspect_ratio(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        result = analyzer.analyze(_landmark_list(_CENTERED_FACE))
        # face_height=0.27, face_width=0.24 → ratio ≈ 1.125
        assert result["face_aspect_ratio"] > 0.5

    def test_too_few_landmarks_returns_none(self):
        from src.vision.landmarks import LandmarkAnalyzer
        analyzer = LandmarkAnalyzer()
        assert analyzer.analyze([(0.5, 0.5)] * 10) is None


# ─── MovementDetector tests ──────────────────────────────────────────────────

class TestMovementDetector:
    def test_empty_landmarks(self):
        from src.vision.movement import MovementDetector
        det = MovementDetector()
        result = det.process_landmarks([])
        assert result["movement_detected"] is False
        assert result["posture"] == "unknown"

    def test_sitting_posture(self):
        from src.vision.movement import MovementDetector
        det = MovementDetector()
        result = det.process_landmarks(_landmark_list(_CENTERED_FACE))
        assert result["posture"] == "sitting"
        assert result["standing"] == False

    def test_standing_posture(self):
        from src.vision.movement import MovementDetector
        det = MovementDetector()
        # shoulder_width=0.16, torso_height=0.37, ratio=2.31 > 2.0 → standing
        result = det.process_landmarks(_landmark_list(_STANDING_POSE))
        assert result["standing"] == True
        assert result["posture"] == "standing"

    def test_displacement_computed(self):
        from src.vision.movement import MovementDetector
        det = MovementDetector()
        lm1 = _landmark_list(_CENTERED_FACE)
        det.process_landmarks(lm1)

        # Shift all landmarks slightly
        lm2 = [(x + 0.01, y + 0.01) for x, y in lm1]
        result = det.process_landmarks(lm2)
        assert result["displacement"] > 0

    def test_feature_vector_shape(self):
        from src.vision.movement import ProctoringMovementAnalyzer
        analyzer = ProctoringMovementAnalyzer()
        analyzer.movement_detector.process_landmarks(_landmark_list(_CENTERED_FACE))
        fv = analyzer.get_feature_vector()
        assert fv.shape[0] == 8  # 6 movement + 2 event counts

    def test_reset_clears_state(self):
        from src.vision.movement import MovementDetector
        det = MovementDetector()
        for _ in range(10):
            det.process_landmarks(_landmark_list(_CENTERED_FACE))
        assert len(det._torso_positions) > 0
        det.reset()
        assert len(det._torso_positions) == 0
        assert len(det._movement_displacements) == 0

    def test_insufficient_landmarks(self):
        from src.vision.movement import MovementDetector
        det = MovementDetector()
        result = det.process_landmarks([(0.5, 0.5)] * 5)
        assert result["posture"] == "unknown"

    def test_proctoring_analyzer_integration(self):
        from src.vision.movement import ProctoringMovementAnalyzer
        analyzer = ProctoringMovementAnalyzer()
        face_result = {
            "face_count": 1,
            "head_pose": {"yaw": 10, "pitch": 5, "roll": 2, "violation": False},
        }
        movement_result = analyzer.movement_detector.process_landmarks(
            _landmark_list(_CENTERED_FACE)
        )
        features = analyzer.update(face_result, movement_result)
        assert features["face_count"] == 1
        assert features["multiple_faces"] is False
        assert "movement_detected" in features

    def test_event_summary(self):
        from src.vision.movement import ProctoringMovementAnalyzer
        analyzer = ProctoringMovementAnalyzer()
        # _STANDING_POSE has ratio > 0.15 → standing → leaving_seat incremented
        movement_result = analyzer.movement_detector.process_landmarks(
            _landmark_list(_STANDING_POSE)
        )
        analyzer.update({}, movement_result)
        summary = analyzer.get_event_summary()
        assert summary["leaving_seat"] >= 1

    def test_analyzer_reset(self):
        from src.vision.movement import ProctoringMovementAnalyzer
        analyzer = ProctoringMovementAnalyzer()
        movement_result = analyzer.movement_detector.process_landmarks(
            _landmark_list(_STANDING_POSE)
        )
        analyzer.update({}, movement_result)
        analyzer.reset()
        assert analyzer.get_event_summary()["leaving_seat"] == 0
