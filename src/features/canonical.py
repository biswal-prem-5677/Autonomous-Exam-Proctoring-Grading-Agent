"""Canonical 22-dimensional feature schema for the Autonomous Exam Proctoring Platform.

Guarantees 100% semantic identity across:
- Training data generation (TrainingDataGenerator)
- Live sensor feature extraction (BehavioralFeatureExtractor)
- Risk engine feature vector construction (RiskEngine)
- Model inference & evaluation (LogisticRegressionScratch, AnomalyDetector)
- Explainability & telemetry visualization
"""

from typing import List, Dict

CANONICAL_FEATURE_NAMES: List[str] = [
    "face_present",          # 0: binary 0.0 or 1.0
    "face_confidence",       # 1: float [0, 1]
    "face_count",            # 2: float, normalized [0, 1]
    "absence_ratio",         # 3: float [0, 1]
    "head_yaw",              # 4: float, normalized [-1, 1] or [0, 1]
    "head_pitch",            # 5: float, normalized [-1, 1] or [0, 1]
    "head_roll",             # 6: float, normalized [-1, 1] or [0, 1]
    "audio_rms",             # 7: float [0, 1]
    "audio_db",              # 8: float, normalized [0, 1]
    "audio_zcr",             # 9: float [0, 1]
    "audio_threshold_flag",  # 10: binary 0.0 or 1.0
    "spectral_centroid",     # 11: float, normalized [0, 1]
    "keyboard_idle_flag",    # 12: binary 0.0 or 1.0
    "keystroke_rate",        # 13: float, normalized [0, 1]
    "key_hold_mean",         # 14: float, normalized [0, 1]
    "key_latency_mean",      # 15: float, normalized [0, 1]
    "mouse_speed",           # 16: float, normalized [0, 1]
    "mouse_distance",        # 17: float, normalized [0, 1]
    "click_rate",            # 18: float, normalized [0, 1]
    "mouse_idle_flag",       # 19: binary 0.0 or 1.0
    "multi_face_events",     # 20: float, normalized [0, 1]
    "session_progress",      # 21: float [0, 1]
]

CANONICAL_INDICES: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_FEATURE_NAMES)}
N_CANONICAL_FEATURES: int = len(CANONICAL_FEATURE_NAMES)
