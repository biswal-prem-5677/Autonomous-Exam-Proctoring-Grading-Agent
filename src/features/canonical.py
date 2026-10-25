"""Canonical 22-dimensional feature schema for the Autonomous Exam Proctoring Platform.

Guarantees 100% semantic identity across:
- Training data generation (TrainingDataGenerator)
- Live sensor feature extraction (BehavioralFeatureExtractor)
- Risk engine feature vector evaluation (RiskEngine)
- Model inference & evaluation (LogisticRegressionScratch, AnomalyDetector)
- Explainability & telemetry visualization
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# ─── Authoritative 22-Dimensional Feature Names ──────────────────────────────

CANONICAL_FEATURE_NAMES: List[str] = [
    "face_present",          # 0: binary 0.0 or 1.0
    "face_confidence",       # 1: float [0, 1]
    "face_count",            # 2: float, normalized [0, 1]
    "absence_ratio",         # 3: float [0, 1]
    "head_yaw",              # 4: float, normalized [-1, 1] (yaw / 90.0)
    "head_pitch",            # 5: float, normalized [-1, 1] (pitch / 90.0)
    "head_roll",             # 6: float, normalized [-1, 1] (roll / 90.0)
    "audio_rms",             # 7: float [0, 1]
    "audio_db",              # 8: float, normalized [0, 1] ((db + 100) / 100)
    "audio_zcr",             # 9: float [0, 1]
    "audio_threshold_flag",  # 10: binary 0.0 or 1.0
    "spectral_centroid",     # 11: float, normalized [0, 1] (centroid / 8000)
    "keyboard_idle_flag",    # 12: binary 0.0 or 1.0
    "keystroke_rate",        # 13: float, normalized [0, 1] (epm / 200)
    "key_hold_mean",         # 14: float, normalized [0, 1] (sec / 0.5)
    "key_latency_mean",      # 15: float, normalized [0, 1] (sec / 1.0)
    "mouse_speed",           # 16: float, normalized [0, 1] (px/s / 500)
    "mouse_distance",        # 17: float, normalized [0, 1] (px / 2000)
    "click_rate",            # 18: float, normalized [0, 1] (clicks / 50)
    "mouse_idle_flag",       # 19: binary 0.0 or 1.0
    "multi_face_events",     # 20: float, normalized [0, 1] (events / 5)
    "session_progress",      # 21: float [0, 1] (elapsed / duration)
]

CANONICAL_INDICES: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_FEATURE_NAMES)}
N_CANONICAL_FEATURES: int = len(CANONICAL_FEATURE_NAMES)
N_FEATURES: int = N_CANONICAL_FEATURES

# ─── Feature Index Constants ────────────────────────────────────────────────

IX_FACE_PRESENT: int = 0
IX_FACE_CONF: int = 1
IX_FACE_COUNT: int = 2
IX_ABSENCE_RATIO: int = 3
IX_HEAD_YAW: int = 4
IX_HEAD_PITCH: int = 5
IX_HEAD_ROLL: int = 6
IX_AUDIO_RMS: int = 7
IX_AUDIO_DB: int = 8
IX_AUDIO_ZCR: int = 9
IX_AUDIO_FLAG: int = 10
IX_SPECTRAL: int = 11
IX_KB_IDLE: int = 12
IX_KB_RATE: int = 13
IX_KEY_HOLD: int = 14
IX_KEY_LAT: int = 15
IX_MOUSE_SPD: int = 16
IX_MOUSE_DIST: int = 17
IX_CLICK_RATE: int = 18
IX_MOUSE_IDLE: int = 19
IX_MULTI_FACE: int = 20
IX_PROGRESS: int = 21

# ─── Normalization Bounds & Baseline Defaults ───────────────────────────────

FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "face_present": (0.0, 1.0),
    "face_confidence": (0.0, 1.0),
    "face_count": (0.0, 1.0),
    "absence_ratio": (0.0, 1.0),
    "head_yaw": (-1.0, 1.0),
    "head_pitch": (-1.0, 1.0),
    "head_roll": (-1.0, 1.0),
    "audio_rms": (0.0, 1.0),
    "audio_db": (0.0, 1.0),
    "audio_zcr": (0.0, 1.0),
    "audio_threshold_flag": (0.0, 1.0),
    "spectral_centroid": (0.0, 1.0),
    "keyboard_idle_flag": (0.0, 1.0),
    "keystroke_rate": (0.0, 1.0),
    "key_hold_mean": (0.0, 1.0),
    "key_latency_mean": (0.0, 1.0),
    "mouse_speed": (0.0, 1.0),
    "mouse_distance": (0.0, 1.0),
    "click_rate": (0.0, 1.0),
    "mouse_idle_flag": (0.0, 1.0),
    "multi_face_events": (0.0, 1.0),
    "session_progress": (0.0, 1.0),
}

FEATURE_DEFAULTS: Dict[str, float] = {
    "face_present": 1.0,
    "face_confidence": 0.95,
    "face_count": 0.25,      # 1 face normalized (1/4)
    "absence_ratio": 0.0,
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    "head_roll": 0.0,
    "audio_rms": 0.05,
    "audio_db": 0.40,
    "audio_zcr": 0.05,
    "audio_threshold_flag": 0.0,
    "spectral_centroid": 0.25,
    "keyboard_idle_flag": 0.0,
    "keystroke_rate": 0.30,
    "key_hold_mean": 0.24,
    "key_latency_mean": 0.10,
    "mouse_speed": 0.20,
    "mouse_distance": 0.25,
    "click_rate": 0.10,
    "mouse_idle_flag": 0.0,
    "multi_face_events": 0.0,
    "session_progress": 0.0,
}


# ─── Authoritative Extraction & Conversion Functions ─────────────────────────

def extract_canonical_features(signals: Dict[str, Any],
                               history: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Extract authoritative 22-dimensional feature vector from multimodal signals.

    This is the single canonical implementation used by live sensors, risk engine,
    and inference pipelines.

    Args:
        signals: dict with sensor observations (face, head_pose, audio, keyboard, mouse, etc.)
        history: optional dict with historical statistics (e.g. face_absence_ratio)

    Returns:
        np.ndarray of shape (22,) with dtype float64.
    """
    history = history or {}
    vec = np.zeros(N_CANONICAL_FEATURES, dtype=np.float64)

    # 0: face_present — binary [0, 1]
    is_present = bool(signals.get("face_present", False))
    vec[IX_FACE_PRESENT] = 1.0 if is_present else 0.0

    # 1: face_confidence — float [0, 1]
    raw_conf = signals.get("face_confidence", 0.95 if is_present else 0.0)
    vec[IX_FACE_CONF] = float(np.clip(float(raw_conf), 0.0, 1.0))

    # 2: face_count — normalized [0, 1] (capped at 4)
    raw_count = signals.get("face_count", 1.0 if is_present else 0.0)
    try:
        val = float(raw_count)
        # normalize: 1 face = 0.25 or 1.0 depending on cap. Normalized to [0, 1] via val / 4.0
        # If already in [0, 1], preserve; if > 1, scale by 4.0
        norm_count = val / 4.0 if val > 1.0 else (val if val <= 1.0 and is_present else 0.0)
        vec[IX_FACE_COUNT] = float(np.clip(norm_count, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_FACE_COUNT] = 0.25 if is_present else 0.0

    # 3: absence_ratio — float [0, 1]
    absence_ratio = history.get("face_absence_ratio", signals.get("absence_ratio", 0.0))
    vec[IX_ABSENCE_RATIO] = float(np.clip(float(absence_ratio), 0.0, 1.0))

    # 4, 5, 6: Head pose normalized [-1, 1]
    head_pose = signals.get("head_pose", {}) or {}
    yaw = head_pose.get("yaw", signals.get("head_yaw", 0.0))
    pitch = head_pose.get("pitch", signals.get("head_pitch", 0.0))
    roll = head_pose.get("roll", signals.get("head_roll", 0.0))

    # Normalize degrees / 90.0 if raw degree scale (> 1.0 or < -1.0)
    def _norm_angle(a: Any) -> float:
        try:
            f = float(a)
            if abs(f) > 1.0:
                f = f / 90.0
            return float(np.clip(f, -1.0, 1.0))
        except (ValueError, TypeError):
            return 0.0

    vec[IX_HEAD_YAW] = _norm_angle(yaw)
    vec[IX_HEAD_PITCH] = _norm_angle(pitch)
    vec[IX_HEAD_ROLL] = _norm_angle(roll)

    # Audio features
    audio = signals.get("audio_features", {}) or {}
    rms = audio.get("rms_energy", signals.get("audio_rms", 0.0))
    vec[IX_AUDIO_RMS] = float(np.clip(float(rms), 0.0, 1.0))

    raw_db = audio.get("volume_db", signals.get("audio_db", -100.0))
    # If already normalized [0, 1] vs raw decibel [-100, 0]
    try:
        f_db = float(raw_db)
        norm_db = (f_db + 100.0) / 100.0 if f_db < 0.0 else f_db
        vec[IX_AUDIO_DB] = float(np.clip(norm_db, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_AUDIO_DB] = 0.0

    zcr = audio.get("zero_crossing_rate", signals.get("audio_zcr", 0.0))
    vec[IX_AUDIO_ZCR] = float(np.clip(float(zcr), 0.0, 1.0))

    # 10: audio_threshold_flag — binary [0, 1]
    vec[IX_AUDIO_FLAG] = 1.0 if (vec[IX_AUDIO_DB] > 0.60 or signals.get("loud_audio", False) or signals.get("audio_threshold_flag", False)) else 0.0

    # 11: spectral_centroid — float [0, 1] (raw / 8000.0)
    spec = audio.get("spectral_centroid", signals.get("spectral_centroid", 0.0))
    try:
        f_spec = float(spec)
        norm_spec = f_spec / 8000.0 if f_spec > 1.0 else f_spec
        vec[IX_SPECTRAL] = float(np.clip(norm_spec, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_SPECTRAL] = 0.0

    # Keyboard features
    kb_idle_sec = signals.get("keyboard_idle_seconds", 0)
    vec[IX_KB_IDLE] = 1.0 if (float(kb_idle_sec) > 10.0 or signals.get("idle_keyboard", False) or signals.get("keyboard_idle_flag", False)) else 0.0

    kb_events_pm = signals.get("keyboard_events_per_minute", signals.get("typing_wpm", 0.0) * 5.0)
    try:
        f_pm = float(kb_events_pm)
        norm_pm = f_pm / 200.0 if f_pm > 1.0 else f_pm
        vec[IX_KB_RATE] = float(np.clip(norm_pm, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_KB_RATE] = 0.0

    raw_hold = signals.get("key_hold_mean", signals.get("mean_key_hold_ms", 120.0) / 1000.0)
    try:
        f_hold = float(raw_hold)
        norm_hold = f_hold / 0.5 if f_hold > 1.0 else f_hold
        vec[IX_KEY_HOLD] = float(np.clip(norm_hold, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_KEY_HOLD] = 0.24

    raw_lat = signals.get("key_latency_mean", signals.get("mean_latency_ms", 100.0) / 1000.0)
    try:
        f_lat = float(raw_lat)
        norm_lat = f_lat / 1.0 if f_lat > 1.0 else f_lat
        vec[IX_KEY_LAT] = float(np.clip(norm_lat, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_KEY_LAT] = 0.10

    # Mouse features
    mouse_spd = signals.get("mouse_avg_speed", signals.get("mouse_velocity", 0.0))
    try:
        f_spd = float(mouse_spd)
        norm_spd = f_spd / 500.0 if f_spd > 1.0 else f_spd
        vec[IX_MOUSE_SPD] = float(np.clip(norm_spd, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_MOUSE_SPD] = 0.0

    mouse_dist = signals.get("mouse_total_distance", 0.0)
    try:
        f_dist = float(mouse_dist)
        norm_dist = f_dist / 2000.0 if f_dist > 1.0 else f_dist
        vec[IX_MOUSE_DIST] = float(np.clip(norm_dist, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_MOUSE_DIST] = 0.0

    clicks = signals.get("mouse_click_count", signals.get("mouse_clicks", 0.0))
    try:
        f_clk = float(clicks)
        norm_clk = f_clk / 50.0 if f_clk > 1.0 else f_clk
        vec[IX_CLICK_RATE] = float(np.clip(norm_clk, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_CLICK_RATE] = 0.0

    mouse_idle_sec = signals.get("mouse_idle_seconds", 0)
    vec[IX_MOUSE_IDLE] = 1.0 if (float(mouse_idle_sec) > 10.0 or signals.get("idle_mouse", False) or signals.get("mouse_idle_flag", False)) else 0.0

    # Temporal & context features
    multi_faces = signals.get("multi_face_events", 0)
    try:
        f_mf = float(multi_faces)
        norm_mf = f_mf / 5.0 if f_mf > 1.0 else f_mf
        vec[IX_MULTI_FACE] = float(np.clip(norm_mf, 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_MULTI_FACE] = 0.0

    dur = max(1.0, float(signals.get("session_duration", 3600.0)))
    prog = signals.get("session_progress", signals.get("session_elapsed", 0.0) / dur)
    try:
        vec[IX_PROGRESS] = float(np.clip(float(prog), 0.0, 1.0))
    except (ValueError, TypeError):
        vec[IX_PROGRESS] = 0.0

    return vec


def dict_to_canonical_vector(d: Dict[str, float]) -> np.ndarray:
    """Convert a dictionary of named features into a canonical 22D array."""
    vec = np.zeros(N_CANONICAL_FEATURES, dtype=np.float64)
    for name, idx in CANONICAL_INDICES.items():
        if name in d:
            val = float(d[name])
        else:
            val = FEATURE_DEFAULTS.get(name, 0.0)
        low, high = FEATURE_BOUNDS.get(name, (0.0, 1.0))
        vec[idx] = float(np.clip(val, low, high))
    return vec


def canonical_vector_to_dict(vector: np.ndarray) -> Dict[str, float]:
    """Convert a canonical 22D numpy array to a named dictionary."""
    if len(vector) != N_CANONICAL_FEATURES:
        raise ValueError(f"Expected {N_CANONICAL_FEATURES} features, got {len(vector)}")
    return {name: float(vector[i]) for i, name in enumerate(CANONICAL_FEATURE_NAMES)}


def validate_canonical_vector(vector: np.ndarray) -> Tuple[bool, str]:
    """Validate dimensionality and bounds of a canonical feature vector."""
    if not isinstance(vector, np.ndarray):
        return False, f"Expected np.ndarray, got {type(vector)}"
    if vector.shape != (N_CANONICAL_FEATURES,):
        return False, f"Expected shape ({N_CANONICAL_FEATURES},), got {vector.shape}"
    for name, idx in CANONICAL_INDICES.items():
        low, high = FEATURE_BOUNDS[name]
        val = vector[idx]
        if val < (low - 0.05) or val > (high + 0.05):
            return False, f"Feature '{name}' (idx {idx}) value {val} out of bounds [{low}, {high}]"
    return True, "Valid canonical feature vector"
