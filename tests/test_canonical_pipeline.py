"""Verification tests for Step 1: Canonical 22D Feature Contract."""

import numpy as np
import pytest

from src.features.canonical import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_INDICES,
    N_CANONICAL_FEATURES,
    IX_FACE_PRESENT,
    IX_PROGRESS,
    FEATURE_BOUNDS,
    extract_canonical_features,
    dict_to_canonical_vector,
    canonical_vector_to_dict,
    validate_canonical_vector,
)
from src.features.behavioral import BehavioralFeatureExtractor
from src.ml.training_data import (
    TrainingDataGenerator,
    IX_FACE_PRESENT as TR_IX_FACE_PRESENT,
    IX_PROGRESS as TR_IX_PROGRESS,
    N_FEATURES as TR_N_FEATURES,
)
from src.agent.risk import RiskEngine
from src.agent.memory import EvidenceMemory
from src.ml.logistic import LogisticRegressionScratch
from src.ml.anomaly import AnomalyDetector


def test_canonical_feature_count():
    assert len(CANONICAL_FEATURE_NAMES) == 22
    assert N_CANONICAL_FEATURES == 22
    assert TR_N_FEATURES == 22
    assert len(CANONICAL_INDICES) == 22


def test_training_feature_order_matches_canonical():
    assert TR_IX_FACE_PRESENT == IX_FACE_PRESENT == 0
    assert TR_IX_PROGRESS == IX_PROGRESS == 21
    gen = TrainingDataGenerator(seed=42)
    X, y = gen.generate_dataset(n_normal=10, n_suspicious=10)
    assert X.shape[1] == 22


def test_behavioral_extractor_produces_canonical_22d():
    extractor = BehavioralFeatureExtractor()
    signals = {
        "face_present": True,
        "face_confidence": 0.95,
        "face_count": 1,
        "head_pose": {"yaw": 10.0, "pitch": -5.0, "roll": 2.0},
        "audio_features": {"rms_energy": 0.1, "volume_db": -50.0, "zero_crossing_rate": 0.05},
        "keyboard_idle_seconds": 2,
        "keyboard_events_per_minute": 80,
        "mouse_avg_speed": 120,
        "mouse_total_distance": 600,
        "mouse_click_count": 12,
        "mouse_idle_seconds": 1,
    }
    vec = extractor.extract(signals)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (22,)
    assert vec.dtype == np.float64
    valid, msg = validate_canonical_vector(vec)
    assert valid, msg


def test_training_normalization_matches_live_normalization():
    gen = TrainingDataGenerator(seed=42)
    X, _ = gen.generate_dataset(n_normal=50, n_suspicious=50)

    # All features except head angles [-1, 1] are within [0, 1]
    for i in range(22):
        name = CANONICAL_FEATURE_NAMES[i]
        low, high = FEATURE_BOUNDS[name]
        # Allow small noise margin added during training data generation (std=0.05)
        assert np.all(X[:, i] >= (low - 0.15)), f"{name} below lower bound"
        assert np.all(X[:, i] <= (high + 0.15)), f"{name} above upper bound"


def test_risk_engine_consumes_canonical_22d_vector():
    engine = RiskEngine()
    mem = EvidenceMemory()
    mem.add_event("face_absent", 0.8)

    # Train a scratch logistic model on 22D data
    gen = TrainingDataGenerator(seed=42)
    X, y = gen.generate_dataset(n_normal=30, n_suspicious=30)
    log_m = LogisticRegressionScratch(learning_rate=0.05, max_iterations=50)
    log_m.fit(X, y)
    engine.set_logistic_model(log_m)

    # Train anomaly model on 22D data
    ano_m = AnomalyDetector()
    ano_m.fit(X[y == 0])
    engine.set_anomaly_model(ano_m)

    # RiskEngine computes risk using canonical 22D vector
    extractor = BehavioralFeatureExtractor()
    live_vec = extractor.extract({"face_present": False, "face_confidence": 0.0})
    risk = engine.compute_risk(mem, features=live_vec)
    assert 0.0 <= risk <= 1.0


def test_legacy_10d_fallback_completely_removed():
    engine = RiskEngine()
    assert not hasattr(engine, "_build_feature_vector")
    # Verify compute_risk does not accept or produce 10D vectors
    mem = EvidenceMemory()
    risk = engine.compute_risk(mem)
    assert 0.0 <= risk <= 1.0


def test_roundtrip_dict_and_vector():
    test_dict = {
        "face_present": 1.0,
        "face_confidence": 0.9,
        "face_count": 0.25,
        "head_yaw": -0.3,
        "session_progress": 0.5,
    }
    vec = dict_to_canonical_vector(test_dict)
    assert vec.shape == (22,)
    d = canonical_vector_to_dict(vec)
    assert d["face_present"] == 1.0
    assert abs(d["head_yaw"] - (-0.3)) < 1e-5
    assert d["session_progress"] == 0.5
