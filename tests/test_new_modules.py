"""Tests for keystroke dynamics, mouse dynamics, browser monitoring,
training data generation, and ML pipeline."""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ════════════════════════════════════════════════════════════════════════════
# KEYSTROKE DYNAMICS
# ════════════════════════════════════════════════════════════════════════════

class TestKeystrokeDynamics:
    def test_record_and_extract(self):
        from src.features.keystroke import KeystrokeDynamics
        kd = KeystrokeDynamics()
        kd.record_key_press("a")
        time.sleep(0.01)
        kd.record_key_release("a")
        features = kd.extract_features()
        assert "hold_time_mean" in features
        assert "latency_mean" in features
        assert "burst_rate" in features
        assert "typing_rate" in features
        assert "consistency" in features

    def test_feature_vector_shape(self):
        from src.features.keystroke import KeystrokeDynamics
        kd = KeystrokeDynamics()
        for ch in "hello world":
            kd.record_key_press(ch)
            time.sleep(0.005)
            kd.record_key_release(ch)
        vec = kd.extract_feature_vector()
        assert len(vec) == 8

    def test_empty_returns_defaults(self):
        from src.features.keystroke import KeystrokeDynamics
        kd = KeystrokeDynamics()
        features = kd.extract_features()
        assert features["n_events"] == 0
        assert features["idle_flag"] == 1.0

    def test_build_profile(self):
        from src.features.keystroke import KeystrokeDynamics, KeystrokeEvent
        kd = KeystrokeDynamics()
        events = []
        t0 = time.time()
        for i, ch in enumerate("abcdefghij"):
            press = t0 + i * 0.05
            release = press + 0.08
            events.append(KeystrokeEvent(key=ch, press_time=press, release_time=release))
        profile = kd.build_profile(events)
        assert profile.n_samples == 10
        assert profile.hold_time_mean > 0

    def test_anomaly_detection_without_profile(self):
        from src.features.keystroke import KeystrokeDynamics
        kd = KeystrokeDynamics()
        is_anom, score = kd.is_anomalous()
        assert not is_anom
        assert score == 0.0

    def test_anomaly_detection_with_profile(self):
        from src.features.keystroke import KeystrokeDynamics, KeystrokeEvent
        kd = KeystrokeDynamics()
        events = []
        t0 = time.time()
        for i in range(20):
            press = t0 + i * 0.08
            release = press + 0.10
            events.append(KeystrokeEvent(key="a", press_time=press, release_time=release))
        kd.build_profile(events)
        for ch in "test typing":
            kd.record_key_press(ch)
            time.sleep(0.06)
            kd.record_key_release(ch)
        is_anom, score = kd.is_anomalous(threshold=2.5)
        assert isinstance(is_anom, bool)
        assert score >= 0.0

    def test_reset_clears_state(self):
        from src.features.keystroke import KeystrokeDynamics
        kd = KeystrokeDynamics()
        kd.record_key_press("a")
        time.sleep(0.01)
        kd.record_key_release("a")
        kd.reset()
        features = kd.extract_features()
        assert features["n_events"] == 0


# ════════════════════════════════════════════════════════════════════════════
# MOUSE DYNAMICS
# ════════════════════════════════════════════════════════════════════════════

class TestMouseDynamics:
    def test_record_move(self):
        from src.features.mouse import MouseDynamics
        md = MouseDynamics()
        md.record_move(0, 0)
        md.record_move(100, 100)
        metrics = md.extract_metrics()
        assert metrics.total_distance > 0

    def test_record_click(self):
        from src.features.mouse import MouseDynamics
        md = MouseDynamics()
        md.record_move(0, 0)
        md.record_move(50, 50)
        md.record_click(100, 200, button="left")
        md.record_click(200, 300, button="right")
        metrics = md.extract_metrics()
        assert metrics.click_count == 2
        assert metrics.right_click_count == 1

    def test_feature_vector_shape(self):
        from src.features.mouse import MouseDynamics
        md = MouseDynamics()
        for i in range(10):
            md.record_move(i * 10, i * 5)
        md.record_click(50, 25, "left")
        vec = md.extract_feature_vector()
        assert len(vec) == 20

    def test_movement_classification(self):
        from src.features.mouse import MouseDynamics, MovementType
        md = MouseDynamics()
        metrics = md.extract_metrics()
        assert metrics.movement_type == MovementType.IDLE

    def test_scroll_recorded(self):
        from src.features.mouse import MouseDynamics
        md = MouseDynamics()
        for i in range(20):
            md.record_move(i * 10, i * 5)
            if i % 5 == 0:
                md.record_scroll(i * 10, i * 5)
        metrics = md.extract_metrics()
        assert metrics.scroll_count == 4

    def test_reset(self):
        from src.features.mouse import MouseDynamics
        md = MouseDynamics()
        md.record_move(100, 200)
        md.reset()
        metrics = md.extract_metrics()
        assert metrics.total_distance == 0.0

    def test_insufficient_events(self):
        from src.features.mouse import MouseDynamics
        md = MouseDynamics()
        md.record_move(100, 200)
        metrics = md.extract_metrics()
        assert metrics.total_distance == 0.0


# ════════════════════════════════════════════════════════════════════════════
# BROWSER MONITORING
# ════════════════════════════════════════════════════════════════════════════

class TestBrowserMonitoring:
    def test_event_buffer(self):
        from src.sensors.browser import EventBuffer, BrowserEvent, BrowserEventType
        buf = EventBuffer(max_events=10)
        assert len(buf) == 0
        event = BrowserEvent(timestamp=time.time(), event_type=BrowserEventType.COPY, severity=0.4)
        buf.add(event)
        assert len(buf) == 1
        events = buf.get_all()
        assert len(events) == 1

    def test_circular_buffer_overflow(self):
        from src.sensors.browser import EventBuffer, BrowserEvent, BrowserEventType
        buf = EventBuffer(max_events=3)
        for i in range(5):
            buf.add(BrowserEvent(timestamp=time.time() + i, event_type=BrowserEventType.COPY, severity=0.4))
        assert len(buf) == 3

    def test_window_filter(self):
        from src.sensors.browser import EventBuffer, BrowserEvent, BrowserEventType
        buf = EventBuffer(max_events=100)
        now = time.time()
        buf.add(BrowserEvent(timestamp=now - 5, event_type=BrowserEventType.COPY, severity=0.4))
        buf.add(BrowserEvent(timestamp=now - 20, event_type=BrowserEventType.COPY, severity=0.4))
        recent = buf.get_events_in_window(10)
        assert len(recent) == 1

    def test_risk_engine_compute(self):
        from src.sensors.browser import BrowserRiskEngine, BrowserEventType
        engine = BrowserRiskEngine(window_seconds=10)
        for i in range(5):
            engine.record_event(BrowserEventType.TAB_FOCUS_LOST)
            time.sleep(0.01)
        risk = engine.compute_risk()
        assert 0.0 <= risk <= 1.0

    def test_risk_engine_no_events(self):
        from src.sensors.browser import BrowserRiskEngine
        engine = BrowserRiskEngine()
        risk = engine.compute_risk()
        assert risk == 0.0

    def test_escalation_devtools(self):
        from src.sensors.browser import BrowserRiskEngine, BrowserEventType
        engine = BrowserRiskEngine()
        engine.record_event(BrowserEventType.DEVTOOLS_OPEN)
        assert engine.should_escalate()

    def test_escalation_fullscreen(self):
        from src.sensors.browser import BrowserRiskEngine, BrowserEventType
        engine = BrowserRiskEngine()
        engine.record_event(BrowserEventType.FULLSCREEN_EXIT)
        assert engine.should_escalate()

    def test_js_injection_string(self):
        from src.sensors.browser import BROWSER_MONITOR_JS
        assert "visibilitychange" in BROWSER_MONITOR_JS
        assert "postMessage" in BROWSER_MONITOR_JS
        assert len(BROWSER_MONITOR_JS) > 100

    def test_risk_decay(self):
        from src.sensors.browser import BrowserRiskEngine, BrowserEventType
        engine = BrowserRiskEngine(window_seconds=10)
        engine.record_event(BrowserEventType.TAB_FOCUS_LOST)
        risk1 = engine.compute_risk()
        assert risk1 > 0
        risk2 = engine.compute_risk()
        assert risk2 <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATOR
# ════════════════════════════════════════════════════════════════════════════

class TestTrainingDataGenerator:
    def test_generate_dataset(self):
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=50, n_suspicious=50)
        assert X.shape == (100, 22)
        assert y.shape == (100,)
        assert set(np.unique(y)).issubset({0, 1})
        assert np.sum(y == 0) == 50
        assert np.sum(y == 1) == 50

    def test_generate_scenario(self):
        from src.ml.training_data import TrainingDataGenerator, BehaviorScenario
        gen = TrainingDataGenerator(seed=42)
        X = gen.generate_scenario_dataset(BehaviorScenario.LOOKING_AWAY, n_samples=20)
        assert X.shape == (20, 22)

    def test_generate_normal(self):
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X = gen.generate_normal_dataset(n_samples=30)
        assert X.shape == (30, 22)

    def test_train_test_split(self):
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=100, n_suspicious=100)
        X_tr, X_te, y_tr, y_te = gen.train_test_split(X, y, test_ratio=0.2)
        assert len(y_tr) + len(y_te) == 200
        assert len(y_te) == 40

    def test_cross_validation_folds(self):
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=100, n_suspicious=100)
        folds = gen.cross_validation_folds(X, y, n_folds=5)
        assert len(folds) == 5
        for X_tr, X_te, y_tr, y_te in folds:
            assert len(X_tr) + len(X_te) == 200

    def test_feature_ranges(self):
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=50, n_suspicious=50)
        assert np.all(X[:, :20] >= -0.05)
        assert np.all(X[:, :20] <= 1.05)

    def test_save_and_load(self, tmp_path):
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=20, n_suspicious=20)
        filepath = str(tmp_path / "dataset.json")
        gen.save_dataset(X, y, filepath)
        X2, y2 = gen.load_dataset(filepath)
        assert X2.shape == X.shape
        assert np.allclose(y, y2)

    def test_feature_statistics(self):
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=50, n_suspicious=50)
        stats = gen.compute_statistics(X, y)
        assert stats["n_samples"] == 100
        assert stats["n_features"] == 22
        assert "feature_means_normal" in stats
        assert "feature_means_suspicious" in stats

    def test_all_scenarios(self):
        from src.ml.training_data import TrainingDataGenerator, BehaviorScenario
        gen = TrainingDataGenerator(seed=42)
        for scenario in BehaviorScenario:
            X = gen.generate_scenario_dataset(scenario, n_samples=10)
            assert X.shape == (10, 22), f"Failed for {scenario}"


# ════════════════════════════════════════════════════════════════════════════
# ML PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class TestMLPipeline:
    def test_pipeline_runs(self):
        from src.ml.pipeline import TrainingPipeline, TrainingConfig
        config = TrainingConfig(n_normal=50, n_suspicious=50, logistic_max_iterations=100)
        pipeline = TrainingPipeline(config)
        results = pipeline.run()
        assert results is not None
        assert 0.0 <= results.optimal_threshold <= 1.0
        assert 0.0 <= results.logistic_metrics["accuracy"] <= 1.0

    def test_pipeline_predict(self):
        from src.ml.pipeline import TrainingPipeline, TrainingConfig
        config = TrainingConfig(n_normal=50, n_suspicious=50, logistic_max_iterations=100)
        pipeline = TrainingPipeline(config)
        pipeline.run()
        features = np.zeros(22)
        result = pipeline.predict_risk(features)
        assert "logistic_risk" in result
        assert "anomaly_score" in result
        assert "combined_risk" in result
        assert 0.0 <= result["combined_risk"] <= 1.0

    def test_save_and_load_models(self, tmp_path):
        from src.ml.pipeline import TrainingPipeline, TrainingConfig
        config = TrainingConfig(n_normal=50, n_suspicious=50, logistic_max_iterations=100)
        pipeline = TrainingPipeline(config)
        pipeline.run()
        model_dir = str(tmp_path / "models")
        pipeline.save_models(model_dir)
        pipeline2 = TrainingPipeline(config)
        pipeline2.load_models(model_dir)
        assert pipeline2.logistic_model is not None
        assert pipeline2.anomaly_detector is not None

    def test_feature_importance(self):
        from src.ml.pipeline import TrainingPipeline, TrainingConfig
        config = TrainingConfig(n_normal=50, n_suspicious=50, logistic_max_iterations=100)
        pipeline = TrainingPipeline(config)
        results = pipeline.run()
        importance = results.feature_importance
        assert isinstance(importance, dict)
        assert len(importance) == 22


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

class TestConvenienceFunctions:
    def test_generate_training_data(self):
        from src.ml.training_data import generate_training_data
        X, y = generate_training_data(n_normal=30, n_suspicious=30, seed=42)
        assert X.shape == (60, 22)
        assert y.shape == (60,)

    def test_generate_evaluation_data(self):
        from src.ml.training_data import generate_evaluation_data
        X, y = generate_evaluation_data(n_samples=40, seed=123)
        assert X.shape == (40, 22)
        assert y.shape == (40,)
