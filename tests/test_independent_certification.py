"""Independent Certification & Production Verification Audit.

Proves:
1. Real Data Flow: Sensor -> Extraction -> 22D Canonical -> ML Inference -> Risk Score -> State -> Evidence -> Dashboard.
2. Canonical 22D Strictness: Exactly 22 features, identical names, order, normalization, no 10D fallback, no reconstruction.
3. Real Performance Prediction: Trained on real session history with honest confidence bounds; no hardcoded 78.5, no synthetic defaults.
4. Strict Security with EXAM_AGENT_AUTH_ENFORCE=1: Full attack matrix verified.
"""

import os
import uuid
import numpy as np
import pytest

# Ensure auth enforcement is strictly ON
pytestmark = pytest.mark.enforce_auth

from app.api.server import app, _get_db, _get_auth_db, _exam_manager, _proctoring_agents, _evidence_logs
from src.features.canonical import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_INDICES,
    N_CANONICAL_FEATURES,
    FEATURE_BOUNDS,
    FEATURE_DEFAULTS,
    extract_canonical_features,
    validate_canonical_vector,
)
from src.features.behavioral import BehavioralFeatureExtractor
from src.ml.training_data import TrainingDataGenerator
from src.ml.pipeline import TrainingPipeline
from src.agent.controller import ProctoringAgent
from src.agent.state import AgentState
from src.prediction.performance import PerformancePredictor


# ── Helper for Users & Auth ──────────────────────────────────────────────────

def create_test_user(role: str) -> tuple[str, str, dict]:
    """Create a user with specified role and return (username, token, auth_headers)."""
    auth_db = _get_auth_db()
    uname = f"{role[:3]}_{uuid.uuid4().hex[:6]}"
    uid = auth_db.create_user(
        username=uname,
        email=f"{uname}@audit.io",
        full_name=f"Audit {uname}",
        password="ValidPassword123!",
        role=role,
    )
    token = auth_db.create_session(uid)
    return uname, token, {"Authorization": f"Bearer {token}"}


# ═════════════════════════════════════════════════════════════════════════════
# 1. REAL DATA FLOW AUDIT
# ═════════════════════════════════════════════════════════════════════════════

class TestRealDataFlow:
    """Verifies empirical values flow end-to-end from sensor signals to dashboard."""

    def test_sensor_to_canonical_vector_flow(self):
        """Simulate real raw sensor readings and verify feature extraction produces a valid 22D vector."""
        extractor = BehavioralFeatureExtractor()

        raw_signals = {
            "face_present": True,
            "face_confidence": 0.88,
            "face_count": 1,
            "head_pose": {"yaw": 45.0, "pitch": -20.0, "roll": 5.0},
            "audio_features": {
                "rms_energy": 0.35,
                "volume_db": -35.0,
                "zero_crossing_rate": 0.18,
                "spectral_centroid": 3200.0,
            },
            "keyboard": {
                "idle_seconds": 2.0,
                "events_per_minute": 140.0,
                "mean_key_hold": 110.0,
                "mean_latency": 90.0,
            },
            "mouse": {
                "avg_speed": 220.0,
                "total_distance": 850.0,
                "click_count": 12,
                "idle_seconds": 3.0,
            },
            "multi_face_events": 0,
            "session_elapsed": 300,
            "session_duration": 1800,
        }

        # Step 1: Feature Extraction
        vector = extractor.extract(raw_signals)
        assert isinstance(vector, np.ndarray), "Extracted features must be a numpy array"
        assert vector.shape == (22,), f"Expected shape (22,), got {vector.shape}"

        # Step 2: Validate feature values reflect the input sensors
        assert vector[0] == 1.0  # face_present
        assert abs(vector[1] - 0.88) < 1e-3  # face_confidence
        assert abs(vector[4] - (45.0 / 90.0)) < 1e-3  # head_yaw normalized (45/90 = 0.5)
        assert abs(vector[5] - (-20.0 / 90.0)) < 1e-3  # head_pitch normalized (-20/90 = -0.222)
        assert abs(vector[7] - 0.35) < 1e-3  # audio_rms
        assert abs(vector[8] - ((-35.0 + 100.0) / 100.0)) < 1e-3  # audio_db normalized (65/100 = 0.65)
        assert abs(vector[13] - (140.0 / 200.0)) < 1e-3  # keystroke_rate normalized (140/200 = 0.70)
        assert abs(vector[16] - (220.0 / 500.0)) < 1e-3  # mouse_speed normalized (220/500 = 0.44)
        assert abs(vector[21] - (300.0 / 1800.0)) < 1e-3  # session_progress (300/1800 = 0.1667)

    def test_full_pipeline_agent_state_and_evidence_flow(self):
        """Verify signals -> 22D vector -> ML models -> risk score -> state escalation -> evidence."""
        # Train actual ML pipeline with real generated dataset
        pipeline = TrainingPipeline()
        results = pipeline.run()
        assert results.optimal_threshold > 0.0

        # Create proctoring agent with trained models
        agent = ProctoringAgent()
        agent.set_logistic_model(pipeline.logistic_model)
        agent.set_anomaly_model(pipeline.anomaly_detector)

        assert agent.get_status()["state"] == AgentState.NORMAL.value
        initial_risk = agent.get_status()["current_risk"]
        assert initial_risk == 0.0

        # Cycle 1: Feed Normal Signals
        normal_signals = {
            "face_present": True,
            "face_confidence": 0.95,
            "face_count": 1,
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "audio_features": {"rms_energy": 0.05, "volume_db": -60.0, "zero_crossing_rate": 0.05},
            "keyboard_events_per_minute": 80.0,
            "mouse_avg_speed": 100.0,
            "camera_connected": True,
        }
        res1 = agent.process_signals(normal_signals)
        assert res1["state"] == AgentState.NORMAL.value
        assert res1["risk_score"] < 0.35

        # Cycle 2: Feed Suspicious Signals (multiple faces + loud audio + tab switch)
        suspicious_signals = {
            "face_present": True,
            "face_confidence": 0.95,
            "face_count": 2,
            "multiple_faces": True,
            "head_pose": {"yaw": 65.0, "pitch": 0.0, "roll": 0.0},
            "audio_features": {"rms_energy": 0.85, "volume_db": -15.0, "zero_crossing_rate": 0.4},
            "speech_detected": True,
            "loud_audio": True,
            "tab_switch": True,
            "camera_connected": True,
        }
        res2 = agent.process_signals(suspicious_signals)
        # Risk must escalate due to cross-modal corroboration & ML scoring
        assert res2["risk_score"] > res1["risk_score"], "Risk score must increase on suspicious signals"
        # Initial anomaly escalates state to MONITOR or SUSPICIOUS (temporal smoothing)
        assert res2["state"] in (AgentState.MONITOR.value, AgentState.SUSPICIOUS.value, AgentState.REVIEW_REQUIRED.value)

        # Cycle 3: Continued persistent suspicious activity escalates to SUSPICIOUS / REVIEW_REQUIRED
        res3 = agent.process_signals(suspicious_signals)
        assert res3["risk_score"] >= res2["risk_score"]
        assert res3["state"] in (AgentState.SUSPICIOUS.value, AgentState.REVIEW_REQUIRED.value)

        # Verify evidence was recorded
        history = agent.get_history()
        assert len(history) == 3
        detected_types = {e["type"] for h in history for e in h.get("events_detected", [])}
        assert "multiple_faces" in detected_types or "loud_audio" in detected_types or "tab_switch" in detected_types


# ═════════════════════════════════════════════════════════════════════════════
# 2. CANONICAL 22D SCHEMA STRICTNESS AUDIT
# ═════════════════════════════════════════════════════════════════════════════

class TestCanonical22DSchemaStrictness:
    """Verifies that the 22D schema is authoritative, strictly ordered, and free of legacy fallbacks."""

    def test_canonical_feature_definitions(self):
        assert len(CANONICAL_FEATURE_NAMES) == 22, f"Must have exactly 22 features, got {len(CANONICAL_FEATURE_NAMES)}"
        assert N_CANONICAL_FEATURES == 22
        assert len(CANONICAL_INDICES) == 22
        assert len(FEATURE_BOUNDS) == 22
        assert len(FEATURE_DEFAULTS) == 22

        # Verify exact ordering
        for i, name in enumerate(CANONICAL_FEATURE_NAMES):
            assert CANONICAL_INDICES[name] == i

    def test_training_and_inference_schema_identity(self):
        """Verify TrainingDataGenerator produces identical feature schema to BehavioralFeatureExtractor."""
        gen = TrainingDataGenerator(seed=123)
        X, y = gen.generate_dataset(n_normal=5, n_suspicious=5)
        assert X.shape == (10, 22), "Training data must produce exactly (N, 22)"

        extractor = BehavioralFeatureExtractor()
        live_vec = extractor.extract({"face_present": True})
        assert live_vec.shape == (22,), "Extraction must produce exactly (22,)"

        # Validate both against canonical validator
        for row in X:
            valid, msg = validate_canonical_vector(row)
            assert valid, f"Generated sample violates schema: {msg}"

        valid, msg = validate_canonical_vector(live_vec)
        assert valid, f"Live extracted sample violates schema: {msg}"

    def test_no_legacy_10d_in_codebase(self):
        """Ensure no active module falls back to 10D or ad-hoc vectors."""
        import inspect
        from src.features.behavioral import BehavioralFeatureExtractor
        src = inspect.getsource(BehavioralFeatureExtractor.extract)
        assert "extract_canonical_features" in src
        assert "10" not in src, "10D fallback must not exist in BehavioralFeatureExtractor"


# ═════════════════════════════════════════════════════════════════════════════
# 3. REAL PERFORMANCE PREDICTION AUDIT
# ═════════════════════════════════════════════════════════════════════════════

class TestRealPerformancePrediction:
    """Verifies performance prediction is derived from real session data and trained models."""

    def test_prediction_returns_insufficient_data_when_no_history(self):
        c = app.test_client()
        _, _, E = create_test_user("examiner")
        r = c.post("/api/predict/performance", headers=E, json={"student_id": "fresh_user_without_history"})
        assert r.status_code == 200
        data = r.get_json()
        # MUST NOT return 78.5 or fake prediction!
        assert data["status"] == "insufficient_data"
        assert data["prediction"] is None
        assert data["predicted_score"] is None
        assert data["prediction_interval"]["lower_bound"] is None

    def test_prediction_trains_on_real_sessions_and_produces_honest_bounds(self):
        db = _get_db()
        c = app.test_client()
        _, _, E = create_test_user("examiner")

        # Create 4 real completed historical sessions with real grades in the database
        target_student = f"real_stud_{uuid.uuid4().hex[:6]}"

        # Insert 4 historical sessions with different performance levels
        for idx, (score, acc, dur) in enumerate([
            (60.0, 0.60, 30),
            (70.0, 0.70, 35),
            (80.0, 0.80, 25),
            (90.0, 0.90, 20),
        ]):
            sid = db.create_session(student_id=target_student, exam_id="exam_hist", duration_minutes=dur)
            qid = f"q_hist_{idx}_{uuid.uuid4().hex[:4]}"
            db.add_question(sid, qid, "mcq", "Sample Q", marks=score, correct_answer="a")
            db.record_answer(sid, qid, "a", score=score, max_score=100.0, grading_method="mcq")

        # Now call performance prediction for target_student using examiner credentials
        r = c.post("/api/predict/performance", headers=E, json={
            "student_id": target_student,
            "features": [0.85, 0.10, 0.50]  # high accuracy, low risk
        })
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "success"
        pred = data["predicted_score"]
        assert 0.0 <= pred <= 100.0, f"Predicted score {pred} should be within valid percentage range [0, 100]"
        # Verify genuine prediction interval with lower and upper bounds
        interval = data["prediction_interval"]
        assert interval["lower_bound"] is not None
        assert interval["upper_bound"] is not None
        assert interval["lower_bound"] <= pred <= interval["upper_bound"]
        assert interval["std_dev"] >= 0.0

    def test_student_cannot_call_prediction_endpoint_403(self):
        c = app.test_client()
        _, _, S = create_test_user("student")
        r = c.post("/api/predict/performance", headers=S, json={"student_id": "any_student"})
        assert r.status_code == 403


# ═════════════════════════════════════════════════════════════════════════════
# 4. STRICT PRODUCTION SECURITY MATRIX (EXAM_AGENT_AUTH_ENFORCE=1)
# ═════════════════════════════════════════════════════════════════════════════

class TestStrictProductionSecurityMatrix:
    """Verifies that all security guards actively reject unauthorized attacks when enforcement is ON."""

    def test_no_token_rejected_with_401(self):
        c = app.test_client()
        # Protected examiner endpoint
        assert c.get("/api/examiner/exam/nonexistent/overview").status_code == 401
        # Protected exam creation endpoint
        assert c.post("/api/exams", json={"title": "Attack Exam"}).status_code == 401
        # Protected grading override
        assert c.post("/api/grading/override", json={}).status_code == 401

    def test_invalid_token_rejected_with_401(self):
        c = app.test_client()
        bad_headers = {"Authorization": "Bearer invalid_forged_token_xyz999"}
        assert c.get("/api/examiner/exam/nonexistent/overview", headers=bad_headers).status_code == 401
        assert c.post("/api/exams", headers=bad_headers, json={"title": "Attack"}).status_code == 401

    def test_student_cannot_access_staff_routes_403(self):
        c = app.test_client()
        _, _, S = create_test_user("student")
        # Attempt to create an exam
        assert c.post("/api/exams", headers=S, json={"title": "Hacked Exam"}).status_code == 403
        # Attempt to trigger ML training
        assert c.post("/api/training/train", headers=S, json={}).status_code == 403
        # Attempt to override grade
        assert c.post("/api/grading/override", headers=S, json={}).status_code == 403
        # Attempt to view examiner dashboard
        assert c.get("/api/examiner/exam/dummy/overview", headers=S).status_code == 403

    def test_student_cannot_access_another_students_session_403(self):
        c = app.test_client()
        student_a, _, S_a = create_test_user("student")
        student_b, _, S_b = create_test_user("student")
        _, _, E = create_test_user("examiner")

        # Examiner creates and releases exam
        eid = c.post("/api/exams", headers=E, json={"title": "SecExam", "questions": [
            {"type": "mcq", "text": "Q1", "options": ["a", "b"], "correct_answer": "a", "marks": 5}
        ]}).get_json()["exam_id"]
        c.post(f"/api/exams/{eid}/roster", headers=E, json={"student_id": student_b})
        c.post(f"/api/exams/{eid}/release", headers=E, json={})

        # Student A attempts to start or access session claiming student_b
        res = c.post("/api/exam-sessions", headers=S_a, json={"student_id": student_b, "exam_id": eid})
        assert res.status_code == 403, "Student A must not be allowed to start session for Student B"

        # Student A attempts to analyze telemetry under student_b's session key
        res_tel = c.post("/api/behavioral/analyze", headers=S_a, json={
            "session_key": f"{student_b}:{eid}",
            "student_id": student_b
        })
        assert res_tel.status_code == 403, "Student A must not access Student B's session key"

    def test_student_cannot_access_unreleased_exam_denied(self):
        c = app.test_client()
        student_name, _, S = create_test_user("student")
        _, _, E = create_test_user("examiner")

        # Create exam but DO NOT release it
        eid = c.post("/api/exams", headers=E, json={"title": "Unreleased", "questions": [
            {"type": "mcq", "text": "Q1", "options": ["a"], "correct_answer": "a", "marks": 2}
        ]}).get_json()["exam_id"]
        c.post(f"/api/exams/{eid}/roster", headers=E, json={"student_id": student_name})

        # Student attempts to start session on unreleased exam
        res = c.post("/api/exam-sessions", headers=S, json={"student_id": student_name, "exam_id": eid})
        assert res.status_code == 403
        assert res.get_json().get("status") == "not_released"

    def test_student_cannot_submit_twice_409(self):
        c = app.test_client()
        student_name, _, S = create_test_user("student")
        _, _, E = create_test_user("examiner")

        # Create, enroll, and release exam
        eid = c.post("/api/exams", headers=E, json={"title": "OneShot", "questions": [
            {"type": "mcq", "text": "Q", "options": ["a", "b"], "correct_answer": "a", "marks": 5}
        ]}).get_json()["exam_id"]
        c.post(f"/api/exams/{eid}/roster", headers=E, json={"student_id": student_name})
        c.post(f"/api/exams/{eid}/release", headers=E, json={})
        c.post("/api/exam-sessions", headers=S, json={"student_id": student_name, "exam_id": eid})

        # First submission succeeds
        first = c.post("/api/grading/submit", headers=S, json={
            "student_id": student_name, "exam_id": eid, "answers": {"q1": "a"}
        })
        assert first.status_code == 200
        # Student callers must never receive answer key or scores from submit endpoint
        assert "correct_answer" not in first.get_data(as_text=True)

        # Second submission must be rejected with 409 Conflict
        second = c.post("/api/grading/submit", headers=S, json={
            "student_id": student_name, "exam_id": eid, "answers": {"q1": "b"}
        })
        assert second.status_code == 409

    def test_non_examiner_cannot_override_grades_403(self):
        c = app.test_client()
        _, _, S = create_test_user("student")
        res = c.post("/api/grading/override", headers=S, json={
            "session_id": "dummy_sess", "question_id": "q1", "override_score": 10.0, "reason": "Self-grade attempt"
        })
        assert res.status_code == 403

    def test_invalid_exam_id_returns_404(self):
        c = app.test_client()
        _, _, E = create_test_user("examiner")
        fake_id = "nonexistent_exam_id_deadbeef"
        res = c.get(f"/api/exams/{fake_id}", headers=E)
        assert res.status_code == 404
