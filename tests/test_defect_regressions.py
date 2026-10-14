"""Comprehensive regression tests covering all 13 defect fixes, persistence, and API additions."""

import pytest
import json
import uuid
import numpy as np
from unittest.mock import MagicMock, patch

from src.vision.face import FaceTracker
from src.grading.tfidf import TFIDFScorer, KeywordCoverageScorer
from src.grading.scorer import GradingOrchestrator
from src.grading.long_answer import LongAnswerGrader
from src.grading.mathematical import MathematicalGrader
from src.grading.programming import ProgrammingGrader
from src.models.models import Exam, Question, QuestionType
from src.fusion import ModalitySource, CorroborationSignal, CrossModalCorroborationEngine
from src.trace.replay import DecisionReplayEngine, _safe_timestamp
from src.agent.controller import ProctoringAgent, AgentState
from src.database.db import Database
from src.database.auth import AuthDatabase, hash_password
from src.exam.exam import ExamManager
from app.api.server import app, _exam_manager, _proctoring_agents


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── Finding 1: MediaPipe fallback to Haar Cascade ────────────────────────────

def test_regression_finding_1_face_mediapipe_fallback():
    tracker = FaceTracker()
    assert tracker is not None
    assert hasattr(tracker, "_face_cascade") or hasattr(tracker, "_face_detection")
    assert hasattr(tracker, "use_mediapipe")
    blank_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    res = tracker.process_frame(blank_frame)
    assert res is not None
    assert "face_present" in res or "face_detected" in res or "faces" in res


# ── Finding 2: TFIDF empty keywords return 0.0 ───────────────────────────────

def test_regression_finding_2_tfidf_empty_keywords():
    scorer = KeywordCoverageScorer()
    res = scorer.score("Some student answer about biology", [])
    assert res["coverage"] == 0.0
    assert res["score"] == 0.0


# ── Finding 3 & 4: Scorer signatures ─────────────────────────────────────────

def test_regression_finding_3_grade_mathematical_signature():
    grader = MathematicalGrader()
    res = grader.grade(
        student_solution="x = 5",
        expected_formula="x = 5",
        expected_variables={"x": 5.0},
        expected_final_value=5.0,
        expected_units="meters",
        marks=10.0,
        tolerance=0.01,
    )
    assert res is not None
    assert hasattr(res, "score")
    assert res.score > 0

    orchestrator = GradingOrchestrator()
    assert hasattr(orchestrator, "grade_mathematical")


def test_regression_finding_4_grade_programming_signature():
    grader = ProgrammingGrader()
    res = grader.grade(
        code_str="def add(a, b): return a + b",
        entry_function_name="add",
        public_tests=[{"input": [2, 3], "expected": 5, "name": "test_add"}],
        marks=10.0,
    )
    assert res is not None
    assert hasattr(res, "score")
    assert res.score >= 9.0

    orchestrator = GradingOrchestrator()
    assert hasattr(orchestrator, "grade_programming")


# ── Finding 5 & 6: Exam models fields and question metadata ──────────────────

def test_regression_finding_5_and_6_exam_model_metadata():
    exam = Exam(
        id="exam_meta_test",
        title="Physics 101",
        instructions="No calculators allowed",
        passing_score=50.0,
        is_released=True,
        settings={"policy": "standard", "duration_minutes": 90},
    )
    assert exam.instructions == "No calculators allowed"
    assert exam.passing_score == 50.0
    assert exam.is_released is True
    assert exam.settings["duration_minutes"] == 90

    d = exam.to_dict()
    assert d["instructions"] == "No calculators allowed"
    assert d["passing_score"] == 50.0
    assert d["is_released"] is True

    q = exam.add_question(
        qtype="mcq",
        text="What is F?",
        options=["ma", "mv", "mg"],
        correct_answer="ma",
        marks=2.0,
        keywords=["force", "mass"],
        tolerance=0.05,
        rubric={"criteria": "formula match"},
        topic="Mechanics",
        difficulty="easy",
        metadata={"unit": "Newton"},
    )
    assert q.rubric == {"criteria": "formula match"}
    assert q.metadata["topic"] == "Mechanics"
    assert q.metadata["difficulty"] == "easy"
    assert q.metadata["unit"] == "Newton"
    qd = q.to_dict()
    assert qd["rubric"] == {"criteria": "formula match"}
    assert qd["metadata"]["unit"] == "Newton"


# ── Finding 7 & 8: Fusion & Trace Modality Enum handling ─────────────────────

def test_regression_finding_7_and_8_fusion_and_trace_modality_enum():
    engine = CrossModalCorroborationEngine()
    # Check both string and enum input
    sig1 = CorroborationSignal(
        modality=ModalitySource.VISION,
        event_type="face_absent",
        severity=0.8,
        confidence=0.8,
        timestamp=100.0,
    )
    sig2 = CorroborationSignal(
        modality="audio",
        event_type="speech_detected",
        severity=0.7,
        confidence=0.7,
        timestamp=100.5,
    )
    res = engine.evaluate([sig1, sig2])
    assert res is not None
    assert res.independent_modalities == 2

    # Verify string normalized to ModalitySource enum
    assert sig2.modality == ModalitySource.AUDIO
    assert sig1.modality == ModalitySource.VISION


# ── Finding 9: Trace replay safe timestamp ───────────────────────────────────

def test_regression_finding_9_replay_safe_timestamp():
    assert _safe_timestamp(12345.67) == 12345.67
    assert _safe_timestamp(12345) == 12345.0
    iso_val = _safe_timestamp("2026-10-14T10:00:00")
    assert isinstance(iso_val, float)
    assert iso_val > 0


# ── Finding 10, 11, 12: ProctoringAgent Controller fixes ─────────────────────

def test_regression_finding_10_11_12_agent_controller():
    agent = ProctoringAgent({"loop_interval_seconds": 1})
    agent._state = AgentState.MONITOR
    # Test transitioning out of monitor state
    agent._history.append({
        "timestamp": 1.0,
        "state": AgentState.MONITOR.value,
        "risk_score": 0.05,
        "events_detected": [],
    })
    # Next decision with low risk should allow NORMAL
    decision = agent._decide(0.05, [])
    assert decision == AgentState.NORMAL


# ── Finding 13: Long Answer Keyword Redistribution ───────────────────────────

def test_regression_finding_13_long_answer_no_keywords_score():
    grader = LongAnswerGrader()
    student_ans = "The mitochondria is the powerhouse of the cell providing energy via ATP production."
    ref_ans = "The mitochondria is the powerhouse of the cell providing energy via ATP production."
    # With empty keywords, score should not collapse to zero or fail
    res = grader.grade(student_answer=student_ans, reference_answer=ref_ans, keywords=[], marks=10.0)
    assert res.final_score > 7.0
    assert res.percentage > 70.0


# ── Server API: PUT /api/auth/me ─────────────────────────────────────────────

def test_api_auth_me_put(client, tmp_path):
    # Register/login user
    username = f"user_{uuid.uuid4().hex[:6]}"
    email = f"{username}@example.com"
    client.post("/api/auth/register", json={
        "username": username,
        "email": email,
        "full_name": "Original Name",
        "password": "Password123!",
        "role": "student",
    })
    login_res = client.post("/api/auth/login", json={
        "username": username,
        "password": "Password123!",
    })
    token = login_res.get_json()["token"]

    # PUT update full_name and email
    put_res = client.put(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        json={"full_name": "Updated Name", "email": f"new_{email}"},
    )
    assert put_res.status_code == 200
    user = put_res.get_json()["user"]
    assert user["full_name"] == "Updated Name"
    assert user["email"] == f"new_{email}"


# ── Server API: Agent Status includes agent_session_id ───────────────────────

def test_api_agent_status_includes_agent_session_id(client):
    res = client.get("/api/agent/status?session_key=nonexistent_key")
    assert res.status_code == 200
    data = res.get_json()
    assert "agent_session_id" in data


# ── Server API: Audit Chain session resolution ───────────────────────────────

def test_api_audit_chain_resolution(client):
    res = client.get("/api/trace/audit-chain/test_session_id")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "resolved_session_id" in data


# ── Server API: Exam Roster & Release Endpoints ──────────────────────────────

def test_api_exam_roster_and_release(client):
    exam_id = f"exam_{uuid.uuid4().hex[:8]}"
    create_res = client.post("/api/exams", json={"exam_id": exam_id, "title": "Roster Exam", "duration_minutes": 45})
    assert create_res.status_code == 200

    # Enroll single student
    r1 = client.post(f"/api/exams/{exam_id}/roster", json={
        "student_id": "stud_001",
        "student_name": "Alice Smith",
        "student_email": "alice@school.edu",
    })
    assert r1.status_code == 200
    assert r1.get_json()["status"] == "enrolled"

    # Get roster
    r2 = client.get(f"/api/exams/{exam_id}/roster")
    assert r2.status_code == 200
    roster = r2.get_json()["roster"]
    assert len(roster) == 1
    assert roster[0]["student_id"] == "stud_001"

    # Bulk enroll
    r3 = client.post(f"/api/exams/{exam_id}/roster/bulk", json={
        "students": [
            {"student_id": "stud_002", "student_name": "Bob Jones"},
            {"student_id": "stud_003", "student_name": "Charlie Brown"},
        ]
    })
    assert r3.status_code == 200
    assert r3.get_json()["enrolled_count"] == 2

    # Check release endpoint
    rel = client.post(f"/api/exams/{exam_id}/release", json={"is_released": True})
    assert rel.status_code == 200
    assert rel.get_json()["is_released"] is True

    # Delete from roster
    del_res = client.delete(f"/api/exams/{exam_id}/roster/stud_001")
    assert del_res.status_code == 200
    assert del_res.get_json()["status"] == "removed"


# ── Server API: Grade Overrides & Examiner Actions ───────────────────────────

def test_api_grade_overrides_and_examiner_actions(client):
    session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # Record Grade Override
    ov_res = client.post("/api/grading/override", json={
        "session_id": session_id,
        "question_id": "q1",
        "original_score": 5.0,
        "override_score": 8.5,
        "reason": "Clarified derivation in office hours",
        "examiner_id": "prof_oak",
    })
    assert ov_res.status_code == 200
    data = ov_res.get_json()
    assert data["status"] == "success"
    assert data["override_score"] == 8.5

    # Retrieve Overrides
    get_ov = client.get(f"/api/grading/overrides/{session_id}")
    assert get_ov.status_code == 200
    assert len(get_ov.get_json()["overrides"]) == 1

    # Record Examiner Action
    act_res = client.post("/api/examiner/action", json={
        "action_type": "warn",
        "session_id": session_id,
        "student_id": "stud_123",
        "exam_id": "exam_456",
        "message": "Please face the camera directly.",
        "examiner_id": "prof_oak",
    })
    assert act_res.status_code == 200
    assert act_res.get_json()["status"] == "success"

    # Retrieve Examiner Actions
    get_act = client.get(f"/api/examiner/actions/{session_id}")
    assert get_act.status_code == 200
    assert len(get_act.get_json()["actions"]) == 1


# ── Server API: Predict Performance Insufficient Data ───────────────────────

def test_api_predict_performance_insufficient_data(client):
    res = client.post("/api/predict/performance", json={
        "student_id": "new_student_without_history",
        "features": [0.5, 0.2, 0.8, 0.1],
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "insufficient_data"
    assert data["prediction"] is None


# ── Security: Preview Mode Authorization Gate ────────────────────────────────

def test_security_preview_authorization_gate(client):
    from app.api.server import _get_auth_db
    auth_db = _get_auth_db()

    # Create an unreleased exam
    exam_id = f"exam_sec_unrel_{uuid.uuid4().hex[:6]}"
    create_res = client.post("/api/exams", json={
        "exam_id": exam_id,
        "title": "Unreleased Test Exam",
        "duration_minutes": 30,
        "is_released": False,
        "questions": [{"type": "mcq", "text": "Q1", "options": ["A", "B"], "correct_answer": "A", "marks": 1}],
    })
    assert create_res.status_code == 200

    # 1. Unauthenticated client attempting preview mode -> 403 unauthorized_preview
    unauth_prev = client.post("/api/exam-sessions", json={
        "exam_id": exam_id,
        "student_id": "anon_attacker",
        "is_preview": True,
    })
    assert unauth_prev.status_code == 403
    assert unauth_prev.get_json()["error"] == "unauthorized_preview"

    # 2. Ordinary student attempting preview mode with header or flag -> 403 unauthorized_preview
    stud_uid = auth_db.create_user(f"stud_{uuid.uuid4().hex[:6]}", f"stud_{uuid.uuid4().hex[:6]}@school.edu", "Student One", "pass123", role="student")
    stud_token = auth_db.create_session(stud_uid)

    stud_prev = client.post("/api/exam-sessions",
                            json={"exam_id": exam_id, "student_id": "stud_1", "is_preview": True},
                            headers={"Authorization": f"Bearer {stud_token}"})
    assert stud_prev.status_code == 403
    assert stud_prev.get_json()["error"] == "unauthorized_preview"

    # 3. Authenticated examiner attempting preview mode -> 200 OK preview allowed
    exam_uid = auth_db.create_user(f"exam_{uuid.uuid4().hex[:6]}", f"exam_{uuid.uuid4().hex[:6]}@school.edu", "Examiner One", "pass123", role="examiner")
    exam_token = auth_db.create_session(exam_uid)

    exam_prev = client.post("/api/exam-sessions",
                            json={"exam_id": exam_id, "student_id": "preview_reviewer", "is_preview": True},
                            headers={"Authorization": f"Bearer {exam_token}"})
    assert exam_prev.status_code == 200
    assert exam_prev.get_json()["is_preview"] is True


# ── Security: Student Session Isolation ─────────────────────────────────────

def test_security_student_session_isolation(client):
    from app.api.server import _get_auth_db
    auth_db = _get_auth_db()

    # Create users
    stud1_uid = auth_db.create_user(f"alice_{uuid.uuid4().hex[:6]}", f"alice_{uuid.uuid4().hex[:6]}@school.edu", "Alice", "pass123", role="student")
    stud2_uid = auth_db.create_user(f"bob_{uuid.uuid4().hex[:6]}", f"bob_{uuid.uuid4().hex[:6]}@school.edu", "Bob", "pass123", role="student")
    tok1 = auth_db.create_session(stud1_uid)
    tok2 = auth_db.create_session(stud2_uid)

    # Create exam
    eid = f"exam_iso_{uuid.uuid4().hex[:6]}"
    client.post("/api/exams", json={
        "exam_id": eid, "title": "Isolation Exam", "is_released": True,
        "questions": [{"type": "mcq", "text": "Q1", "options": ["A", "B"], "correct_answer": "A", "marks": 1}]
    })

    # Enroll Alice in roster
    client.post(f"/api/exams/{eid}/roster", json={"student_id": stud1_uid})

    # Alice starts session
    sess_res = client.post("/api/exam-sessions", json={"exam_id": eid, "student_id": stud1_uid},
                           headers={"Authorization": f"Bearer {tok1}"})
    assert sess_res.status_code == 200
    session_id = sess_res.get_json()["session_id"]

    # Bob attempts to submit answer to Alice's session -> 403 Forbidden
    hack_ans = client.post("/api/answers", json={
        "session_id": session_id, "question_id": "q1", "answer": "B"
    }, headers={"Authorization": f"Bearer {tok2}"})
    assert hack_ans.status_code == 403

    # Bob attempts to finish Alice's session -> 403 Forbidden
    hack_fin = client.post(f"/api/exam-sessions/{session_id}/finish",
                           headers={"Authorization": f"Bearer {tok2}"})
    assert hack_fin.status_code == 403

    # Alice submits answer to her own session -> 200 OK
    own_ans = client.post("/api/answers", json={
        "session_id": session_id, "question_id": "q1", "answer": "A"
    }, headers={"Authorization": f"Bearer {tok1}"})
    assert own_ans.status_code == 200


# ── Model Hot-Swap & Adversarial Benchmark APIs ──────────────────────────────

def test_models_hot_swap_and_adversarial_apis(client):
    # Test GET status
    st = client.get("/api/models/status")
    assert st.status_code == 200
    data = st.get_json()
    assert data["status"] == "success"
    assert "active_agents_count" in data
    assert "logistic_model" in data
    assert "anomaly_model" in data

    # Test POST activate hot-swap
    act = client.post("/api/models/activate")
    assert act.status_code == 200
    act_data = act.get_json()
    assert act_data["status"] == "success"
    assert act_data["hot_swap_applied"] is True

    # Test adversarial benchmark endpoint
    adv = client.get("/api/ml/adversarial-benchmark")
    assert adv.status_code == 200
    adv_data = adv.get_json()
    assert adv_data["status"] == "success"
    assert "scenarios" in adv_data
    assert "summary" in adv_data


# ── Honest Performance Prediction Confidence Bounds ─────────────────────────

def test_honest_performance_prediction_bounds(client):
    # Endpoint returns model version and prediction bounds
    res = client.get("/api/analytics/predict-performance?student_id=test_student_honest")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "prediction_model_version" in data
    assert "prediction_interval" in data
    assert data["prediction_interval"]["confidence_level"] == 0.95
