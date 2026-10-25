"""Tests for strict authentication & identity binding (Step 4 of final stabilization)."""

import pytest
import uuid
from app.api.server import app, _get_auth_db, _get_db


@pytest.fixture
def auth_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def seeded_users(monkeypatch):
    monkeypatch.setenv("EXAM_AGENT_AUTH_ENFORCE", "1")
    auth_db = _get_auth_db()

    # Create examiner user
    examiner_name = f"examiner_{uuid.uuid4().hex[:6]}"
    examiner_uid = auth_db.create_user(
        username=examiner_name,
        email=f"{examiner_name}@school.edu",
        full_name="Examiner Real",
        password="Password123!",
        role="examiner",
    )
    examiner_token = auth_db.create_session(examiner_uid)

    # Create student user
    student_name = f"student_{uuid.uuid4().hex[:6]}"
    student_uid = auth_db.create_user(
        username=student_name,
        email=f"{student_name}@school.edu",
        full_name="Student Real",
        password="Password123!",
        role="student",
    )
    student_token = auth_db.create_session(student_uid)

    return {
        "examiner_name": examiner_name,
        "examiner_uid": examiner_uid,
        "examiner_token": examiner_token,
        "student_name": student_name,
        "student_uid": student_uid,
        "student_token": student_token,
    }


def test_grade_override_binds_to_authenticated_examiner(auth_client, seeded_users):
    examiner_token = seeded_users["examiner_token"]
    examiner_name = seeded_users["examiner_name"]
    db = _get_db()

    # Create real session and answer in DB
    sid = db.create_session(student_id="student_target", exam_id="exam_override_1")
    db.record_answer(session_id=sid, question_id="q1", student_answer="42", score=5.0)

    # Attempt to spoof examiner_id as "spoofed_admin"
    res = auth_client.post(
        "/api/grading/override",
        headers={"Authorization": f"Bearer {examiner_token}"},
        json={
            "session_id": sid,
            "question_id": "q1",
            "original_score": 5.0,
            "override_score": 9.5,
            "reason": "Clear derivation",
            "examiner_id": "spoofed_admin",
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    # Must bind to authenticated user, rejecting spoofed examiner_id
    assert data["examiner_id"] == examiner_name

    # Verify DB audit record also stored authenticated examiner_name
    overrides = db.get_grade_overrides(session_id=sid)
    assert len(overrides) == 1
    assert overrides[0]["examiner_id"] == examiner_name


def test_create_exam_binds_creator_identity(auth_client, seeded_users):
    examiner_token = seeded_users["examiner_token"]
    examiner_name = seeded_users["examiner_name"]
    db = _get_db()
    eid = f"exam_bind_{uuid.uuid4().hex[:6]}"

    # Attempt to spoof created_by
    res = auth_client.post(
        "/api/exams",
        headers={"Authorization": f"Bearer {examiner_token}"},
        json={
            "exam_id": eid,
            "title": "Identity Binding Exam",
            "duration_minutes": 30,
            "created_by": "spoofed_creator",
        },
    )
    assert res.status_code == 200

    # Verify DB has created_by bound to real examiner
    saved_exam = db.get_exam(eid)
    assert saved_exam is not None
    assert saved_exam.get("created_by") == examiner_name


def test_student_forbidden_from_staff_routes_when_enforced(auth_client, seeded_users):
    student_token = seeded_users["student_token"]

    # Student cannot call grade override
    res = auth_client.post(
        "/api/grading/override",
        headers={"Authorization": f"Bearer {student_token}"},
        json={
            "session_id": "sess_dummy",
            "question_id": "q1",
            "override_score": 100.0,
        },
    )
    assert res.status_code == 403
    assert "Forbidden" in res.get_json()["error"]


def test_unauthenticated_request_rejected_when_enforced(auth_client, seeded_users):
    # Missing Authorization header on staff route -> 401
    res = auth_client.post(
        "/api/grading/override",
        json={
            "session_id": "sess_dummy",
            "question_id": "q1",
            "override_score": 100.0,
        },
    )
    assert res.status_code == 401
    assert "Authentication required" in res.get_json()["error"]
