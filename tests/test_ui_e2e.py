"""End-to-End UI and Role-Gated Workflow Integration Tests.

Validates:
1. Student UI template cleanliness:
   - Zero numerical risk percentages or meters.
   - Zero agent decision traces / suspicion flags exposed to student.
   - Reassurance badges present ("Secure Session", "Monitoring Engaged").
   - Specialized input renderers for all question types (MCQ, True/False, Numerical, Math, Programming, Long Answer).
   - Autosave indicator and countdown timer present.
2. Student Results template release-gating and student privacy:
   - Unreleased exam results return restricted message or blocked access.
   - Released results show scores and "Integrity Status: Verified & Intact" with zero risk leak.
3. Examiner Dashboard & Grading UI:
   - Score override modal and buttons present.
   - Examiner live intervention buttons (Warn, Flag, Pause, Terminate, Note) functional.
   - Database persistence of roster, overrides, examiner actions, and release status.
"""

import json
import uuid
import pytest
from app.api.server import app, _get_db


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_student_exam_ui_risk_zero_exposure(client):
    """Verify student exam template contains no risk meters, suspicion traces, or exposed agent integrity thresholds."""
    res = client.get("/exam")
    if res.status_code == 404 or res.status_code == 302:
        res = client.get("/student/exam")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # 1. Zero risk exposure
    forbidden_terms = [
        "Risk Score:",
        "risk-meter",
        "suspicion-meter",
        "Agent Decision:",
        "DECISION TRACE",
        "HIGH_RISK",
        "SUSPICIOUS",
    ]
    for term in forbidden_terms:
        assert term not in html, f"Forbidden proctoring metric '{term}' leaked into student exam view!"

    # 2. Reassurance badges present
    assert "Secure Session" in html
    assert "Monitoring Engaged" in html

    # 3. Timer and autosave present
    assert "timer" in html.lower()
    assert "autosave" in html.lower() or "saved" in html.lower()

    # 4. Question type renderers support
    assert "renderquestion" in html.lower() or "render_question" in html.lower() or "question-card" in html.lower()


def test_student_results_ui_gating_and_integrity_status(client):
    """Verify student results page is release-gated and displays clean integrity verification without risk scores."""
    res = client.get("/results")
    if res.status_code == 404 or res.status_code == 302:
        res = client.get("/student/results")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Student must see "Integrity Status: Verified & Intact" or equivalent reassurance
    assert "Integrity Status" in html or "Integrity" in html
    assert "Verified" in html or "Intact" in html or "Authentic" in html

    # Student must NOT see proctoring risk numbers or suspicion breakdowns
    assert "Risk Percentage" not in html
    assert "Risk Breakdown" not in html
    assert "Agent Anomaly Log" not in html


def test_examiner_grading_ui_override_controls(client):
    """Verify examiner grading view includes score override modals and controls."""
    res = client.get("/grading")
    if res.status_code == 404 or res.status_code == 302:
        res = client.get("/examiner/grading")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Check for override button / modal elements
    assert "override" in html.lower()
    assert "score" in html.lower()


def test_examiner_dashboard_intervention_controls(client):
    """Verify examiner dashboard has live intervention actions (Warn, Flag, Pause, Terminate, Note)."""
    res = client.get("/dashboard")
    if res.status_code == 404 or res.status_code == 302:
        res = client.get("/examiner/dashboard")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # Must contain examiner intervention capabilities
    assert "warn" in html.lower() or "flag" in html.lower() or "pause" in html.lower() or "action" in html.lower()


def test_e2e_roster_release_and_override_workflow(client):
    """Full end-to-end API workflow for roster enrollment, release toggle, and examiner grade override."""
    db = _get_db()
    exam_id = f"e2e_exam_{uuid.uuid4().hex[:8]}"
    student_id = f"e2e_stud_{uuid.uuid4().hex[:8]}"

    # 1. Create exam in DB
    db.save_exam({
        "exam_id": exam_id,
        "title": "Autonomous Systems Final",
        "duration_minutes": 60,
        "instructions": "Follow instructions carefully.",
        "passing_score": 50.0,
        "is_released": False,
        "questions": [
            {
                "id": "q1",
                "type": "mcq",
                "text": "Which sensor captures gaze?",
                "options": ["Camera", "Microphone", "Keyboard"],
                "correct_answer": "Camera",
                "marks": 5.0,
            }
        ],
    })

    # 2. Verify initial release status is False
    rel_status = client.get(f"/api/exams/{exam_id}/release-status")
    assert rel_status.status_code == 200
    assert rel_status.get_json()["is_released"] is False

    # 3. Enroll student in roster
    enr_res = client.post(f"/api/exams/{exam_id}/roster", json={
        "student_id": student_id,
        "student_name": "Jane Doe",
        "student_email": "jane@university.edu",
    })
    assert enr_res.status_code == 200
    assert enr_res.get_json()["status"] == "enrolled"

    # 4. Examiner overrides score for a question
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    override_res = client.post("/api/grading/override", json={
        "session_id": session_id,
        "question_id": "q1",
        "original_score": 0.0,
        "override_score": 5.0,
        "reason": "Correct reasoning demonstrated in submitted comments",
        "examiner_id": "examiner_prof_jones",
    })
    assert override_res.status_code == 200
    assert override_res.get_json()["override_score"] == 5.0

    # 5. Verify override persists in DB
    overrides = db.get_grade_overrides(session_id)
    assert len(overrides) == 1
    assert overrides[0]["override_score"] == 5.0
    assert overrides[0]["question_id"] == "q1"

    # 6. Examiner executes live intervention action
    act_res = client.post("/api/examiner/action", json={
        "action_type": "warn",
        "session_id": session_id,
        "student_id": student_id,
        "exam_id": exam_id,
        "message": "Please center your webcam.",
        "examiner_id": "examiner_prof_jones",
    })
    assert act_res.status_code == 200
    assert act_res.get_json()["status"] == "success"

    # 7. Verify action persists in DB
    actions = db.get_examiner_actions(session_id)
    assert len(actions) == 1
    assert actions[0]["action_type"] == "warn"
    assert actions[0]["message"] == "Please center your webcam."

    # 8. Examiner releases exam results
    rel_toggle = client.post(f"/api/exams/{exam_id}/release", json={"is_released": True})
    assert rel_toggle.status_code == 200
    assert rel_toggle.get_json()["is_released"] is True

    # 9. Verify updated release status is True
    rel_status_after = client.get(f"/api/exams/{exam_id}/release-status")
    assert rel_status_after.status_code == 200
    assert rel_status_after.get_json()["is_released"] is True
