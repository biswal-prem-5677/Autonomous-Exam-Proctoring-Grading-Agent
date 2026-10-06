"""End-to-end browser-to-server lifecycle integration test.

Tests the full lifecycle as executed by the browser:
1. Exam creation (examiner)
2. Preflight verification with exam_id and student_id
3. Exam session start (POST /api/exam-sessions) creating server-side agent, controller, and evidence log
4. Real image frame decoding and FaceTracker vision inference (POST /api/vision/analyze)
5. Multimodal telemetry ingestion & agent loop processing (POST /api/behavioral/analyze)
6. IntegrityDecisionTrace inspection (GET /api/agent/status?session_key=...)
7. Answer submission & grading (POST /api/grading/submit & POST /api/exam-sessions/.../submit)
8. Examiner live overview & student inspection (GET /api/examiner/exam/.../overview & student)
"""
import base64
import json
import numpy as np
import cv2
import pytest
from app.api.server import app, _active_sessions, _proctoring_agents, _evidence_logs


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_full_browser_to_examiner_lifecycle(client):
    exam_id = "e2e_exam_test_001"
    student_id = "e2e_student_alice"
    expected_session_key = f"{student_id}:{exam_id}"

    # ── Step 1: Create Exam ──────────────────────────────────────────────────
    exam_payload = {
        "id": exam_id,
        "title": "Computer Science Integrity Benchmark",
        "description": "Comprehensive E2E test exam",
        "duration_minutes": 45,
        "created_by": "examiner_dr_smith",
        "questions": [
            {
                "id": "q1",
                "type": "mcq",
                "text": "What is the time complexity of quicksort average case?",
                "options": ["O(n)", "O(n log n)", "O(n^2)", "O(log n)"],
                "correct_answer": "O(n log n)",
                "marks": 5.0,
            },
            {
                "id": "q2",
                "type": "numerical",
                "text": "What is 2^10?",
                "correct_answer": 1024,
                "tolerance": 0.0,
                "marks": 5.0,
            },
            {
                "id": "q3",
                "type": "short_answer",
                "text": "Define idempotency in HTTP REST APIs.",
                "correct_answer": "An operation that produces the same result no matter how many times it is executed.",
                "keywords": ["same result", "multiple executions", "identical"],
                "marks": 10.0,
            },
        ],
    }
    r = client.post("/api/exams", data=json.dumps(exam_payload), content_type="application/json")
    assert r.status_code == 200, f"Create exam failed: {r.data}"

    # ── Step 2: Student Preflight Check ──────────────────────────────────────
    # Proves Bug 1 is resolved: exam_id and student_id are properly supplied and preflight passes
    preflight_payload = {
        "require_webcam": False,
        "require_microphone": False,
        "lock_fullscreen": False,
        "camera_available": True,
        "online": True,
        "browser_capabilities": {
            "fullscreen": True,
            "get_user_media": True,
        },
        "exam_id": exam_id,
        "student_id": student_id,
    }
    r_pref = client.post("/api/preflight", data=json.dumps(preflight_payload), content_type="application/json")
    assert r_pref.status_code == 200
    pref_data = r_pref.get_json()
    assert pref_data.get("can_proceed") is True, f"Preflight rejected: {pref_data}"

    # ── Step 3: Exam Session Start (POST /api/exam-sessions) ─────────────────
    # Proves Bug 2 is resolved: frontend creates server-side session and binds session_key
    r_session = client.post("/api/exam-sessions", data=json.dumps({
        "student_id": student_id,
        "exam_id": exam_id,
        "duration_minutes": 45,
    }), content_type="application/json")
    assert r_session.status_code == 200, f"Start session failed: {r_session.data}"
    sess_data = r_session.get_json()
    assert sess_data["session_key"] == expected_session_key
    assert sess_data["student_id"] == student_id
    assert sess_data["exam_id"] == exam_id
    assert len(sess_data["questions"]) == 3

    # Assert server-side registries are genuinely populated
    assert expected_session_key in _active_sessions
    assert expected_session_key in _proctoring_agents
    assert expected_session_key in _evidence_logs

    # ── Step 4: Real Webcam Frame Vision Inference ────────────────────────────
    # Generate a real 640x480 synthetic JPEG image frame
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple background pattern
    cv2.rectangle(img, (100, 100), (300, 300), (128, 128, 128), -1)
    _, buffer = cv2.imencode(".jpg", img)
    jpg_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

    vision_payload = {
        "session_key": expected_session_key,
        "image": jpg_b64,
    }
    r_vis = client.post("/api/vision/analyze", data=json.dumps(vision_payload), content_type="application/json")
    assert r_vis.status_code == 200
    vis_data = r_vis.get_json()
    assert vis_data["image_processed"] is True
    assert vis_data["vision_status"] == "OK"
    assert "face_present" in vis_data
    assert "risk_contribution" in vis_data

    # ── Step 5: Real Telemetry Ingestion into Agent Loop ──────────────────────
    # Send keyboard, mouse, and browser stats under the bound session_key
    behavioral_payload = {
        "session_key": expected_session_key,
        "face_present": True,
        "face_count": 1,
        "face_confidence": 0.92,
        "head_pose": {"yaw": 5, "pitch": 2, "roll": 0},
        "keyboard": {
            "events_per_minute": 110,
            "mean_key_hold": 85,
            "mean_latency": 140,
            "std_key_hold": 15,
            "std_latency": 30,
            "idle_seconds": 2,
            "deviation": 0.12,
        },
        "mouse": {
            "avg_speed": 220,
            "total_distance": 1450,
            "clicks": 8,
            "idle_seconds": 3,
            "deviation": 0.15,
        },
        "browser": {
            "tab_switches": 0,
            "window_blurs": 0,
            "fullscreen_exits": 0,
            "visibility_changes": 0,
        },
        "session_elapsed": 60,
    }
    r_beh = client.post("/api/behavioral/analyze", data=json.dumps(behavioral_payload), content_type="application/json")
    assert r_beh.status_code == 200
    beh_data = r_beh.get_json()
    assert beh_data["session_key"] == expected_session_key
    assert beh_data["agent_state"] in ("NORMAL", "LOW_CONCERN", "MONITOR")
    assert isinstance(beh_data["agent_risk"], (int, float))

    # ── Step 6: Verify Live Agent Trace ──────────────────────────────────────
    r_trace = client.get(f"/api/agent/status?session_key={expected_session_key}")
    assert r_trace.status_code == 200
    trace_data = r_trace.get_json()
    assert trace_data["iteration"] >= 1
    assert trace_data["state"] in ("NORMAL", "LOW_CONCERN", "MONITOR")
    assert "current_risk" in trace_data
    # Assert real IntegrityDecisionTrace was created
    last_trace = trace_data.get("last_trace")
    assert last_trace is not None
    assert "uncertainty" in last_trace
    assert "corroboration" in last_trace
    assert "decision_reason" in last_trace

    # ── Step 7: Answer Submission & Session Completion ───────────────────────
    answers = {
        "q1": "O(n log n)",
        "q2": "1024",
        "q3": "An operation that produces the same result no matter how many times it is executed.",
    }
    r_submit = client.post("/api/grading/submit", data=json.dumps({
        "student_id": student_id,
        "exam_id": exam_id,
        "session_key": expected_session_key,
        "answers": answers,
    }), content_type="application/json")
    assert r_submit.status_code == 200
    submit_data = r_submit.get_json()
    assert submit_data["percentage"] > 80.0
    assert submit_data["answers_count"] == 3

    # Close session controller
    r_end = client.post(f"/api/exam-sessions/{student_id}/{exam_id}/submit", data=json.dumps({
        "answers": answers,
    }), content_type="application/json")
    assert r_end.status_code == 200
    assert r_end.get_json()["status"] == "submitted"

    # ── Step 8: Examiner Overview & Student Detail Inspection ─────────────────
    r_overview = client.get(f"/api/examiner/exam/{exam_id}/overview")
    assert r_overview.status_code == 200
    overview_data = r_overview.get_json()
    student_entries = [s for s in overview_data.get("students", []) if s["student_id"] == student_id]
    assert len(student_entries) == 1, f"Student {student_id} not visible in examiner overview: {overview_data}"
    student_entry = student_entries[0]
    assert student_entry["student_id"] == student_id
    assert student_entry["state"] is not None

    r_detail = client.get(f"/api/examiner/student/{student_id}/{exam_id}")
    assert r_detail.status_code == 200
    detail_data = r_detail.get_json()
    assert detail_data["student_id"] == student_id
    assert detail_data["exam_id"] == exam_id
    assert "risk_timeline" in detail_data
    assert len(detail_data["risk_timeline"]) >= 1
    assert "contributing_signals" in detail_data
