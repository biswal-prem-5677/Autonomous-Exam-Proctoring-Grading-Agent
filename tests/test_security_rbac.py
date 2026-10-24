"""Security / RBAC / override / telemetry regression tests (run WITH the route guard on)."""
import threading
import uuid

import pytest

from app.api.server import app, _get_auth_db, _active_sessions

pytestmark = pytest.mark.enforce_auth


def _user(role):
    n = f"{role[:3]}{uuid.uuid4().hex[:6]}"
    uid = _get_auth_db().create_user(username=n, email=n + "@t.io", full_name=n, password="secret12",
                                     role=role, org_id=None, preferences={})
    return n, {"Authorization": "Bearer " + _get_auth_db().create_session(uid)}


@pytest.fixture()
def world():
    c = app.test_client()
    _, E = _user("examiner")
    sn, S = _user("student")
    eid = c.post("/api/exams", headers=E, json={"title": "Sec", "duration_minutes": 30, "questions": [
        {"type": "mcq", "text": "q", "options": ["a", "b"], "correct_answer": "a", "marks": 4}]}).get_json()["exam_id"]
    c.post(f"/api/exams/{eid}/roster", headers=E, json={"student_id": sn})
    c.post(f"/api/exams/{eid}/release", headers=E, json={})
    return c, E, S, sn, eid


def test_anonymous_is_rejected_on_protected_routes(world):
    c, E, S, sn, eid = world
    for method, url in [("post", "/api/exams"), ("get", f"/api/exams/{eid}"), ("get", "/api/exams"),
                        ("post", f"/api/exams/{eid}/roster"), ("post", f"/api/exams/{eid}/release"),
                        ("get", f"/api/examiner/exam/{eid}/overview"), ("post", "/api/models/activate"),
                        ("post", "/api/training/train"), ("post", "/api/grading/override")]:
        assert getattr(c, method)(url, json={}).status_code == 401, url


def test_public_routes_stay_public():
    c = app.test_client()
    assert c.get("/api/health").status_code == 200
    assert c.post("/api/auth/login", json={}).status_code != 401


def test_student_cannot_use_staff_routes(world):
    c, E, S, sn, eid = world
    for method, url in [("post", "/api/exams"), ("post", f"/api/exams/{eid}/questions"),
                        ("post", f"/api/exams/{eid}/roster"), ("post", f"/api/exams/{eid}/release"),
                        ("post", "/api/grading/override"), ("post", "/api/training/train"),
                        ("post", "/api/models/activate"), ("post", "/api/examiner/action"),
                        ("get", f"/api/examiner/exam/{eid}/overview"), ("get", "/api/sessions")]:
        assert getattr(c, method)(url, headers=S, json={}).status_code == 403, url


def test_student_never_receives_answer_key_and_only_sees_enrolled_exams(world):
    c, E, S, sn, eid = world
    other = c.post("/api/exams", headers=E, json={"title": "NotMine"}).get_json()["exam_id"]
    c.post(f"/api/exams/{other}/roster", headers=E, json={"student_id": "someone_else"})   # restricted exam
    ids = [e["exam_id"] for e in c.get("/api/exams", headers=S).get_json()["exams"]]
    assert eid in ids and other not in ids
    assert "correct_answer" not in str(c.get(f"/api/exams/{eid}", headers=S).get_json())
    start = c.post("/api/exam-sessions", headers=S, json={"student_id": sn, "exam_id": eid}).get_json()
    assert "correct_answer" not in str(start["questions"])


def test_student_cannot_touch_another_students_session(world):
    c, E, S, sn, eid = world
    other, S2 = _user("student")
    assert c.get(f"/api/proctor/summary/{sn}/{eid}", headers=S2).status_code == 403
    assert c.post(f"/api/exam-sessions/{sn}/{eid}/answers", headers=S2, json={}).status_code == 403
    assert c.get(f"/api/agent/status?session_key={sn}:{eid}", headers=S2).status_code == 403


def test_autosave_with_token_and_override_reaches_every_screen(world):
    c, E, S, sn, eid = world
    c.post("/api/exam-sessions", headers=S, json={"student_id": sn, "exam_id": eid})
    assert c.post(f"/api/exam-sessions/{sn}/{eid}/answers", headers=S,
                  json={"question_id": "q1", "answer": "b"}).status_code == 200      # wrong answer
    c.post("/api/grading/submit", headers=S, json={"student_id": sn, "exam_id": eid, "answers": {"q1": "b"}})
    c.post(f"/api/exam-sessions/{sn}/{eid}/submit", headers=S, json={})
    key = f"{sn}:{eid}"
    for sid in (key, _active_sessions[key]["session_id"]):          # UI may send either id form
        r = c.post("/api/grading/override", headers=E, json={
            "session_id": sid, "question_id": "q1", "original_score": 0, "override_score": 4, "reason": "ok"})
        assert r.status_code == 200
    detail = c.get(f"/api/student/results/{eid}", headers=S).get_json()
    assert detail["total_score"] == 4.0
    assert [x["total_score"] for x in c.get("/api/student/results", headers=S).get_json()["results"]] == [4.0]
    subs = c.get(f"/api/examiner/submissions?exam_id={eid}", headers=E).get_json()["submissions"]
    assert [x["total_score"] for x in subs] == [4.0]


def test_clipboard_activity_becomes_an_agent_event(world):
    c, E, S, sn, eid = world
    c.post("/api/exam-sessions", headers=S, json={"student_id": sn, "exam_id": eid})
    key = f"{sn}:{eid}"
    c.post("/api/behavioral/analyze", headers=S, json={
        "session_key": key, "face_present": True, "face_count": 1, "face_confidence": 0.9,
        "vision_status": "OK", "browser": {"delta_pastes": 1}})
    hist = c.get(f"/api/agent/history?session_key={key}", headers=E).get_json()["history"]
    assert "paste_detected" in {e["type"] for h in hist for e in h.get("events_detected", [])}


def test_authenticated_requests_work_from_many_threads(world):
    c, E, S, sn, eid = world
    results = []
    def hit():
        results.append(app.test_client().get("/api/dashboard", headers=S).status_code)
    ts = [threading.Thread(target=hit) for _ in range(12)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert results == [200] * 12


def test_student_cannot_bruteforce_answers_via_grading_submit(world):
    c, E, S, sn, eid = world
    c.post("/api/exam-sessions", headers=S, json={"student_id": sn, "exam_id": eid})
    first = c.post("/api/grading/submit", headers=S, json={"student_id": sn, "exam_id": eid, "answers": {"q1": "b"}})
    assert first.status_code == 200
    assert "total_score" not in first.get_json() and "per_question" not in first.get_json()
    again = c.post("/api/grading/submit", headers=S, json={"student_id": sn, "exam_id": eid, "answers": {"q1": "a"}})
    assert again.status_code == 409


def test_two_exams_created_in_same_second_get_distinct_ids():
    c = app.test_client()
    _, E = _user("examiner")
    ids = {c.post("/api/exams", headers=E, json={"title": f"X{i}"}).get_json()["exam_id"] for i in range(5)}
    assert len(ids) == 5
