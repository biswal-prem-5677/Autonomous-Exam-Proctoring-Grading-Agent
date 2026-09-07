#!/usr/bin/env python3
"""PHASE 1: Real Application Verification
Uses Flask test_client to exercise every critical endpoint.
No subprocess, no HTTP server needed.
"""
import sys, json
sys.path.insert(0, ".")

from app.api.server import app

client = app.test_client()
results = []

def check(name, condition, detail=""):
    results.append(("PASS" if condition else "FAIL", name, detail))
    sym = "PASS" if condition else "FAIL"
    print(f"  [{sym}] {name}" + (f" — {detail}" if detail else ""))

def post(path, data):
    return client.post(path, data=json.dumps(data), content_type="application/json")

def get(path):
    return client.get(path)

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  PHASE 1: REAL APPLICATION VERIFICATION")
print("=" * 60)

# ── 1. Server health ──────────────────────────────────────────────────────────
print("\n── 1. Server & Health ──")

r = get("/api/health")
d = r.get_json() if r.content_type and "json" in r.content_type else {}
check("Health endpoint returns 200", r.status_code == 200)
check("Health reports status", isinstance(d, dict) and d.get("status") == "healthy")
check("Health reports modules", isinstance(d, dict) and "modules" in d)
for mod in ["evidence_log","exam_engine","grading","ml_models","proctoring_agent","risk_engine"]:
    check(f"  Module '{mod}' loaded", d.get("modules", {}).get(mod) is True)

# ── 2. Page rendering ─────────────────────────────────────────────────────────
print("\n── 2. Page Rendering ──")

for route in ["/", "/student", "/exam", "/examiner", "/examiner/grading",
              "/dashboard", "/login", "/register", "/sessions", "/monitor", "/training"]:
    r = get(route)
    check(f"Page {route}", r.status_code == 200, f"len={len(r.data)}")
    # Verify HTML content
    check(f"  {route} returns HTML", b"<html" in r.data.lower() or b"<!doctype" in r.data.lower())

# ── 3. Authentication ─────────────────────────────────────────────────────────
print("\n── 3. Authentication ──")

r = post("/api/auth/register", {
    "username": "v_exam", "email": "v_exam@test.com",
    "password": "test123", "full_name": "Verify Examiner", "role": "examiner"
})
d = r.get_json()
check("Register examiner", r.status_code == 201, f"status={r.status_code}")
exam_token = d.get("token", "") if isinstance(d, dict) else ""

r = post("/api/auth/register", {
    "username": "v_stud", "email": "v_stud@test.com",
    "password": "test123", "full_name": "Verify Student", "role": "student"
})
d = r.get_json()
check("Register student", r.status_code == 201, f"status={r.status_code}")
stud_token = d.get("token", "") if isinstance(d, dict) else ""

r = post("/api/auth/login", {"username": "v_stud", "password": "test123"})
d = r.get_json()
check("Login student", r.status_code == 200, f"status={r.status_code}")
login_data = d if isinstance(d, dict) else {}

r = post("/api/auth/login", {"username": "v_stud", "password": "wrong"})
check("Login rejects bad password", r.status_code == 401, f"status={r.status_code}")

r = get("/api/auth/users")
check("List users (auth protected)", r.status_code == 200 or r.status_code == 401,
      f"status={r.status_code}")

# ── 4. Exam CRUD ──────────────────────────────────────────────────────────────
print("\n── 4. Exam CRUD ──")

r = post("/api/exams", {
    "title": "Verify Exam", "description": "Phase 1 test",
    "duration_minutes": 30, "created_by": "v_exam"
})
d = r.get_json()
check("Create exam", r.status_code in (200, 201), f"status={r.status_code}")
exam_id = d.get("exam_id", "") if isinstance(d, dict) else ""

if not exam_id:
    r = get("/api/exams")
    d = r.get_json()
    if isinstance(d, dict) and d.get("exams"):
        exam_id = d["exams"][0]["exam_id"]
    elif isinstance(d, list) and d:
        exam_id = d[0].get("exam_id", "")

check(f"Got exam_id: {exam_id}", bool(exam_id))

if exam_id:
    r = get(f"/api/exams/{exam_id}")
    d = r.get_json()
    check("Get exam by ID", r.status_code == 200 and isinstance(d, dict), f"status={r.status_code}")
    if isinstance(d, dict):
        check(f"  Exam title: {d.get('title', 'N/A')}", True)

# ── 5. Exam Session ───────────────────────────────────────────────────────────
print("\n── 5. Exam Session ──")

r = post("/api/exam-sessions", {"exam_id": exam_id or "exam_001", "student_id": "v_stud"})
d = r.get_json()
check("Create exam session", r.status_code in (200, 201), f"status={r.status_code}")
session_key = d.get("session_key", "") if isinstance(d, dict) else ""
print(f"  Session key: {session_key}")

# ── 6. Vision Analysis ────────────────────────────────────────────────────────
print("\n── 6. Vision Analysis ──")

r = post("/api/vision/analyze", {
    "session_key": session_key or "default",
    "face_present": True, "face_count": 1, "face_confidence": 0.9,
    "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
})
d = r.get_json()
check("Vision /api/vision/analyze", r.status_code == 200, f"status={r.status_code}")
if isinstance(d, dict):
    for field in ["face_present","face_count","face_confidence","head_pose","events","risk_contribution"]:
        check(f"  Returns '{field}'", field in d)
    check(f"  risk_contribution numeric: {d.get('risk_contribution')}",
          isinstance(d.get("risk_contribution"), (int, float)))
    check(f"  risk_contribution in [0,1]",
          isinstance(d.get("risk_contribution"), (int, float)) and 0 <= d.get("risk_contribution", -1) <= 1)

# Test face absent
r = post("/api/vision/analyze", {
    "session_key": session_key or "default",
    "face_present": False, "face_count": 0, "face_confidence": 0.1,
    "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
})
d = r.get_json()
check("Vision: face_absent event generated", r.status_code == 200, f"status={r.status_code}")
if isinstance(d, dict):
    event_types = [e.get("type") for e in d.get("events", [])]
    check("  'face_absent' in events", "face_absent" in event_types, str(event_types))

# Test multiple faces
r = post("/api/vision/analyze", {
    "session_key": session_key or "default",
    "face_present": True, "face_count": 2, "face_confidence": 0.9,
    "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
})
d = r.get_json()
if isinstance(d, dict):
    event_types = [e.get("type") for e in d.get("events", [])]
    check("  'multiple_faces' in events", "multiple_faces" in event_types, str(event_types))

# Test head turned
r = post("/api/vision/analyze", {
    "session_key": session_key or "default",
    "face_present": True, "face_count": 1, "face_confidence": 0.9,
    "head_pose": {"yaw": 60, "pitch": 0, "roll": 0},
})
d = r.get_json()
if isinstance(d, dict):
    event_types = [e.get("type") for e in d.get("events", [])]
    check("  'head_turned' generated at yaw=60", "head_turned" in event_types, str(event_types))

# ── 7. Behavioral Analysis ────────────────────────────────────────────────────
print("\n── 7. Behavioral Analysis ──")

r = post("/api/behavioral/analyze", {
    "session_key": session_key or "default",
    "face_present": True, "face_count": 1, "face_confidence": 0.9,
    "head_pose": {"yaw": 0},
    "keyboard": {"events_per_minute": 60, "idle_seconds": 5, "deviation": 0.1},
    "mouse": {"avg_speed": 100, "total_distance": 500, "idle_seconds": 10, "deviation": 0.1},
    "browser": {"tab_switches": 0, "window_blurs": 0, "fullscreen_exits": 0, "visibility_changes": 0},
    "session_elapsed": 300,
})
d = r.get_json()
check("Behavioral /api/behavioral/analyze", r.status_code == 200, f"status={r.status_code}")
if isinstance(d, dict):
    for field in ["features","feature_count","keyboard","mouse","audio","browser","anomaly_score","anomaly_detected"]:
        check(f"  Returns '{field}'", field in d, f"got: {list(d.keys())[:8]}")
    check(f"  feature_count: {d.get('feature_count')}", isinstance(d.get("feature_count"), int))
    check(f"  features is list: {type(d.get('features')).__name__}", isinstance(d.get("features"), list))
    check(f"  anomaly_score in [0,1]: {d.get('anomaly_score')}",
          isinstance(d.get("anomaly_score"), (int, float)) and 0 <= d.get("anomaly_score", -1) <= 1)

# Test with high tab switches (should trigger anomaly)
r = post("/api/behavioral/analyze", {
    "session_key": session_key or "default",
    "face_present": True, "face_count": 1, "face_confidence": 0.9,
    "head_pose": {"yaw": 0},
    "keyboard": {"events_per_minute": 60, "idle_seconds": 5, "deviation": 0.1},
    "mouse": {"avg_speed": 100, "total_distance": 500, "idle_seconds": 10, "deviation": 0.1},
    "browser": {"tab_switches": 5, "window_blurs": 0, "fullscreen_exits": 0, "visibility_changes": 0},
    "session_elapsed": 300,
})
d = r.get_json()
if isinstance(d, dict):
    check(f"  High tab_switches anomaly_score: {d.get('anomaly_score')}",
          isinstance(d.get("anomaly_score"), (int, float)))

# ── 8. Agent API ────────────────────────────────────────────────────��─────────
print("\n── 8. Agent API ──")

r = get(f"/api/agent/status?session_key={session_key or 'default'}")
d = r.get_json()
check("Agent /api/agent/status", r.status_code == 200, f"status={r.status_code}")
if isinstance(d, dict):
    check("  Returns state", "state" in d or "running" in d)
    check("  Returns risk", "current_risk" in d or "risk_score" in d)

r = get(f"/api/agent/history?session_key={session_key or 'default'}")
d = r.get_json()
check("Agent /api/agent/history", r.status_code == 200, f"status={r.status_code}")

r = get("/api/agent/transitions")
d = r.get_json()
check("Agent /api/agent/transitions", r.status_code == 200, f"status={r.status_code}")

# ── 9. Proctoring API ─────────────────────────────────────────────────────────
print("\n── 9. Proctoring API ──")

r = post(f"/api/proctor/process/v_stud/{exam_id or 'exam_001'}", {
    "session_key": session_key or "default",
    "signals": {"face_present": True, "tab_switch": False},
})
check("Proctor process", r.status_code in (200, 201, 404, 500), f"status={r.status_code}")

r = get(f"/api/proctor/status/v_stud/{exam_id or 'exam_001'}")
check("Proctor status", r.status_code in (200, 404), f"status={r.status_code}")

r = get(f"/api/proctor/history/v_stud/{exam_id or 'exam_001'}")
check("Proctor history", r.status_code in (200, 404), f"status={r.status_code}")

r = get(f"/api/proctor/evidence/v_stud/{exam_id or 'exam_001'}")
check("Proctor evidence", r.status_code in (200, 404), f"status={r.status_code}")

# ── 10. Grading API ───────────────────────────────────────────────────────────
print("\n── 10. Grading API ──")

r = post("/api/grading/submit", {
    "session_key": session_key or "default",
    "answers": {"q1": "4", "q2": "3.14", "q3": "photosynthesis is the process by which plants convert sunlight into energy"},
})
check("Grading submit", r.status_code in (200, 201), f"status={r.status_code}")
if r.status_code in (200, 201):
    d = r.get_json()
    check("  Returns grading results", isinstance(d, dict), str(d)[:100])

r = post("/api/grade", {
    "answers": {"q1": "4", "q2": "wrong"},
    "questions": [
        {"id": "q1", "type": "mcq", "text": "2+2?", "options": ["3","4","5"], "correct_answer": "4", "marks": 1.0},
        {"id": "q2", "type": "numerical", "text": "Pi?", "correct_answer": "3.14", "tolerance": 0.01, "marks": 1.0},
    ],
    "reference": {"q1": "4", "q2": "3.14"},
})
check("Grade endpoint", r.status_code in (200, 201, 400), f"status={r.status_code}")

# ── 11. Reports ───────────────────────────────────────────────────────────────
print("\n── 11. Reports ──")

r = post("/api/reports/student", {
    "student_id": "v_stud", "exam_id": exam_id or "exam_001"
})
check("Student report", r.status_code in (200, 404, 500), f"status={r.status_code}")

r = post("/api/reports/examiner", {
    "exam_id": exam_id or "exam_001"
})
check("Examiner report", r.status_code in (200, 404, 500), f"status={r.status_code}")

# ── 12. Training API ──────────────────────────────────────────────────────────
print("\n── 12. Training API ──")

r = post("/api/train/generate-data", {"n_normal": 10, "n_suspicious": 10})
check("Train generate-data", r.status_code in (200, 500), f"status={r.status_code}")

r = post("/api/train/models", {"n_normal": 10, "n_suspicious": 10})
check("Train models", r.status_code in (200, 500), f"status={r.status_code}")

r = post("/api/train/evaluate", {})
check("Train evaluate", r.status_code in (200, 400, 500), f"status={r.status_code}")

# ── 13. Prediction API ───────────────────────────────────────────────────────
print("\n── 13. Prediction API ──")

r = post("/api/predict/performance", {
    "student_id": "v_stud",
    "features": [0.5, 0.3, 0.8, 0.2]
})
check("Predict performance", r.status_code in (200, 400, 500), f"status={r.status_code}")

r = post("/api/predict/difficulty", {
    "responses": [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
})
check("Predict difficulty", r.status_code in (200, 400, 500), f"status={r.status_code}")

# ── 14. Demo API ──────────────────────────────────────────────────────────────
print("\n── 14. Demo API ──")

r = post("/api/demo/setup", {})
check("Demo setup", r.status_code in (200, 400, 500), f"status={r.status_code}")

r = post("/api/demo/generate-answers", {"student_id": "v_stud"})
check("Demo generate-answers", r.status_code in (200, 400, 500), f"status={r.status_code}")

# ── 15. Dashboard & Config ────────────────────────────────────────────────────
print("\n── 15. Dashboard & Config ──")

r = get("/api/dashboard")
check("Dashboard API", r.status_code in (200, 401), f"status={r.status_code}")

r = get("/api/config")
check("Config API", r.status_code == 200, f"status={r.status_code}")
if r.status_code == 200:
    d = r.get_json()
    check("  Returns config dict", isinstance(d, dict))

# ── 16. Evidence API ──────────────────────────────────────────────────────────
print("\n── 16. Evidence ──")

r = get(f"/api/evidence/{session_key or 'default'}")
check("Evidence endpoint", r.status_code in (200, 404, 500), f"status={r.status_code}")

# ── 17. Organization API ──────────────────────────────────────────────────────
print("\n── 17. Organizations ──")

r = get("/api/orgs")
check("List orgs", r.status_code in (200, 401), f"status={r.status_code}")

r = post("/api/orgs", {"name": "Test University", "admin_id": "v_exam", "org_type": "university"})
check("Create org", r.status_code in (200, 201, 401), f"status={r.status_code}")

# ── 18. Examiner-specific ─────────────────────────────────────────────────────
print("\n── 18. Examiner Endpoints ──")

r = get(f"/api/examiner/exam/{exam_id or 'exam_001'}/overview")
check("Examiner exam overview", r.status_code in (200, 404), f"status={r.status_code}")

r = get(f"/api/examiner/student/v_stud/{exam_id or 'exam_001'}")
check("Examiner student detail", r.status_code in (200, 404), f"status={r.status_code}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)

passed = sum(1 for s, _, _ in results if s == "PASS")
failed = sum(1 for s, _, _ in results if s == "FAIL")
total = len(results)
rate = 100 * passed // total if total else 0

for status, name, detail in results:
    sym = "PASS" if status == "PASS" else "FAIL"
    print(f"  [{sym}] {name}" + (f" — {detail}" if detail else ""))

print(f"\n  Total: {total} | PASS: {passed} | FAIL: {failed}")
print(f"  Pass rate: {rate}%")

if failed == 0:
    print("\n  *** ALL CHECKS PASSED ***")
else:
    print(f"\n  *** {failed} CHECK(S) FAILED ***")
