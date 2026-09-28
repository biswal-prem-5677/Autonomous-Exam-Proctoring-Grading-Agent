"""Test suite for world-class intelligence and auditability REST API endpoints."""

import json
import pytest
from app.api.server import app


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_grading_mathematical(client):
    """Test step-by-step mathematical grading endpoint."""
    payload = {
        "student_solution": "Governing equation: F = m*a\nSubstitution: m=10, a=9.8\nResult: 98 N",
        "expected_formula": "F = m*a",
        "expected_variables": {"m": 10, "a": 9.8},
        "expected_final_value": 98.0,
        "expected_units": "N",
        "marks": 10.0,
    }
    res = client.post("/api/grading/mathematical", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "result" in data
    r = data["result"]
    assert r["score"] > 8.0
    assert r["max_score"] == 10.0
    assert len(r["steps"]) >= 3


def test_api_grading_programming_safe(client):
    """Test programming grader endpoint with safe code and test execution."""
    code = """
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in seen:
            return [seen[diff], i]
        seen[n] = i
    return []
"""
    payload = {
        "code": code,
        "entry_function": "two_sum",
        "public_tests": [
            {"input": [[2, 7, 11, 15], 9], "expected": [0, 1], "name": "t1"},
            {"input": [[3, 2, 4], 6], "expected": [1, 2], "name": "t2"},
        ],
        "marks": 10.0,
    }
    res = client.post("/api/grading/programming", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    r = data["result"]
    assert r["syntax_valid"] is True
    assert r["security"]["is_safe"] is True
    assert r["metrics"]["passed_tests"] == 2
    assert r["score"] >= 9.0


def test_api_grading_programming_unsafe(client):
    """Test programming grader endpoint rejects unsafe code execution."""
    code = """
import os
def malicious():
    os.system("echo hacked")
"""
    payload = {
        "code": code,
        "entry_function": "malicious",
        "public_tests": [],
        "marks": 10.0,
    }
    res = client.post("/api/grading/programming", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    r = data["result"]
    assert r["security"]["is_safe"] is False
    assert r["score"] == 0.0


def test_api_grading_plagiarism(client):
    """Test cohort-wide MinHash/LSH plagiarism detection endpoint."""
    payload = {
        "exam_id": "midterm_exam",
        "threshold": 0.50,
        "submissions": {
            "student_1": "The quick brown fox jumps over the lazy dog repeatedly across the green park.",
            "student_2": "The quick brown fox jumps over the lazy dog repeatedly across the green meadow.",
            "student_3": "Quantum mechanics describes the fundamental physical properties of nature at subatomic scales.",
        },
    }
    res = client.post("/api/grading/plagiarism", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["total_submissions"] == 3
    assert len(data["flagged_pairs"]) >= 1
    flagged = data["flagged_pairs"][0]
    assert {flagged["student_a"], flagged["student_b"]} == {"student_1", "student_2"}


def test_api_grading_consensus(client):
    """Test multi-perspective rubric consensus endpoint."""
    payload = {
        "student_answer": (
            "Photosynthesis occurs within the chloroplasts of plant cells where chlorophyll absorbs sunlight. "
            "Light reactions split water molecules to generate ATP and NADPH, releasing oxygen. "
            "Subsequently, the Calvin cycle fixes carbon dioxide to produce glucose. "
            "Therefore, light energy is transformed into stable chemical energy because of enzymatic reactions."
        ),
        "rubric": {
            "keywords": ["chloroplast", "chlorophyll", "ATP", "Calvin cycle", "glucose", "oxygen"],
            "model_answer": "Photosynthesis in chloroplasts uses chlorophyll to absorb light, splitting water to make ATP and fixing carbon dioxide into glucose.",
        },
        "marks": 10.0,
    }
    res = client.post("/api/grading/consensus", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    r = data["result"]
    assert "consensus_status" in r
    assert r["final_score"] > 7.0
    assert "grader_totals" in r


def test_api_baseline_profile(client):
    """Test student baseline retrieval and sample updating."""
    # Update sample
    sample_payload = {
        "typing_wpm": 55.0,
        "mouse_velocity": 120.0,
        "mouse_clicks": 14,
        "face_present": 1.0,
        "face_count": 1,
        "gaze_off_screen_seconds": 1.2,
        "pause_duration_seconds": 0.5,
    }
    res_post = client.post("/api/baseline/test_student_42", json=sample_payload)
    assert res_post.status_code == 200
    d_post = res_post.get_json()
    assert d_post["status"] == "success"
    assert d_post["student_id"] == "test_student_42"
    assert d_post["sample_count"] >= 1

    # Get sample
    res_get = client.get("/api/baseline/test_student_42")
    assert res_get.status_code == 200
    d_get = res_get.get_json()
    assert d_get["student_id"] == "test_student_42"
    assert "maturity" in d_get
    assert "profile" in d_get


def test_api_audit_chain_and_verify(client):
    """Test cryptographic audit chain verification endpoint."""
    res_verify = client.post("/api/trace/audit-chain/test_session_99/verify")
    assert res_verify.status_code == 200
    d_verify = res_verify.get_json()
    assert d_verify["status"] == "success"
    assert d_verify["is_intact"] is True

    res_get = client.get("/api/trace/audit-chain/test_session_99")
    assert res_get.status_code == 200
    d_get = res_get.get_json()
    assert d_get["status"] == "success"
    assert "entries" in d_get


def test_api_replay_trace(client):
    """Test decision replay trace endpoint."""
    res = client.get("/api/trace/replay/session_nonexistent")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["total_steps"] == 0
    assert isinstance(data["timeline"], list)


def test_api_adversarial_benchmark(client):
    """Test adversarial red-team stress test benchmark endpoint."""
    payload = {"escalation_threshold": 0.65}
    res = client.post("/api/ml/adversarial/benchmark", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "report" in data
    rep = data["report"]
    assert "summary" in rep
    assert "calibration" in rep
    assert "matrix" in rep
