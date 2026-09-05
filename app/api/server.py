"""Flask REST API backend for the Exam Proctoring & Grading Agent.

Exposes all backend functionality as JSON endpoints for the HTML/JS frontend.
No external APIs — all computation is local.
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

# ── Backend imports ──────────────────────────────────────────────────────────
from src.models.models import (
    Question, QuestionType, Exam, ExamSession, AgentState, ProctoringEvent,
)
from src.exam.exam import ExamManager, ExamController
from src.exam.session import ExamSessionManager
from src.exam.questions import QuestionBank
from src.grading.scorer import GradingOrchestrator
from src.grading.long_answer import LongAnswerGrader
from src.agent.controller import ProctoringAgent, AgentPhase
from src.agent.risk import RiskEngine
from src.agent.memory import EvidenceMemory
from src.agent.state import AgentStateMachine
from src.features.behavioral import BehavioralFeatureExtractor
from src.features.temporal import TemporalFeatureAggregator
from src.ml.logistic import LogisticRegressionScratch
from src.ml.anomaly import AnomalyDetector
from src.ml.calibration import ModelCalibrator, ModelEvaluator
from src.ml.training_data import TrainingDataGenerator
from src.prediction.performance import PerformancePredictor
from src.prediction.knowledge import KnowledgeTracer
from src.prediction.difficulty import DifficultyEstimator
from src.reports.student_report import StudentReportGenerator
from src.reports.examiner_report import ExaminerReportGenerator
from src.reports.evidence import EvidenceLog, EvidenceSeverity
from src.utils.config import load_config

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__,
            static_folder="../static",
            template_folder="../templates")
CORS(app)

# Global state
_config = load_config()
_exam_manager = ExamManager(_config)
_active_sessions: Dict[str, Dict] = {}
_proctoring_agents: Dict[str, ProctoringAgent] = {}
_evidence_logs: Dict[str, EvidenceLog] = {}
_knowledge_tracers: Dict[str, KnowledgeTracer] = {}
_performance_predictors: Dict[str, PerformancePredictor] = {}
_difficulty_estimators: Dict[str, DifficultyEstimator] = {}
_grading_orchestrator = GradingOrchestrator()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES — serve the HTML frontend
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Landing / home page."""
    return render_template("index.html")


@app.route("/student")
def student_exam_old():
    """Student exam interface (legacy)."""
    return render_template("student_exam.html")


@app.route("/exam")
def student_exam_new():
    """New student exam interface with webcam."""
    return render_template("student/exam.html")


@app.route("/examiner")
def examiner_dashboard():
    """Examiner monitoring dashboard."""
    return render_template("examiner_dashboard.html")


@app.route("/examiner/grading")
def examiner_grading():
    """Examiner grading page."""
    return render_template("examiner/grading.html")


@app.route("/examiner/create_exam")
def examiner_create_exam():
    """Examiner create exam page."""
    return render_template("examiner/create_exam.html")


@app.route("/student/results")
def student_results():
    """Student results page."""
    return render_template("student/results.html")


@app.route("/dashboard")
def dashboard_page():
    """Unified dashboard (login required)."""
    return render_template("dashboard.html")


@app.route("/login")
def login_page():
    """Login page."""
    return render_template("auth/login.html")


@app.route("/register")
def register_page():
    """Registration page."""
    return render_template("auth/register.html")


@app.route("/sessions")
def sessions_page():
    """Proctoring sessions page."""
    return render_template("dashboard.html")


@app.route("/monitor")
def monitor_page():
    """Live monitor page."""
    return render_template("monitor/index.html")


@app.route("/training")
def training():
    """Model training page."""
    return render_template("training.html")


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH & INFO
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "version": _config.get("app", {}).get("version", "1.0.0"),
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "exam_engine": True,
            "grading": True,
            "proctoring_agent": True,
            "risk_engine": True,
            "evidence_log": True,
            "ml_models": True,
            "performance_prediction": True,
        }
    })


@app.route("/api/config")
def get_config():
    """Return safe (non-sensitive) configuration."""
    return jsonify({
        "exam": {
            "default_duration_minutes": _config.get("exam", {}).get("default_duration_minutes", 60),
            "auto_submit": _config.get("exam", {}).get("auto_submit", True),
        },
        "risk": _config.get("risk", {}),
        "grading": _config.get("grading", {}),
        "proctoring": _config.get("proctoring", {}),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAM MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/exams", methods=["GET"])
def list_exams():
    exams = []
    for eid, exam in _exam_manager._exams.items():
        exams.append({
            "exam_id": exam.exam_id,
            "title": exam.title,
            "duration_minutes": exam.duration_minutes,
            "question_count": len(exam.questions),
            "topic": exam.topic,
            "difficulty": exam.difficulty,
            "created_at": exam.created_at.isoformat(),
        })
    return jsonify({"exams": exams, "count": len(exams)})


@app.route("/api/exams", methods=["POST"])
def create_exam():
    data = request.get_json(force=True)
    exam_id = data.get("exam_id", f"exam_{int(time.time())}")
    exam = _exam_manager.create_exam(
        exam_id=exam_id,
        title=data.get("title", "Untitled Exam"),
        duration_minutes=data.get("duration_minutes", 60),
        questions=[],
    )
    exam.topic = data.get("topic", "")
    exam.difficulty = data.get("difficulty", "medium")
    return jsonify({
        "exam_id": exam.exam_id,
        "title": exam.title,
        "duration_minutes": exam.duration_minutes,
        "question_count": len(exam.questions),
        "status": "created",
    })


@app.route("/api/exams/<exam_id>", methods=["GET"])
def get_exam(exam_id):
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    return jsonify({
        "exam_id": exam.exam_id,
        "title": exam.title,
        "duration_minutes": exam.duration_minutes,
        "topic": exam.topic,
        "difficulty": exam.difficulty,
        "questions": [q.to_dict() for q in exam.questions],
        "question_count": len(exam.questions),
        "total_marks": sum(q.marks for q in exam.questions),
    })


@app.route("/api/exams/<exam_id>/questions", methods=["POST"])
def add_question(exam_id):
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    data = request.get_json(force=True)
    q = exam.add_question(
        qtype=data.get("type", "mcq"),
        text=data.get("text", ""),
        options=data.get("options"),
        correct_answer=data.get("correct_answer"),
        marks=data.get("marks", 1.0),
        keywords=data.get("keywords", []),
        tolerance=data.get("tolerance"),
        topic=data.get("topic", ""),
        difficulty=data.get("difficulty", "medium"),
        metadata=data.get("metadata", {}),
    )
    return jsonify({"question": q.to_dict(), "status": "added"})


@app.route("/api/exams/<exam_id>/questions/bulk", methods=["POST"])
def add_questions_bulk(exam_id):
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    data = request.get_json(force=True)
    questions_data = data.get("questions", [])
    added = []
    for qd in questions_data:
        q = exam.add_question(
            qtype=qd.get("type", "mcq"),
            text=qd.get("text", ""),
            options=qd.get("options"),
            correct_answer=qd.get("correct_answer"),
            marks=qd.get("marks", 1.0),
            keywords=qd.get("keywords", []),
            tolerance=qd.get("tolerance"),
            topic=qd.get("topic", ""),
            difficulty=qd.get("difficulty", "medium"),
        )
        added.append(q.to_dict())
    return jsonify({"questions": added, "count": len(added)})


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAM SESSIONS (student taking exam)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/exam-sessions", methods=["POST"])
def create_session():
    """Start a new exam session for a student."""
    data = request.get_json(force=True)
    student_id = data.get("student_id", f"student_{int(time.time())}")
    exam_id = data.get("exam_id")
    duration = data.get("duration_minutes", 60)

    if not exam_id:
        return jsonify({"error": "exam_id required"}), 400

    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    controller = _exam_manager.get_or_create_controller(
        student_id, exam_id, duration
    )
    session = controller.session
    if not session:
        return jsonify({"error": "Failed to create session"}), 500

    controller.start_exam()

    session_key = f"{student_id}:{exam_id}"
    _active_sessions[session_key] = {
        "student_id": student_id,
        "exam_id": exam_id,
        "controller": controller,
        "started_at": datetime.now().isoformat(),
    }

    # Initialize proctoring agent for this session
    agent = ProctoringAgent(_config)
    _proctoring_agents[session_key] = agent

    # Initialize evidence log
    _evidence_logs[session_key] = EvidenceLog(session_id=session.session_id)

    # Initialize knowledge tracer
    _knowledge_tracers[session_key] = KnowledgeTracer()

    # Initialize difficulty estimator
    _difficulty_estimators[session_key] = DifficultyEstimator()

    return jsonify({
        "session_id": session.session_id,
        "student_id": student_id,
        "exam_id": exam_id,
        "duration_minutes": duration,
        "started_at": session.start_time.isoformat(),
        "questions": [q.to_dict() for q in exam.questions],
        "total_marks": sum(q.marks for q in exam.questions),
        "status": "started",
    })


@app.route("/api/exam-sessions/<student_id>/<exam_id>/status", methods=["GET"])
def session_status(student_id, exam_id):
    session_key = f"{student_id}:{exam_id}"
    session_info = _active_sessions.get(session_key)
    if not session_info:
        return jsonify({"error": "Session not found"}), 404

    controller = session_info["controller"]
    status = controller.get_session_status()

    # Add questions if available
    exam = _exam_manager.get_exam(exam_id)
    if exam:
        status["questions"] = [q.to_dict() for q in exam.questions]
        status["total_marks"] = sum(q.marks for q in exam.questions)

    return jsonify(status)


@app.route("/api/exam-sessions/<student_id>/<exam_id>/answers", methods=["POST"])
def submit_answer(student_id, exam_id):
    """Submit an answer for a question."""
    session_key = f"{student_id}:{exam_id}"
    session_info = _active_sessions.get(session_key)
    if not session_info:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(force=True)
    question_id = data.get("question_id")
    answer = data.get("answer")

    if question_id is None:
        return jsonify({"error": "question_id required"}), 400

    controller = session_info["controller"]
    controller.submit_answer(question_id, answer)

    # Update knowledge tracer
    kt = _knowledge_tracers.get(session_key)
    if kt:
        exam = _exam_manager.get_exam(exam_id)
        if exam:
            for q in exam.questions:
                if q.id == question_id:
                    kt.update(q.metadata.get("topic", "general"), 0.5)
                    break

    return jsonify({
        "question_id": question_id,
        "answer": answer,
        "status": "saved",
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/exam-sessions/<student_id>/<exam_id>/submit", methods=["POST"])
def end_session(student_id, exam_id):
    """End exam session and trigger grading."""
    session_key = f"{student_id}:{exam_id}"
    session_info = _active_sessions.get(session_key)
    if not session_info:
        return jsonify({"error": "Session not found"}), 404

    controller = session_info["controller"]
    session = controller.end_exam()

    return jsonify({
        "session_id": session.session_id,
        "status": "submitted",
        "answers_count": len(session.answers),
        "events_count": len(session.events),
        "ended_at": session.end_time.isoformat() if session.end_time else None,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADING
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/grade", methods=["POST"])
def grade_answers():
    """Grade all submitted answers for an exam."""
    data = request.get_json(force=True)
    exam_id = data.get("exam_id")
    answers = data.get("answers", {})
    reference_answers = data.get("reference_answers", {})
    student_id = data.get("student_id", "unknown")

    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    if not answers:
        return jsonify({
            "per_question": {},
            "total_score": 0.0,
            "total_max": 0.0,
            "percentage": 0.0,
            "message": "No answers submitted",
        })

    results = _grading_orchestrator.grade_all(
        exam.questions, answers, reference_answers
    )

    # Build per-question details
    per_question = {}
    for q in exam.questions:
        qid = q.id
        if qid in results.get("per_question", {}):
            per_question[qid] = {
                "question_id": qid,
                "question_text": q.text,
                "question_type": q.type.value,
                "marks": q.marks,
                "student_answer": answers.get(qid, ""),
                "correct_answer": q.correct_answer,
                **results["per_question"][qid],
            }

    return jsonify({
        "student_id": student_id,
        "exam_id": exam_id,
        "per_question": per_question,
        "total_score": results.get("total_score", 0.0),
        "total_max": results.get("total_max", 0.0),
        "percentage": results.get("percentage", 0.0),
        "graded_at": datetime.now().isoformat(),
    })


@app.route("/api/grade/single", methods=["POST"])
def grade_single_question():
    """Grade a single question answer."""
    data = request.get_json(force=True)
    qtype = data.get("type", "mcq")
    student_answer = data.get("student_answer", "")
    correct_answer = data.get("correct_answer")
    marks = data.get("marks", 1.0)
    keywords = data.get("keywords", [])
    reference_answer = data.get("reference_answer", "")
    tolerance = data.get("tolerance", 0.02)

    q = Question(
        id="q_temp",
        type=QuestionType(qtype),
        text="",
        correct_answer=correct_answer,
        marks=marks,
        keywords=keywords,
        tolerance=tolerance,
    )

    grader = _grading_orchestrator
    if qtype == "mcq":
        result = grader.mcq_grader.grade(q, student_answer)
    elif qtype == "numerical":
        result = grader.numerical_grader.grade(q, student_answer)
    elif qtype == "short_answer":
        tfidf = grader.tfidf_scorer.cosine_similarity(reference_answer, student_answer)
        kw = grader.keyword_scorer.score(reference_answer, student_answer, keywords)
        score = 0.5 * tfidf + 0.5 * kw["coverage"]
        result = {
            "score": round(score * marks, 2),
            "max_score": marks,
            "similarity": round(tfidf, 4),
            "keyword_coverage": round(kw["coverage"], 4),
            "keywords_found": kw["found"],
            "keywords_missing": kw["missing"],
            "method": "tfidf_cosine",
        }
    elif qtype == "long_answer":
        result = grader.long_answer_grader.grade(
            student_answer, reference_answer, keywords=keywords, marks=marks
        )
        result = {
            "score": result.final_score,
            "max_score": result.max_score,
            "semantic_score": round(result.semantic_score, 4),
            "keyword_score": round(result.keyword_score, 4),
            "structure_score": round(result.structure_score, 4),
            "coverage_score": round(result.coverage_score, 4),
            "details": result.details,
            "method": "multi_signal",
        }
    else:
        result = {"score": 0.0, "max_score": marks, "method": "unknown"}

    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  PROCTORING
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/proctor/process", methods=["POST"])
def process_proctor_signals():
    """Process a batch of proctoring signals through the agent loop."""
    data = request.get_json(force=True)
    session_key = data.get("session_key")
    signals = data.get("signals", {})

    if not session_key:
        return jsonify({"error": "session_key required"}), 400

    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({"error": "No active proctoring agent for this session"}), 404

    result = agent.process_signals(signals)

    # Record evidence
    evidence_log = _evidence_logs.get(session_key)
    if evidence_log:
        for evt in result.get("events_detected", []):
            evidence_log.add_from_risk_event(
                event_type=evt["type"],
                confidence=evt["confidence"],
                description=f"Detected {evt['type']} from {evt['source']} signal",
                signal_source=evt["source"],
            )

    return jsonify(result)


@app.route("/api/proctor/status/<student_id>/<exam_id>", methods=["GET"])
def proctor_status(student_id, exam_id):
    """Get current proctoring status for a session."""
    session_key = f"{student_id}:{exam_id}"
    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({"status": "no_agent", "message": "Proctoring not initialized"})

    status = agent.get_status()
    evidence_log = _evidence_logs.get(session_key)
    if evidence_log:
        status["evidence_summary"] = evidence_log.to_dict()

    return jsonify(status)


@app.route("/api/proctor/history/<student_id>/<exam_id>", methods=["GET"])
def proctor_history(student_id, exam_id):
    """Get proctoring event history."""
    session_key = f"{student_id}:{exam_id}"
    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({"history": []})

    return jsonify({
        "history": agent.get_history(),
        "count": len(agent.get_history()),
    })


@app.route("/api/proctor/evidence/<student_id>/<exam_id>", methods=["GET"])
def get_evidence(student_id, exam_id):
    """Get evidence log for a session."""
    session_key = f"{student_id}:{exam_id}"
    evidence_log = _evidence_logs.get(session_key)
    if not evidence_log:
        return jsonify({"timeline": [], "summary": {}})

    severity = request.args.get("severity")
    entries = evidence_log.get_entries(severity=severity) if severity else evidence_log.get_entries()

    return jsonify({
        "timeline": [e.to_dict() for e in entries],
        "summary": evidence_log.to_dict(),
    })


@app.route("/api/proctor/reset/<student_id>/<exam_id>", methods=["POST"])
def reset_proctor(student_id, exam_id):
    """Reset proctoring state for a new session."""
    session_key = f"{student_id}:{exam_id}"
    agent = _proctoring_agents.get(session_key)
    if agent:
        agent.reset()

    if session_key in _evidence_logs:
        _evidence_logs[session_key].clear()

    return jsonify({"status": "reset"})


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMINER DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/examiner/exam/<exam_id>/overview", methods=["GET"])
def examiner_overview(exam_id):
    """Get overview of all students for an exam."""
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    students = []
    active_count = 0
    completed_count = 0
    review_count = 0
    total_risk = 0.0
    total_score = 0.0
    scored_count = 0

    for session_key, session_info in _active_sessions.items():
        if session_info["exam_id"] != exam_id:
            continue

        student_id = session_info["student_id"]
        controller = session_info["controller"]
        status = controller.get_session_status()

        # Get risk from evidence log
        evidence_log = _evidence_logs.get(session_key)
        evidence_summary = evidence_log.to_dict() if evidence_log else {}

        # Count answers per question
        answers_count = status.get("answers_count", 0)
        total_qs = len(exam.questions)
        progress = (answers_count / total_qs * 100) if total_qs > 0 else 0

        risk_score = status.get("risk_score", 0.0)
        state = status.get("state", "NORMAL")
        remaining = status.get("remaining_seconds", 0)

        if remaining > 0:
            active_count += 1
        else:
            completed_count += 1

        if state in ("HIGH_RISK", "REVIEW_REQUIRED"):
            review_count += 1

        total_risk += risk_score
        if progress >= 100:
            # Has completed — compute score from grading
            graded = _grade_session_internal(student_id, exam_id, exam)
            score_pct = graded.get("percentage", 0.0)
            total_score += score_pct
            scored_count += 1
        else:
            progress_val = round(progress, 1)

        students.append({
            "student_id": student_id,
            "session_id": status.get("session_id", ""),
            "state": state,
            "risk_score": round(risk_score, 4),
            "progress": round(progress, 1),
            "answers_count": answers_count,
            "total_questions": total_qs,
            "remaining_seconds": remaining,
            "elapsed_seconds": status.get("elapsed_seconds", 0),
            "events_count": status.get("events_count", 0),
            "evidence_summary": evidence_summary,
            "last_activity": session_info.get("started_at", ""),
        })

    avg_risk = (total_risk / len(students)) if students else 0.0
    avg_score = (total_score / scored_count) if scored_count > 0 else 0.0

    return jsonify({
        "exam_id": exam_id,
        "exam_title": exam.title,
        "total_students": len(students),
        "active_students": active_count,
        "completed_students": completed_count,
        "review_required": review_count,
        "average_risk": round(avg_risk, 4),
        "average_score": round(avg_score, 2),
        "students": students,
    })


@app.route("/api/examiner/student/<student_id>/<exam_id>", methods=["GET"])
def student_detail(student_id, exam_id):
    """Full detail for a single student."""
    session_key = f"{student_id}:{exam_id}"
    session_info = _active_sessions.get(session_key)

    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    controller = None
    status = {}
    if session_info:
        controller = session_info["controller"]
        status = controller.get_session_status()

    # Risk history
    agent = _proctoring_agents.get(session_key)
    history = agent.get_history() if agent else []

    # Evidence
    evidence_log = _evidence_logs.get(session_key)
    evidence_data = evidence_log.to_dict() if evidence_log else {}

    # Grading results
    if controller and controller.session:
        answers = controller.session.answers
    else:
        answers = {}
    grading = _grade_session_internal(student_id, exam_id, exam, answers)

    # Build risk timeline from history
    risk_timeline = []
    for h in history:
        risk_timeline.append({
            "timestamp": datetime.now().isoformat(),
            "iteration": h.get("iteration"),
            "risk_score": h.get("risk_score"),
            "state": h.get("state"),
            "events": h.get("events_detected", []),
        })

    # Contributing signals
    contributing_signals = _compute_contributing_signals(evidence_data)

    # Signal activity breakdown
    signal_activity = _compute_signal_activity(history)

    return jsonify({
        "student_id": student_id,
        "exam_id": exam_id,
        "exam_title": exam.title,
        "session_status": status,
        "risk_timeline": risk_timeline,
        "risk_score": status.get("risk_score", 0.0),
        "state": status.get("state", "NORMAL"),
        "contributing_signals": contributing_signals,
        "signal_activity": signal_activity,
        "evidence": evidence_data,
        "grading": grading,
        "events_count": status.get("events_count", 0),
        "answers_count": status.get("answers_count", 0),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/reports/student", methods=["POST"])
def generate_student_report():
    data = request.get_json(force=True)
    student_id = data.get("student_id", "unknown")
    exam_id = data.get("exam_id")
    grading_results = data.get("grading_results", {})
    proctoring_summary = data.get("proctoring_summary", {})

    session_key = f"{student_id}:{exam_id}"
    kt = _knowledge_tracers.get(session_key)
    knowledge_state = kt.get_all_mastery() if kt else {}

    predictor = _performance_predictors.get(session_key)
    prediction = predictor.predict(grading_results) if predictor else {}

    generator = StudentReportGenerator()
    report = generator.generate(
        grading_results=grading_results,
        proctoring_summary={**proctoring_summary, "student_id": student_id},
        prediction=prediction,
        knowledge_state=knowledge_state,
    )

    return jsonify(report)


@app.route("/api/reports/examiner", methods=["POST"])
def generate_examiner_report():
    data = request.get_json(force=True)
    exam_id = data.get("exam_id")
    student_reports = data.get("student_reports", [])

    generator = ExaminerReportGenerator()
    report = generator.generate(
        exam_id=exam_id,
        student_reports=student_reports,
        exam_config=_config.get("exam", {}),
    )

    return jsonify(report)


# ═══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/predict/performance", methods=["POST"])
def predict_performance():
    data = request.get_json(force=True)
    session_key = data.get("session_key")
    features_list = data.get("features", [])

    if not session_key:
        return jsonify({"error": "session_key required"}), 400

    # Convert list to numpy array for prediction
    import numpy as np
    features = np.array(features_list) if features_list else np.array([])

    predictor = _performance_predictors.get(session_key)
    if not predictor:
        predictor = PerformancePredictor()
        _performance_predictors[session_key] = predictor

    # If we have features but no model trained, return a placeholder
    if features.size > 0 and not hasattr(predictor, '_fitted'):
        return jsonify({"error": "Model not trained. Provide training data first."}), 400

    if features.size > 0:
        result = predictor.predict(features)
        return jsonify({"prediction": result.tolist() if hasattr(result, 'tolist') else result})
    return jsonify({})


@app.route("/api/predict/difficulty", methods=["POST"])
def estimate_difficulty():
    data = request.get_json(force=True)
    session_key = data.get("session_key")
    question_id = data.get("question_id", "unknown")
    correct_count = data.get("correct_count", 0)
    total_count = data.get("total_count", 1)

    if not session_key:
        return jsonify({"error": "session_key required"}), 400

    estimator = _difficulty_estimators.get(session_key)
    if not estimator:
        estimator = DifficultyEstimator()
        _difficulty_estimators[session_key] = estimator

    difficulty = estimator.estimate_from_responses(question_id, correct_count, total_count)

    return jsonify({
        "question_id": question_id,
        "difficulty": round(difficulty, 4),
        "correct_count": correct_count,
        "total_count": total_count,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/train/generate-data", methods=["POST"])
def generate_training_data():
    data = request.get_json(force=True) or {}
    n_normal = data.get("n_normal", 200)
    n_suspicious = data.get("n_suspicious", 100)
    scenarios = data.get("scenarios")

    gen = TrainingDataGenerator(seed=42)
    X, y = gen.generate_dataset(
        n_normal=n_normal,
        n_suspicious=n_suspicious,
    )

    # Define feature names (22 features as per training_data.py)
    feature_names = [
        "face_present", "face_confidence", "face_count", "absence_ratio",
        "head_yaw", "head_pitch", "head_roll",
        "audio_rms", "audio_db", "audio_zcr", "audio_threshold_flag", "spectral_centroid",
        "keyboard_idle_flag", "keystroke_rate", "key_hold_mean", "key_latency_mean",
        "mouse_speed", "mouse_distance", "click_rate", "mouse_idle_flag",
        "multi_face_events", "session_progress"
    ]

    return jsonify({
        "total_samples": len(X),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "class_distribution": {
            "normal": int(sum(1 for l in y if l == 0)),
            "suspicious": int(sum(1 for l in y if l == 1)),
        },
        "sample_preview": X[:5].tolist() if hasattr(X, "tolist") else X[:5],
    })


@app.route("/api/train/models", methods=["POST"])
def train_models():
    """Train logistic regression and anomaly models."""
    import numpy as np
    import random
    data = request.get_json(force=True) or {}
    n_normal = data.get("n_normal", 200)
    n_suspicious = data.get("n_suspicious", 100)

    gen = TrainingDataGenerator(seed=42)
    X, y = gen.generate_dataset(n_normal=n_normal, n_suspicious=n_suspicious)

    feature_names = [
        "face_present", "face_confidence", "face_count", "absence_ratio",
        "head_yaw", "head_pitch", "head_roll",
        "audio_rms", "audio_db", "audio_zcr", "audio_threshold_flag", "spectral_centroid",
        "keyboard_idle_flag", "keystroke_rate", "key_hold_mean", "key_latency_mean",
        "mouse_speed", "mouse_distance", "click_rate", "mouse_idle_flag",
        "multi_face_events", "session_progress"
    ]

    # Train/test split (70/30)
    n = len(X)
    split = int(0.7 * n)
    indices = list(range(n))
    import random
    random.seed(42)
    random.shuffle(indices)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Logistic Regression
    lr = LogisticRegressionScratch(
        learning_rate=0.01,
        max_iterations=1000,
        regularization=0.1,
    )
    lr.fit(X_train, y_train)

    # Anomaly Detector
    anomaly = AnomalyDetector(mahalanobis_confidence=0.95)
    anomaly.fit(X_train[y_train == 0])  # Fit on normal class only

    # Evaluate
    evaluator = ModelEvaluator()

    lr_preds = lr.predict(X_test)
    lr_probas = lr.predict_proba(X_test)
    lr_metrics = evaluator.evaluate_classifier(y_test, lr_preds, lr_probas)

    anomaly_scores = anomaly.score(X_test)
    anomaly_threshold = float(np.percentile(anomaly.score(X_train[y_train == 0]), 95))
    anomaly_preds = (anomaly_scores > anomaly_threshold).astype(int)
    anomaly_metrics = evaluator.evaluate_classifier(y_test, anomaly_preds, anomaly_scores)

    return jsonify({
        "logistic_regression": {
            "metrics": lr_metrics,
            "feature_weights": dict(zip(feature_names[:len(lr.weights)], lr.weights.tolist())),
        },
        "anomaly_detection": {
            "metrics": anomaly_metrics,
            "threshold": round(anomaly_threshold, 4),
            "method": "mahalanobis",
        },
        "dataset": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        },
        "confusion_matrix_lr": lr_metrics.get("confusion_matrix"),
        "confusion_matrix_anomaly": anomaly_metrics.get("confusion_matrix"),
    })


@app.route("/api/train/evaluate", methods=["POST"])
def run_evaluation():
    """Run full evaluation with ablation study."""
    data = request.get_json(force=True) or {}
    n_normal = data.get("n_normal", 200)
    n_suspicious = data.get("n_suspicious", 100)
    run_ablation = data.get("run_ablation", True)

    gen = TrainingDataGenerator(seed=42)
    X, y = gen.generate_dataset(n_normal=n_normal, n_suspicious=n_suspicious)

    feature_names = [
        "face_present", "face_confidence", "face_count", "absence_ratio",
        "head_yaw", "head_pitch", "head_roll",
        "audio_rms", "audio_db", "audio_zcr", "audio_threshold_flag", "spectral_centroid",
        "keyboard_idle_flag", "keystroke_rate", "key_hold_mean", "key_latency_mean",
        "mouse_speed", "mouse_distance", "click_rate", "mouse_idle_flag",
        "multi_face_events", "session_progress"
    ]

    n = len(X)
    split = int(0.7 * n)
    import random
    random.seed(42)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    evaluator = ModelEvaluator()
    results = {}

    # Full model
    lr_full = LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000)
    lr_full.fit(X_train, y_train)
    full_preds = lr_full.predict(X_test)
    full_probas = lr_full.predict_proba(X_test)
    results["full_multimodal"] = evaluator.evaluate_classifier(y_test, full_preds, full_probas)

    if run_ablation:
        ablation_configs = [
            ("vision_only", [0, 1, 2, 3, 4]),
            ("vision_keyboard", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ("vision_keyboard_mouse", list(range(14))),
            ("vision_keyboard_mouse_browser", list(range(20))),
        ]

        for name, feature_indices in ablation_configs:
            if max(feature_indices) < X_train.shape[1]:
                X_train_sub = X_train[:, feature_indices]
                X_test_sub = X_test[:, feature_indices]
                lr_ab = LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000)
                lr_ab.fit(X_train_sub, y_train)
                preds_ab = lr_ab.predict(X_test_sub)
                probas_ab = lr_ab.predict_proba(X_test_sub)
                results[name] = evaluator.evaluate_classifier(y_test, preds_ab, probas_ab)

    return jsonify({
        "ablation_results": results,
        "feature_names": feature_names,
        "dataset_info": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "total_features": len(feature_names),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO / SAMPLE DATA
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/demo/setup", methods=["POST"])
def demo_setup():
    """Set up a complete demo exam with sample data."""
    data = request.get_json(force=True) or {}
    exam_id = data.get("exam_id", f"demo_{int(time.time())}")

    # Create exam with sample questions
    exam = _exam_manager.create_exam(
        exam_id=exam_id,
        title=data.get("title", "Mathematics Mid-Term Examination"),
        duration_minutes=data.get("duration_minutes", 30),
        questions=[],
    )
    exam.topic = "Mathematics"
    exam.difficulty = "medium"

    # Add sample questions
    sample_questions = data.get("questions") or [
        {"type": "mcq", "text": "What is 5 + 3?", "options": ["6", "7", "8", "9"],
         "correct_answer": "8", "marks": 2, "topic": "Arithmetic", "difficulty": "easy"},
        {"type": "mcq", "text": "Solve: 2x = 10, find x",
         "options": ["3", "4", "5", "6"], "correct_answer": "5", "marks": 3,
         "topic": "Algebra", "difficulty": "medium"},
        {"type": "numerical", "text": "What is the area of a circle with radius 7? (use π ≈ 3.14)",
         "correct_answer": 153.86, "marks": 5, "tolerance": 0.5, "topic": "Geometry", "difficulty": "medium"},
        {"type": "numerical", "text": "Calculate: sqrt(144) + 12",
         "correct_answer": 24, "marks": 3, "tolerance": 0.1, "topic": "Arithmetic", "difficulty": "easy"},
        {"type": "short_answer",
         "text": "Define Newton's First Law of Motion.",
         "correct_answer": "An object at rest stays at rest and an object in motion stays in motion with the same speed and direction unless acted upon by an unbalanced force.",
         "keywords": ["inertia", "rest", "motion", "force", "unbalanced"],
         "marks": 5, "topic": "Physics", "difficulty": "easy"},
        {"type": "short_answer",
         "text": "What is the derivative of x^2?",
         "correct_answer": "The derivative of x^2 with respect to x is 2x.",
         "keywords": ["derivative", "2x", "power rule", "differentiation"],
         "marks": 4, "topic": "Calculus", "difficulty": "easy"},
        {"type": "mcq", "text": "Which of the following is a prime number?",
         "options": ["4", "6", "9", "11"], "correct_answer": "11", "marks": 2,
         "topic": "Number Theory", "difficulty": "easy"},
        {"type": "numerical", "text": "If f(x) = 3x^2, find f(4).",
         "correct_answer": 48, "marks": 4, "tolerance": 1.0, "topic": "Calculus", "difficulty": "medium"},
        {"type": "short_answer",
         "text": "Explain the Pythagorean theorem.",
         "correct_answer": "In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides: a^2 + b^2 = c^2.",
         "keywords": ["right triangle", "hypotenuse", "square", "a^2", "b^2", "c^2"],
         "marks": 5, "topic": "Geometry", "difficulty": "easy"},
        {"type": "long_answer",
         "text": "Explain the process of photosynthesis and its importance to life on Earth. Include the chemical equation and describe the role of chlorophyll.",
         "correct_answer": "Photosynthesis is the process by which green plants convert light energy into chemical energy stored in glucose. Chlorophyll absorbs sunlight, which drives the reaction: 6CO2 + 6H2O + light → C6H12O6 + 6O2. This process is vital as it produces oxygen for respiration and forms the base of most food chains.",
         "keywords": ["photosynthesis", "chlorophyll", "glucose", "carbon dioxide", "water", "oxygen", "sunlight", "energy"],
         "marks": 10, "topic": "Biology", "difficulty": "medium"},
    ]

    for sq in sample_questions:
        qtype = sq.pop("type", "mcq")
        text = sq.pop("text", "")
        exam.add_question(qtype=qtype, text=text, **sq)

    # Create sample students
    students = data.get("students") or [
        {"student_id": "STU001", "name": "Alice Johnson"},
        {"student_id": "STU002", "name": "Bob Smith"},
        {"student_id": "STU003", "name": "Carol Davis"},
    ]

    return jsonify({
        "exam_id": exam_id,
        "title": exam.title,
        "duration_minutes": exam.duration_minutes,
        "question_count": len(exam.questions),
        "total_marks": sum(q.marks for q in exam.questions),
        "questions": [q.to_dict() for q in exam.questions],
        "students": students,
        "status": "demo_ready",
    })


@app.route("/api/demo/generate-answers", methods=["POST"])
def generate_demo_answers():
    """Generate sample student answers for demo."""
    data = request.get_json(force=True) or {}
    exam_id = data.get("exam_id")
    student_id = data.get("student_id", "STU001")

    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    # Sample answers — mix of correct, partially correct, and incorrect
    sample_answers = {
        "q1": "8",
        "q2": "5",
        "q3": "154.0",
        "q4": "24",
        "q5": "Newton's First Law states that an object at rest stays at rest and an object in motion stays in motion unless acted upon by an external force. This is also known as the law of inertia.",
        "q6": "The derivative of x^2 is 2x. This follows from the power rule of differentiation.",
        "q7": "11",
        "q8": "48",
        "q9": "The Pythagorean theorem states that in a right-angled triangle, a squared plus b squared equals c squared, where c is the hypotenuse.",
        "q10": "Photosynthesis is how plants make food. They use sunlight, water, and carbon dioxide to produce glucose and oxygen. Chlorophyll is the green pigment that captures light energy. The chemical equation is 6CO2 + 6H2O -> C6H12O6 + 6O2. This process is important because it provides oxygen for animals to breathe and forms the base of the food chain.",
    }

    return jsonify({
        "student_id": student_id,
        "exam_id": exam_id,
        "answers": sample_answers,
        "reference_answers": {
            "q5": exam.questions[4].correct_answer,
            "q6": exam.questions[5].correct_answer,
            "q9": exam.questions[8].correct_answer,
            "q10": exam.questions[9].correct_answer,
        } if len(exam.questions) >= 10 else {},
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATION & AUTHORIZATION
# ============================================================

def _get_auth_db():
    from src.database.auth import get_auth_db
    return get_auth_db()

_vision_analyzers: Dict[str, Dict] = {}

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(force=True)
    required = ["username", "email", "full_name", "password", "role"]
    for f in required:
        if not data.get(f):
            return jsonify({"error": f + " is required"}), 400
    auth_db = _get_auth_db()
    if auth_db.get_user_by_username(data["username"]):
        return jsonify({"error": "Username already exists"}), 409
    if auth_db.get_user_by_email(data["email"]):
        return jsonify({"error": "Email already exists"}), 409
    role_str = data["role"]
    if role_str not in ("super_admin", "admin", "examiner", "student", "viewer"):
        return jsonify({"error": "Invalid role"}), 400
    user_id = auth_db.create_user(
        username=data["username"], email=data["email"],
        full_name=data["full_name"], password=data["password"],
        role=role_str, org_id=data.get("org_id"),
        preferences=data.get("preferences", {}),
    )
    token = auth_db.create_session(user_id)
    user = auth_db.get_user(user_id)
    return jsonify({"user": user, "token": token, "message": "Registration successful"}), 201


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(force=True)
    username = data.get("username", "")
    password = data.get("password", "")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    auth_db = _get_auth_db()
    user = auth_db.verify_password(username, password)
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
    token = auth_db.create_session(user["user_id"])
    return jsonify({"user": user, "token": token, "message": "Login successful"})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        _get_auth_db().delete_session(token)
    return jsonify({"message": "Logged out"})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return jsonify({"error": "No token provided"}), 401
    user = _get_auth_db().get_session_user(token)
    if not user:
        return jsonify({"error": "Invalid or expired session"}), 401
    return jsonify({"user": user})


@app.route("/api/auth/users", methods=["GET"])
def list_users_api():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user or user.get("role") not in ("super_admin", "admin", "examiner"):
        return jsonify({"error": "Unauthorized"}), 401
    org_id = request.args.get("org_id")
    role_filter = request.args.get("role")
    users = _get_auth_db().list_users(org_id=org_id, role=role_filter)
    return jsonify({"users": users, "count": len(users)})


@app.route("/api/auth/users", methods=["POST"])
def create_user_api():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    me = _get_auth_db().get_session_user(token) if token else None
    if not me or me.get("role") not in ("super_admin", "admin"):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True)
    required = ["username", "email", "full_name", "password", "role"]
    for f in required:
        if not data.get(f):
            return jsonify({"error": f + " is required"}), 400
    auth_db = _get_auth_db()
    if auth_db.get_user_by_username(data["username"]):
        return jsonify({"error": "Username already exists"}), 409
    user_id = auth_db.create_user(
        username=data["username"], email=data["email"],
        full_name=data["full_name"], password=data["password"],
        role=data["role"], org_id=data.get("org_id") or me.get("org_id"),
    )
    return jsonify({"user": auth_db.get_user(user_id)}), 201


@app.route("/api/auth/users/<user_id>", methods=["PUT"])
def update_user_api(user_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    me = _get_auth_db().get_session_user(token) if token else None
    if not me:
        return jsonify({"error": "Unauthorized"}), 401
    if me["user_id"] != user_id and me.get("role") not in ("super_admin", "admin"):
        return jsonify({"error": "Forbidden"}), 403
    data = request.get_json(force=True)
    allowed = {"full_name", "email", "avatar_url", "preferences"}
    updates = {k: v for k, v in data.items() if k in allowed and v is not None}
    _get_auth_db().update_user(user_id, **updates)
    return jsonify({"user": _get_auth_db().get_user(user_id)})


@app.route("/api/auth/users/<user_id>", methods=["DELETE"])
def delete_user_api(user_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    me = _get_auth_db().get_session_user(token) if token else None
    if not me or me.get("role") not in ("super_admin", "admin"):
        return jsonify({"error": "Unauthorized"}), 401
    _get_auth_db().delete_user(user_id)
    return jsonify({"message": "User deleted"})


@app.route("/api/orgs", methods=["GET"])
def list_orgs_api():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    orgs = _get_auth_db().list_orgs()
    return jsonify({"organizations": orgs, "count": len(orgs)})


@app.route("/api/orgs", methods=["POST"])
def create_org_api():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    me = _get_auth_db().get_session_user(token) if token else None
    if not me or me.get("role") not in ("super_admin", "admin"):
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True)
    org_id = _get_auth_db().create_org(
        name=data.get("name", "Untitled Org"), admin_id=me["user_id"],
        org_type=data.get("org_type", "university"), settings=data.get("settings", {}),
    )
    return jsonify({"organization": _get_auth_db().get_org(org_id)}), 201


@app.route("/api/orgs/<org_id>", methods=["GET"])
def get_org_api(org_id):
    org = _get_auth_db().get_org(org_id)
    if not org:
        return jsonify({"error": "Organization not found"}), 404
    return jsonify({"organization": org})


@app.route("/api/dashboard", methods=["GET"])
def dashboard_api():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    active_exams = []
    for eid, exam in _exam_manager._exams.items():
        active_exams.append({
            "exam_id": eid, "title": exam.title,
            "topic": exam.topic, "difficulty": exam.difficulty,
            "question_count": len(exam.questions),
            "total_marks": sum(q.marks for q in exam.questions),
        })
    total_sessions = len(_proctoring_agents)
    active_count = 0
    review_count = 0
    risk_scores = []
    recent_evidence = []
    for skey, agent in _proctoring_agents.items():
        status = agent.get_status()
        state = status.get("state", "NORMAL")
        if state == "NORMAL" and status.get("iteration", 0) > 0:
            active_count += 1
        elif state in ("SUSPICIOUS", "HIGH_RISK", "REVIEW_REQUIRED"):
            review_count += 1
        risk_scores.append(status.get("current_risk", 0.0))
        for h in agent.get_history()[-3:]:
            for evt in h.get("events_detected", []):
                recent_evidence.append({
                    "session_id": skey, "event_type": evt.get("type", ""),
                    "source": evt.get("source", ""), "confidence": evt.get("confidence", 0),
                    "state": h.get("state", "NORMAL"),
                })
    recent_evidence.sort(key=lambda x: x.get("event_type", ""), reverse=True)
    recent_evidence = recent_evidence[:20]
    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    return jsonify({
        "active_exams_count": len(active_exams), "total_sessions": total_sessions,
        "active_sessions": active_count, "review_sessions": review_count,
        "completed_sessions": 0, "avg_risk_score": round(avg_risk, 4),
        "avg_risk_percent": round(avg_risk * 100, 1),
        "active_exams": active_exams, "recent_evidence": recent_evidence,
        "model_status": {
            "logistic_regression": True, "anomaly_detector": True,
            "risk_engine": True, "temporal_fusion": True,
        },
        "system_status": "online",
    })


@app.route("/api/vision/analyze", methods=["POST"])
def vision_analyze():
    data = request.get_json(force=True)
    session_key = data.get("session_key", "default")
    if session_key not in _vision_analyzers:
        from src.vision.face import FaceTracker
        from src.vision.landmarks import LandmarkAnalyzer
        from src.vision.movement import MovementDetector
        _vision_analyzers[session_key] = {
            "face_tracker": FaceTracker(), "landmark_analyzer": LandmarkAnalyzer(),
            "movement_detector": MovementDetector(),
        }
    analyzers = _vision_analyzers[session_key]
    face_present = data.get("face_present", True)
    face_count = data.get("face_count", 1)
    face_confidence = data.get("face_confidence", 0.9)
    head_pose = data.get("head_pose", {"yaw": 0, "pitch": 0, "roll": 0})
    landmarks = data.get("landmarks", [])
    result = {
        "face_present": face_present, "face_count": face_count,
        "face_confidence": face_confidence, "head_pose": head_pose,
        "movement": {}, "events": [], "risk_contribution": 0.0,
    }
    if landmarks and len(landmarks) >= 5:
        try:
            result["movement"] = analyzers["movement_detector"].process_landmarks(landmarks)
        except Exception:
            pass
    yaw = abs(head_pose.get("yaw", 0))
    if not face_present:
        result["events"].append({"type": "face_absent", "confidence": 1.0 - face_confidence, "source": "vision"})
        result["risk_contribution"] += 0.30
    if face_count > 1:
        result["events"].append({"type": "multiple_faces", "confidence": 0.8, "source": "vision"})
        result["risk_contribution"] += 0.50
    if yaw > 45:
        result["events"].append({"type": "head_turned", "confidence": min(yaw / 90, 1.0), "source": "vision"})
        result["risk_contribution"] += 0.20
    if yaw > 20:
        result["events"].append({"type": "gaze_away", "confidence": min(yaw / 60, 1.0), "source": "vision"})
        result["risk_contribution"] += 0.15
    result["risk_contribution"] = min(result["risk_contribution"], 1.0)
    return jsonify(result)


@app.route("/api/behavioral/analyze", methods=["POST"])
def behavioral_analyze():
    data = request.get_json(force=True)
    session_key = data.get("session_key", "default")
    from src.features.behavioral import BehavioralFeatureExtractor
    extractor = BehavioralFeatureExtractor()
    signals = {
        "face_present": data.get("face_present", True), "face_confidence": data.get("face_confidence", 0.9),
        "face_count": data.get("face_count", 1), "head_pose": data.get("head_pose", {"yaw": 0}),
        "audio_features": data.get("audio_features", {"rms_energy": 0.05, "volume_db": -60, "zero_crossing_rate": 0.1}),
        "keyboard_idle_seconds": data.get("keyboard_idle_seconds", 30),
        "keyboard_events_per_minute": data.get("keyboard_events_per_minute", 60),
        "mouse_avg_speed": data.get("mouse_avg_speed", 100),
        "mouse_total_distance": data.get("mouse_total_distance", 500),
        "mouse_click_count": data.get("mouse_click_count", 10),
        "mouse_idle_seconds": data.get("mouse_idle_seconds", 15),
        "multi_face_events": data.get("multi_face_events", 0),
        "session_elapsed": data.get("session_elapsed", 600),
        "total_events": data.get("total_events", 5),
    }
    try:
        features = extractor.extract(signals)
        feature_list = features.tolist() if hasattr(features, 'tolist') else list(features)
    except Exception:
        feature_list = [0.0] * 22
    kb = data.get("keyboard", {})
    keyboard_stats = {
        "keystrokes_per_min": kb.get("events_per_minute", 0), "mean_key_hold_ms": kb.get("mean_key_hold", 0),
        "mean_latency_ms": kb.get("mean_latency", 0),
        "idle_ratio": round(kb.get("idle_seconds", 0) / max(data.get("session_elapsed", 600), 1), 3),
        "deviation": round(kb.get("deviation", 0.0), 4),
    }
    mouse = data.get("mouse", {})
    mouse_stats = {
        "velocity_px_s": mouse.get("avg_speed", 0), "distance_px": mouse.get("total_distance", 0),
        "click_rate_s": mouse.get("click_rate", 0),
        "idle_ratio": round(mouse.get("idle_seconds", 0) / max(data.get("session_elapsed", 600), 1), 3),
        "deviation": round(mouse.get("deviation", 0.0), 4),
    }
    audio = data.get("audio_features", {})
    audio_stats = {
        "rms": round(audio.get("rms_energy", 0), 4), "db": round(audio.get("volume_db", -100), 1),
        "zcr": round(audio.get("zero_crossing_rate", 0), 4),
        "spectral_centroid": round(audio.get("spectral_centroid", 0), 1),
        "activity": "Active" if audio.get("volume_db", -100) > -40 else "Normal",
    }
    browser = data.get("browser", {})
    browser_stats = {
        "tab_switches": browser.get("tab_switches", 0), "window_blurs": browser.get("window_blurs", 0),
        "fullscreen_exits": browser.get("fullscreen_exits", 0), "visibility_changes": browser.get("visibility_changes", 0),
    }
    anomaly_score = round(min(
        keyboard_stats["deviation"] * 0.3 + mouse_stats["deviation"] * 0.3 +
        (1.0 if browser_stats["tab_switches"] > 3 else 0.0) * 0.4, 1.0), 4)
    result = {
        "session_key": session_key, "features": feature_list, "feature_count": len(feature_list),
        "keyboard": keyboard_stats, "mouse": mouse_stats, "audio": audio_stats, "browser": browser_stats,
        "anomaly_score": anomaly_score, "anomaly_detected": anomaly_score > 0.5,
    }
    return jsonify(result)


@app.route("/api/agent/status", methods=["GET"])
def agent_status_api():
    session_key = request.args.get("session_key", "")
    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({"state": "NORMAL", "running": False, "iteration": 0, "current_risk": 0.0, "evidence_events": 0, "history_length": 0})
    return jsonify(agent.get_status())


@app.route("/api/agent/history", methods=["GET"])
def agent_history_api():
    session_key = request.args.get("session_key", "")
    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({"history": []})
    return jsonify({"history": agent.get_history()})


@app.route("/api/agent/transitions", methods=["GET"])
def agent_transitions_api():
    session_key = request.args.get("session_key", "")
    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({"transitions": []})
    transitions = []
    for h in agent.get_history():
        transitions.append({
            "iteration": h.get("iteration"), "state": h.get("state", "NORMAL"),
            "risk_score": h.get("risk_score", 0.0),
            "events": [e.get("type") for e in h.get("events_detected", [])],
        })
    return jsonify({"transitions": transitions})


@app.route("/api/evidence/<session_id>", methods=["GET"])
def get_evidence_entries(session_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    evidence_log = _evidence_logs.get(session_id)
    if evidence_log:
        return jsonify({"entries": [e.to_dict() for e in evidence_log.get_entries()], "count": len(evidence_log.get_entries())})
    return jsonify({"entries": [], "count": 0})


@app.route("/api/evidence/<session_id>/<entry_id>", methods=["PUT"])
def update_evidence_entry(session_id, entry_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True)
    evidence_log = _evidence_logs.get(session_id)
    if evidence_log:
        for e in evidence_log.get_entries():
            if e.entry_id == entry_id:
                e.review_state = data.get("review_state", e.review_state)
                e.review_note = data.get("review_note", e.review_note)
                e.reviewed_by = user.get("full_name", user.get("username"))
                e.reviewed_at = datetime.utcnow()
                return jsonify({"entry": e.to_dict(), "message": "Updated"})
    return jsonify({"error": "Entry not found"}), 404


@app.route("/api/exams/<exam_id>/customize", methods=["GET"])
def get_exam_customize(exam_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    settings = _get_auth_db().get_exam_settings(exam_id)
    if not settings:
        return jsonify({"exam_id": exam_id, "exam_title": exam.title, "settings": {
            "randomize_questions": False, "randomize_options": False,
            "show_results_immediately": False, "allow_review_after_submit": True,
            "require_webcam": True, "require_microphone": False, "enable_proctoring": True,
            "auto_submit_on_timeout": True, "warn_before_submit": True, "max_tab_switches": 5,
            "lock_fullscreen": False, "prevent_copy_paste": True,
            "max_risk_score_before_flag": 0.7, "grading_release_delay_minutes": 0,
            "allow_student_notes": True, "calculator_allowed": False, "scratch_pad_allowed": True,
            "passing_score_percent": 40.0, "time_warning_minutes": 5,
        }, "instructions": "", "passing_score": 40.0})
    return jsonify(settings)


@app.route("/api/exams/<exam_id>/customize", methods=["PUT"])
def update_exam_customize(exam_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    me = _get_auth_db().get_session_user(token) if token else None
    if not me or me.get("role") not in ("super_admin", "admin", "examiner"):
        return jsonify({"error": "Unauthorized"}), 401
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    data = request.get_json(force=True)
    _get_auth_db().save_exam_settings(exam_id, data.get("settings", {}), data.get("instructions", ""), data.get("passing_score", 40.0))
    return jsonify({"exam_id": exam_id, "message": "Settings saved"})


@app.route("/api/preflight", methods=["POST"])
def preflight_check():
    data = request.get_json(force=True) or {}
    checks = {}
    checks["camera"] = {"required": data.get("require_webcam", True), "passed": bool(data.get("camera_available", False)), "message": "Camera detected" if data.get("camera_available") else "Camera not available"}
    checks["microphone"] = {"required": data.get("require_microphone", False), "passed": bool(data.get("microphone_available", False)), "message": "Microphone detected" if data.get("microphone_available") else "Microphone not available"}
    browser_caps = data.get("browser_capabilities", {})
    checks["browser"] = {"required": True, "passed": True, "message": "Browser compatible"}
    checks["fullscreen"] = {"required": data.get("lock_fullscreen", False), "passed": browser_caps.get("fullscreen", False), "message": "Fullscreen supported" if browser_caps.get("fullscreen") else "Not available"}
    checks["connection"] = {"required": True, "passed": data.get("online", True), "message": "Online" if data.get("online") else "Offline"}
    checks["models"] = {"required": True, "passed": True, "message": "Models loaded"}
    checks["session"] = {"required": True, "passed": bool(data.get("session_id") or data.get("exam_id")), "message": "Session configured"}
    required_passed = all(c["passed"] for c in checks.values() if c["required"])
    return jsonify({"checks": checks, "required_passed": required_passed, "can_proceed": required_passed})


@app.route("/api/grading/inspect/<student_id>/<exam_id>", methods=["GET"])
def grading_inspect(student_id, exam_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    session_key = student_id + ":" + exam_id
    session_info = _active_sessions.get(session_key)
    answers = {}
    if session_info and session_info.get("controller") and session_info["controller"].session:
        answers = session_info["controller"].session.answers
    else:
        try:
            db = _get_db()
            sid = None
            for s in db.list_sessions(limit=100):
                if s["student_id"] == student_id and s["exam_id"] == exam_id:
                    sid = s["session_id"]
                    break
            if sid:
                ans = db.get_answers(sid)
                answers = {a["question_id"]: a["student_answer"] for a in ans}
        except Exception:
            pass
    ref_answers = {}
    for q in exam.questions:
        if q.correct_answer and q.type in (QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER):
            ref_answers[q.id] = str(q.correct_answer)
    results = _grading_orchestrator.grade_all(exam.questions, answers, ref_answers)
    per_question = {}
    for q in exam.questions:
        qid = q.id
        if qid in results.get("per_question", {}):
            r = results["per_question"][qid]
            per_question[qid] = {
                "question_id": qid, "question_text": q.text, "question_type": q.type.value,
                "marks": q.marks, "student_answer": answers.get(qid, ""),
                "correct_answer": q.correct_answer, "keywords": q.keywords,
                "score": r.get("score", 0.0), "max_score": r.get("max_score", q.marks),
                "correct": r.get("correct", False),
                "grading_method": r.get("details", {}).get("method", "unknown") if isinstance(r.get("details"), dict) else r.get("method", "unknown"),
                "details": r.get("details", {}),
                "similarity": r.get("combined_similarity", r.get("semantic_similarity", 0)),
                "cosine_similarity": r.get("cosine_similarity", 0), "keyword_coverage": r.get("keyword_coverage", 0),
            }
    return jsonify({
        "student_id": student_id, "exam_id": exam_id, "per_question": per_question,
        "total_score": results.get("total_score", 0.0), "total_max": results.get("total_max", 0.0),
        "percentage": results.get("percentage", 0.0),
        "answers_submitted": len(answers), "total_questions": len(exam.questions),
    })


#  KNOWLEDGE TRACING
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/knowledge/<student_id>/<exam_id>", methods=["GET"])
def get_knowledge_state(student_id, exam_id):
    session_key = f"{student_id}:{exam_id}"
    kt = _knowledge_tracers.get(session_key)
    if not kt:
        return jsonify({"mastery": {}, "message": "No knowledge data yet"})
    return jsonify({"mastery": kt.get_all_mastery()})


@app.route("/api/knowledge/update", methods=["POST"])
def update_knowledge():
    data = request.get_json(force=True)
    session_key = data.get("session_key")
    topic = data.get("topic", "general")
    correct = data.get("correct", False)

    kt = _knowledge_tracers.get(session_key)
    if kt:
        kt.update(topic, 1.0 if correct else 0.0)
        return jsonify({"mastery": kt.get_all_mastery()})
    return jsonify({"error": "Session not found"}), 404


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _grade_session_internal(student_id, exam_id, exam, answers=None):
    """Grade a session's answers internally."""
    if answers is None:
        session_key = f"{student_id}:{exam_id}"
        session_info = _active_sessions.get(session_key)
        if session_info and session_info["controller"].session:
            answers = session_info["controller"].session.answers
        else:
            answers = {}

    if not answers:
        return {
            "per_question": {},
            "total_score": 0.0,
            "total_max": sum(q.marks for q in exam.questions),
            "percentage": 0.0,
        }

    # Build reference answers map
    ref_answers = {}
    for q in exam.questions:
        if q.correct_answer and q.type in (QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER):
            ref_answers[q.id] = str(q.correct_answer)

    results = _grading_orchestrator.grade_all(exam.questions, answers, ref_answers)

    per_question = {}
    for q in exam.questions:
        qid = q.id
        if qid in results.get("per_question", {}):
            per_question[qid] = {
                "question_id": qid,
                "question_text": q.text,
                "question_type": q.type.value,
                "marks": q.marks,
                **results["per_question"][qid],
            }

    return {
        "student_id": student_id,
        "exam_id": exam_id,
        "per_question": per_question,
        "total_score": results.get("total_score", 0.0),
        "total_max": results.get("total_max", 0.0),
        "percentage": results.get("percentage", 0.0),
    }


def _compute_contributing_signals(evidence_data):
    """Compute risk contribution breakdown from evidence."""
    from src.agent.risk import RiskEngine
    rengine = RiskEngine()

    event_counts = {}
    for entry in evidence_data.get("timeline", []):
        etype = entry.get("event_type", "unknown")
        event_counts[etype] = event_counts.get(etype, 0) + 1

    contributions = []
    for etype, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        weight = rengine.EVENT_WEIGHTS.get(etype, 0.05)
        contribution = round(weight * count * 100, 1)
        contributions.append({
            "signal": etype.replace("_", " ").title(),
            "contribution": contribution,
            "count": count,
            "weight": weight,
        })

    return contributions


def _compute_signal_activity(history):
    """Compute per-signal activity from proctoring history."""
    signal_data = {}
    for h in history:
        for evt in h.get("events_detected", []):
            source = evt.get("source", "unknown")
            if source not in signal_data:
                signal_data[source] = {"count": 0, "events": []}
            signal_data[source]["count"] += 1
            signal_data[source]["events"].append(evt["type"])

    result = []
    for source, data in signal_data.items():
        result.append({
            "signal": source.upper(),
            "count": data["count"],
            "events": list(set(data["events"])),
            "activity_level": min(data["count"] / 10.0, 1.0),
        })

    return sorted(result, key=lambda x: -x["count"])


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "message": str(e)}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Autonomous Exam Proctoring & Grading Agent")
    print("  API Server starting on http://localhost:5000")
    print("=" * 60)
    print("  Student Exam:  http://localhost:5000/student")
    print("  Examiner:      http://localhost:5000/examiner")
    print("  Analytics:     http://localhost:5000/analytics")
    print("  Training:      http://localhost:5000/training")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
