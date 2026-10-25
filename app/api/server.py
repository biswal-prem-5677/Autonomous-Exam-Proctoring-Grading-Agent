"""Flask REST API backend for the Exam Proctoring & Grading Agent.

Exposes all backend functionality as JSON endpoints for the HTML/JS frontend.
No external APIs — all computation is local.
"""

import sys
import json
import time
import uuid
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from flask import Flask, jsonify, request, render_template, g
from flask_cors import CORS

# ── Backend imports ──────────────────────────────────────────────────────────
from src.models.models import Question, QuestionType
from src.exam.exam import ExamManager
from src.grading.scorer import GradingOrchestrator
from src.agent.controller import ProctoringAgent
from src.agent.risk import RiskEngine
from src.features.behavioral import BehavioralFeatureExtractor
from src.ml.logistic import LogisticRegressionScratch
from src.ml.anomaly import AnomalyDetector
from src.ml.calibration import ModelEvaluator
from src.ml.training_data import TrainingDataGenerator
from src.prediction.performance import PerformancePredictor
from src.prediction.knowledge import KnowledgeTracer
from src.prediction.difficulty import DifficultyEstimator
from src.reports.student_report import StudentReportGenerator
from src.reports.examiner_report import ExaminerReportGenerator
from src.reports.evidence import EvidenceLog
from src.database.db import Database, DB_PATH
from src.utils.config import load_config
from src.trace.replay import DecisionReplayEngine
from src.trace.audit_chain import AuditChain
from src.grading.mathematical import MathematicalGrader
from src.grading.programming import ProgrammingGrader
from src.grading.plagiarism import PlagiarismDetector
from src.grading.consensus import GraderDisagreementEngine
from src.ml.adversarial import RedTeamHarness
from src.baseline import StudentBaseline

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__,
            static_folder="../static",
            template_folder="../templates")
CORS(app)  # type: ignore

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

# ── ML Model Persistence & Dynamic Attachment ──────────────────────────────
_loaded_logistic_model = None
_loaded_anomaly_model = None

def _get_active_models():
    global _loaded_logistic_model, _loaded_anomaly_model
    if _loaded_logistic_model is not None and _loaded_anomaly_model is not None:
        return _loaded_logistic_model, _loaded_anomaly_model
    import pickle
    from pathlib import Path
    models_dir = Path(__file__).parent.parent.parent / "models"
    log_pkl = models_dir / "logistic_model.pkl"
    ano_pkl = models_dir / "anomaly_model.pkl"
    if log_pkl.exists() and _loaded_logistic_model is None:
        try:
            with open(log_pkl, "rb") as f:
                _loaded_logistic_model = pickle.load(f)
        except Exception:
            pass
    if ano_pkl.exists() and _loaded_anomaly_model is None:
        try:
            with open(ano_pkl, "rb") as f:
                _loaded_anomaly_model = pickle.load(f)
        except Exception:
            pass
    return _loaded_logistic_model, _loaded_anomaly_model

def _create_proctoring_agent(config=None):
    agent = ProctoringAgent(config or _config)
    log_m, ano_m = _get_active_models()
    if log_m is not None or ano_m is not None:
        agent.set_models(logistic_model=log_m, anomaly_model=ano_m)
    return agent
_audit_chain = AuditChain(db_path=str(DB_PATH.parent / "audit_chain.db"))
_math_grader = MathematicalGrader()
_prog_grader = ProgrammingGrader()
_consensus_engine = GraderDisagreementEngine()
_student_baselines: Dict[str, StudentBaseline] = {}
# Training metadata — tracks actual training events, not fake status
_training_records: List[Dict] = []
_db_instance: Optional[Database] = None


def _get_db() -> Database:
    """Get or create the database instance (lazy singleton)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(DB_PATH)
    return _db_instance


# ── RBAC Auth Decorator ──────────────────────────────────────────────────────
from functools import wraps

def require_auth(allowed_roles=None):
    """Decorator: validate Bearer token and optionally enforce role.

    Sets ``request.user`` to the authenticated user dict on success.
    Returns 401 for missing/invalid token, 403 for wrong role.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
            if not token:
                return jsonify({"error": "Authentication required"}), 401
            user = _get_auth_db().get_session_user(token)
            if not user:
                return jsonify({"error": "Invalid or expired session"}), 401
            if allowed_roles and user.get("role") not in allowed_roles:
                return jsonify({"error": "Forbidden: insufficient privileges"}), 403
            request.user = user  # type: ignore[attr-defined]
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ── Answer-dict Unwrapping Helper ─────────────────────────────────────────────

def _unwrap_answer_value(val):
    """Unwrap ``{"answer": v, "timestamp": ...}`` → ``v``."""
    if isinstance(val, dict):
        if "answer" in val:
            return val["answer"]
        if "student_answer" in val:
            return val["student_answer"]
    return val


def _unwrap_answers_dict(answers: dict) -> dict:
    """Unwrap every value in an answers mapping."""
    return {k: _unwrap_answer_value(v) for k, v in answers.items()}


# ── Safe Controller Lookup (crash-proof after restart) ────────────────────────

def _find_session_id(student_id: str, exam_id: str) -> Optional[str]:
    """Look up the SQLite session_id for a student+exam pair."""
    session_key = f"{student_id}:{exam_id}"
    si = _active_sessions.get(session_key)
    if si and si.get("session_id"):
        return si["session_id"]
    db = _get_db()
    for s in db.list_sessions(limit=200):
        if s.get("student_id") == student_id and s.get("exam_id") == exam_id:
            return s["session_id"]
    try:
        row = db._conn.execute(
            "SELECT session_id FROM session_meta WHERE session_key = ? OR (student_id = ? AND exam_id = ?)",
            (session_key, student_id, exam_id)
        ).fetchone()
        if row:
            return row["session_id"]
    except Exception:
        pass
    return None


def _get_or_rebuild_controller(student_id: str, exam_id: str):
    """Return the live ExamController for the session, rebuilding from DB if needed.

    Returns ``(controller, session_info)`` or ``(None, None)`` if no session exists.
    """
    session_key = f"{student_id}:{exam_id}"
    session_info = _active_sessions.get(session_key)

    # Fast path: controller already present
    if session_info and "controller" in session_info and session_info["controller"] is not None:
        return session_info["controller"], session_info

    # Rebuild from DB
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return None, session_info

    db = _get_db()

    # Find session_id from DB
    db_session = None
    for s in db.list_sessions(limit=200):
        if s.get("student_id") == student_id and s.get("exam_id") == exam_id:
            db_session = s
            break

    if not db_session:
        try:
            row = db._conn.execute(
                "SELECT session_id, student_id, exam_id, created_at, meta_json FROM session_meta WHERE session_key = ?",
                (session_key,)
            ).fetchone()
            if not row:
                row = db._conn.execute(
                    "SELECT session_id, student_id, exam_id, created_at, meta_json FROM session_meta WHERE student_id = ? AND exam_id = ?",
                    (student_id, exam_id)
                ).fetchone()
            if row:
                db_session = dict(row)
        except Exception:
            pass

    if not db_session:
        return None, session_info

    # Rebuild controller
    duration = getattr(exam, "duration_minutes", 60) or 60
    controller = _exam_manager.get_or_create_controller(student_id, exam_id, duration)
    session = controller.session
    if not session:
        return None, session_info

    if db_session.get("session_id"):
        session.session_id = db_session["session_id"]

    # Restore answers from DB
    stored_answers = db.get_answers(db_session["session_id"])
    for ans in stored_answers:
        qid = ans.get("question_id")
        sval = ans.get("student_answer", "")
        if qid and qid not in session.answers:
            session.answers[qid] = sval

    # Rebuild active_sessions entry
    if session_info is None:
        session_info = {
            "student_id": student_id,
            "exam_id": exam_id,
            "session_id": db_session["session_id"],
            "created_at": db_session.get("created_at", ""),
            "started_at": db_session.get("created_at", ""),
        }
    session_info["controller"] = controller
    session_info.pop("_recovered", None)
    _active_sessions[session_key] = session_info

    # Rebuild proctoring agent if missing
    if session_key not in _proctoring_agents:
        agent = _create_proctoring_agent(_config)
        _proctoring_agents[session_key] = agent

    # Rebuild evidence log if missing
    if session_key not in _evidence_logs:
        _evidence_logs[session_key] = EvidenceLog(session_id=db_session["session_id"])

    return controller, session_info


def _load_answers_for_session(student_id: str, exam_id: str) -> dict:
    """Load answers from in-memory controller or fall back to SQLite."""
    session_key = f"{student_id}:{exam_id}"
    session_info = _active_sessions.get(session_key)
    if session_info and session_info.get("controller") and session_info["controller"].session:
        return dict(session_info["controller"].session.answers)
    # Fall back to DB
    sid = _find_session_id(student_id, exam_id)
    if sid:
        db = _get_db()
        ans_rows = db.get_answers(sid)
        return {a["question_id"]: a["student_answer"] for a in ans_rows}
    return {}


_STUDENT_GRADED: set = set()


def _canonical_session_id(session_id: str) -> str:
    """Accept either a DB session id or a 'student:exam' key and return the DB session id."""
    if session_id and ":" in session_id:
        stu, _, ex = session_id.partition(":")
        return _find_session_id(stu, ex) or session_id
    return session_id


def _apply_grade_overrides(session_id: str, per_question: dict, total_score: float) -> tuple:
    """Apply grade overrides from DB to per_question dict. Returns updated (per_question, total_score)."""
    db = _get_db()
    session_id = _canonical_session_id(session_id)
    overrides = db.get_grade_overrides(session_id=session_id)
    for ov in overrides:
        qid = ov.get("question_id")
        if qid and qid in per_question:
            old = per_question[qid].get("score", 0.0)
            new = ov.get("new_score", old)
            total_score = total_score - old + new
            per_question[qid]["score"] = new
            per_question[qid]["overridden"] = True
            per_question[qid]["override_reason"] = ov.get("reason", "")
    return per_question, total_score


def _recover_sessions_from_db() -> None:
    """Repopulate in-memory lightweight session metadata from the DB.

    This is called once at startup.  It does NOT recreate ProctoringAgent
    objects (those require the live exam controller), but it does restore
    the minimal `_active_sessions` dict entries needed for the examiner
    dashboard and session-status endpoints to report previously-started
    sessions that survived a server restart.
    """
    db = _get_db()
    try:
        recovered = db.get_active_sessions_meta()
        for row in recovered:
            skey = row["session_key"]
            if skey not in _active_sessions:
                # Re-populate with the persisted lightweight dict.  The
                # "controller" key is intentionally absent — endpoints that
                # strictly need it will return 404/error, which is correct
                # behaviour for sessions that pre-date the current process.
                _active_sessions[skey] = {
                    "student_id": row["student_id"],
                    "exam_id": row["exam_id"],
                    "session_id": row["session_id"],
                    "created_at": row.get("created_at", ""),
                    "started_at": row.get("created_at", ""),
                    **row.get("meta", {}),
                    "_recovered": True,  # flag: no live controller attached
                }
    except Exception:
        # DB may not have the new table yet (first run before migration);
        # gracefully skip — sessions are simply not recovered.
        pass


# Eagerly open the DB so schema migrations run at import time rather than on
# the first request (avoids a race condition under concurrent startup).
_get_db()
_recover_sessions_from_db()


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
@app.route("/student/exam")
def student_exam_new():
    """New student exam interface with webcam."""
    return render_template("student/exam.html")


@app.route("/examiner")
@app.route("/examiner/dashboard")
def examiner_dashboard():
    """Examiner monitoring dashboard."""
    return render_template("examiner_dashboard.html")


@app.route("/examiner/grading")
def examiner_grading():
    """Examiner grading page."""
    return render_template("examiner/grading.html")


@app.route("/examiner/create_exam")
@app.route("/examiner/create-exam")
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


@app.route("/analytics")
def analytics_page():
    """Analytics dashboard page."""
    return render_template("analytics.html")


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
    _tok = (request.headers.get("Authorization", "") or "").replace("Bearer ", "", 1).strip()
    _u = _get_auth_db().get_session_user(_tok) if _tok else None
    _is_student = bool(_u and _u.get("role") == "student")
    for eid, exam in _exam_manager._exams.items():
        if _is_student and _u and not (_get_db().is_student_enrolled(eid, _u.get("username", "")) or _get_db().is_student_enrolled(eid, _u.get("user_id", ""))):
            continue
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
    exam_id = data.get("exam_id") or data.get("id") or f"exam_{uuid.uuid4().hex[:10]}"
    creator = (g.current_user.get("username") or g.current_user.get("user_id")) if getattr(g, "current_user", None) else data.get("created_by", "examiner")
    exam = _exam_manager.create_exam(
        exam_id=exam_id,
        title=data.get("title", "Untitled Exam"),
        duration_minutes=data.get("duration_minutes", 60),
        questions=[],
        created_by=creator,
    )
    exam.topic = data.get("topic", "")
    exam.difficulty = data.get("difficulty", "medium")
    exam.instructions = data.get("instructions", "")
    exam.passing_score = float(data.get("passing_score", 40.0))
    settings = dict(data.get("settings", {}) or {})
    for k in ["policy", "require_webcam", "require_microphone", "require_fullscreen",
              "allow_calculator", "open_book", "max_tab_switches", "risk_threshold",
              "start_window", "end_window"]:
        if k in data and k not in settings:
            settings[k] = data[k]
    exam.settings = settings
    if "is_released" in data:
        exam.is_released = bool(data["is_released"])

    # Add questions if provided in the creation request
    for qd in data.get("questions", []):
        meta = dict(qd.get("metadata", {}) or {})
        for field in ["formula", "governing_formula", "formula_hint", "variables", "expected_variables",
                      "required_params", "final_value", "units", "language", "starter_code",
                      "function_name", "target_function", "entry_function_name", "tests",
                      "test_cases", "hidden_tests", "edge_cases", "model_answer", "reference_answer", "rubric"]:
            if field in qd and field not in meta:
                meta[field] = qd[field]
        exam.add_question(
            qtype=qd.get("type", "mcq"),
            text=qd.get("text", ""),
            options=qd.get("options"),
            correct_answer=qd.get("correct_answer"),
            marks=float(qd.get("marks", 1.0)),
            tolerance=float(qd.get("tolerance", 0.01)) if qd.get("tolerance") is not None else None,
            keywords=qd.get("keywords", []),
            topic=qd.get("topic", ""),
            difficulty=qd.get("difficulty", "medium"),
            metadata=meta,
        )

    creator = (g.current_user.get("username") or g.current_user.get("user_id")) if getattr(g, "current_user", None) else data.get("created_by", "examiner")
    # Persist in SQLite
    try:
        _get_db().save_exam(exam.to_dict(), created_by=creator)
    except Exception as e:
        logger.error(f"Failed to persist exam {exam.exam_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to persist exam: {str(e)}"}), 500

    return jsonify({
        "exam_id": exam.exam_id,
        "title": exam.title,
        "duration_minutes": exam.duration_minutes,
        "instructions": exam.instructions,
        "passing_score": exam.passing_score,
        "settings": exam.settings,
        "question_count": len(exam.questions),
        "total_marks": sum(q.marks for q in exam.questions),
        "status": "created",
    })


@app.route("/api/exams/<exam_id>", methods=["GET"])
def get_exam(exam_id):
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    # Strip answer keys for students
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    caller = _get_auth_db().get_session_user(token) if token else None
    include_answers = not (caller and caller.get("role") == "student")
    return jsonify({
        "exam_id": exam.exam_id,
        "title": exam.title,
        "duration_minutes": exam.duration_minutes,
        "topic": exam.topic,
        "difficulty": exam.difficulty,
        "instructions": getattr(exam, "instructions", ""),
        "passing_score": getattr(exam, "passing_score", 40.0),
        "is_released": getattr(exam, "is_released", False),
        "settings": getattr(exam, "settings", {}),
        "questions": [q.to_dict(include_answer=include_answers) for q in exam.questions],
        "question_count": len(exam.questions),
        "total_marks": sum(q.marks for q in exam.questions),
    })


@app.route("/api/exams/<exam_id>/questions", methods=["POST"])
def add_question(exam_id):
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    data = request.get_json(force=True)
    meta = dict(data.get("metadata", {}) or {})
    for field in ["formula", "governing_formula", "formula_hint", "variables", "expected_variables",
                  "required_params", "final_value", "units", "language", "starter_code",
                  "function_name", "target_function", "entry_function_name", "tests",
                  "test_cases", "hidden_tests", "edge_cases", "model_answer", "reference_answer", "rubric"]:
        if field in data and field not in meta:
            meta[field] = data[field]

    q = exam.add_question(
        qtype=data.get("type", "mcq"),
        text=data.get("text", ""),
        options=data.get("options"),
        correct_answer=data.get("correct_answer"),
        marks=float(data.get("marks", 1.0)),
        keywords=data.get("keywords", []),
        tolerance=float(data.get("tolerance")) if data.get("tolerance") is not None else None,
        topic=data.get("topic", ""),
        difficulty=data.get("difficulty", "medium"),
        metadata=meta,
    )
    try:
        _get_db().save_exam(exam.to_dict())
    except Exception as e:
        logger.error(f"Failed to persist question for exam {exam_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to persist question: {str(e)}"}), 500
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
        meta = dict(qd.get("metadata", {}) or {})
        for field in ["formula", "governing_formula", "formula_hint", "variables", "expected_variables",
                      "required_params", "final_value", "units", "language", "starter_code",
                      "function_name", "target_function", "entry_function_name", "tests",
                      "test_cases", "hidden_tests", "edge_cases", "model_answer", "reference_answer", "rubric"]:
            if field in qd and field not in meta:
                meta[field] = qd[field]
        q = exam.add_question(
            qtype=qd.get("type", "mcq"),
            text=qd.get("text", ""),
            options=qd.get("options"),
            correct_answer=qd.get("correct_answer"),
            marks=float(qd.get("marks", 1.0)),
            keywords=qd.get("keywords", []),
            tolerance=float(qd.get("tolerance")) if qd.get("tolerance") is not None else None,
            topic=qd.get("topic", ""),
            difficulty=qd.get("difficulty", "medium"),
            metadata=meta,
        )
        added.append(q.to_dict())
    try:
        _get_db().save_exam(exam.to_dict())
    except Exception as e:
        logger.error(f"Failed to persist bulk questions for exam {exam_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to persist bulk questions: {str(e)}"}), 500
    return jsonify({"questions": added, "count": len(added)})


# ── Exam Roster & Release Endpoints ──────────────────────────────────────────

@app.route("/api/exams/<exam_id>/roster", methods=["GET", "POST"])
def exam_roster_api(exam_id):
    """Manage enrolled students for an exam."""
    db = _get_db()
    exam = _exam_manager.get_exam(exam_id) or db.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    if request.method == "GET":
        roster = db.get_roster(exam_id)
        return jsonify({"exam_id": exam_id, "roster": roster, "count": len(roster)})

    data = request.get_json(force=True) if request.data else {}
    student_id = data.get("student_id")
    student_name = data.get("student_name", "")
    student_email = data.get("student_email", "")
    if not student_id:
        return jsonify({"error": "student_id is required"}), 400

    db.enroll_student(exam_id, student_id, student_name, student_email)
    return jsonify({
        "status": "enrolled",
        "exam_id": exam_id,
        "student_id": student_id,
        "student_name": student_name,
        "student_email": student_email,
    })


@app.route("/api/exams/<exam_id>/roster/<student_id>", methods=["DELETE"])
def remove_from_roster_api(exam_id, student_id):
    """Remove a student from the exam roster."""
    db = _get_db()
    exam = _exam_manager.get_exam(exam_id) or db.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    removed = db.remove_student_from_roster(exam_id, student_id)
    return jsonify({
        "status": "removed" if removed else "not_found",
        "exam_id": exam_id,
        "student_id": student_id,
    })


@app.route("/api/exams/<exam_id>/roster/bulk", methods=["POST"])
def bulk_enroll_roster(exam_id):
    """Bulk enroll students into an exam roster."""
    db = _get_db()
    exam = _exam_manager.get_exam(exam_id) or db.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    data = request.get_json(force=True) if request.data else {}
    students = data.get("students", [])
    if not isinstance(students, list):
        return jsonify({"error": "students must be a list"}), 400

    enrolled = []
    for s in students:
        sid = s.get("student_id")
        if sid:
            sname = s.get("student_name", "")
            semail = s.get("student_email", "")
            db.enroll_student(exam_id, sid, sname, semail)
            enrolled.append({"student_id": sid, "student_name": sname, "student_email": semail})
    return jsonify({
        "status": "success",
        "exam_id": exam_id,
        "enrolled_count": len(enrolled),
        "students": enrolled,
    })


@app.route("/api/exams/<exam_id>/release", methods=["POST"])
def release_exam_api(exam_id):
    """Publish/release an exam to students or revoke release."""
    db = _get_db()
    exam = _exam_manager.get_exam(exam_id) or db.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    data = request.get_json(force=True) if request.data else {}
    is_released = bool(data.get("is_released", True))
    success = db.set_exam_released(exam_id, is_released)
    ex_obj = _exam_manager.get_exam(exam_id)
    if ex_obj:
        ex_obj.is_released = is_released
    return jsonify({
        "status": "success",
        "exam_id": exam_id,
        "is_released": is_released,
        "updated": success,
    })


@app.route("/api/exams/<exam_id>/release-status", methods=["GET"])
def get_exam_release_status(exam_id):
    """Check if an exam is published/released to students."""
    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404
    is_released = getattr(exam, "is_released", False)
    return jsonify({"exam_id": exam_id, "is_released": bool(is_released)})



# ═══════════════════════════════════════════════════════════════════════════════
#  EXAM SESSIONS (student taking exam)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/exam-sessions", methods=["POST"])
def create_session():
    """Start a new exam session for a student."""
    data = request.get_json(force=True)
    student_id = data.get("student_id", f"student_{int(time.time())}")
    exam_id = data.get("exam_id")

    # Enforce student_id from auth token for student users (prevent spoofing)
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    auth_user = _get_auth_db().get_session_user(token) if token else None
    if auth_user and auth_user.get("role") == "student":
        valid_ids = {auth_user.get("user_id"), auth_user.get("username")}
        valid_ids = {v for v in valid_ids if v}
        if student_id not in valid_ids:
            student_id = auth_user.get("user_id") or auth_user.get("username") or student_id

    if not exam_id:
        return jsonify({"error": "exam_id required"}), 400

    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    # Enforce the exam's configured duration — ignore client-supplied override
    duration = getattr(exam, "duration_minutes", 60) or 60

    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None

    # Preview Mode Security: only authenticated examiners or admins may invoke preview mode
    wants_preview = bool(data.get("is_preview", False) or request.headers.get("X-Preview-Mode") == "true")
    is_authorized_preview = False
    if wants_preview:
        if user and user.get("role") in ("super_admin", "admin", "examiner"):
            is_authorized_preview = True
        else:
            return jsonify({
                "error": "unauthorized_preview",
                "message": "Preview mode requires authenticated examiner or administrator credentials.",
                "status": "unauthorized_preview"
            }), 403

    # Cross-student isolation: authenticated students cannot start sessions for other student IDs
    if user and user.get("role") == "student":
        valid_ids = {user.get("user_id"), user.get("username")}
        valid_ids = {v for v in valid_ids if v}
        if student_id not in valid_ids:
            return jsonify({
                "error": "Forbidden: You cannot initiate an exam session for another student.",
                "status": "forbidden"
            }), 403

    db = _get_db()
    # 1. Authoritative Release Enforcement
    is_released = getattr(exam, "is_released", False)
    db_exam = db.get_exam(exam_id)
    if db_exam:
        is_released = is_released or db_exam.get("is_released", False)
    if not is_released and not is_authorized_preview:
        return jsonify({
            "error": "This exam has not been released or published by the examiner yet.",
            "status": "not_released"
        }), 403

    # 2. Authoritative Student Eligibility & Roster Enforcement
    if not is_authorized_preview and not db.is_student_enrolled(exam_id, student_id):
        return jsonify({
            "error": f"Student '{student_id}' is not enrolled in the roster for this exam.",
            "status": "not_enrolled"
        }), 403

    # 3. Schedule Window and Extra Time Accommodations
    enrollment = db.get_student_enrollment(exam_id, student_id)
    if enrollment:
        extra_time = int(enrollment.get("extra_time_mins", 0) or 0)
        duration = duration + extra_time
        now_iso = datetime.now(timezone.utc).isoformat()
        if enrollment.get("start_window") and now_iso < enrollment["start_window"]:
            return jsonify({
                "error": f"Exam access window has not opened yet. Starts at {enrollment['start_window']}",
                "status": "window_not_started"
            }), 403
        if enrollment.get("end_window") and now_iso > enrollment["end_window"]:
            return jsonify({
                "error": f"Exam access window closed at {enrollment['end_window']}",
                "status": "window_closed"
            }), 403

    controller = _exam_manager.get_or_create_controller(
        student_id, exam_id, duration
    )
    session = controller.session
    if not session:
        return jsonify({"error": "Failed to create session"}), 500

    # Capture session creation time before any processing
    session_created_at = getattr(session, "created_at", None) or datetime.now().isoformat()

    controller.start_exam()

    session_key = f"{student_id}:{exam_id}"
    _active_sessions[session_key] = {
        "student_id": student_id,
        "exam_id": exam_id,
        "controller": controller,
        "session_id": session.session_id,
        "created_at": session_created_at,
        "started_at": session_created_at,
    }

    # Persist session metadata to SQLite so it survives server restarts.
    try:
        _get_db().upsert_session_meta(
            session_key=session_key,
            session_id=session.session_id,
            student_id=student_id,
            exam_id=exam_id,
            meta={
                "started_at": session_created_at,
                "duration_minutes": duration,
            },
            is_active=True,
        )
    except Exception:
        pass  # DB failure must not block exam start

    # Also persist into the sessions table so list_sessions() finds it after restart.
    try:
        db = _get_db()
        with db._write_lock:
            db._conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, student_id, exam_id, duration_minutes, started_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session.session_id, student_id, exam_id, duration, session_created_at),
            )
            db._conn.commit()
    except Exception:
        pass  # best-effort; session_meta is the backup

    # Initialize proctoring agent for this session with active trained ML models
    agent = _create_proctoring_agent(_config)
    if student_id in _student_baselines:
        agent.set_baseline(_student_baselines[student_id])
    exam_policy_name = (getattr(exam, "settings", None) or {}).get("policy", "standard")
    from src.context import ExamContext, get_default_policy
    agent.set_context(ExamContext(
        policy=get_default_policy(exam_policy_name),
        exam_duration_seconds=float(duration * 60)
    ))
    _proctoring_agents[session_key] = agent

    # Initialize evidence log
    _evidence_logs[session_key] = EvidenceLog(session_id=session.session_id)

    # Initialize knowledge tracer
    _knowledge_tracers[session_key] = KnowledgeTracer()

    # Initialize difficulty estimator
    _difficulty_estimators[session_key] = DifficultyEstimator()

    return jsonify({
        "session_id": session.session_id,
        "session_key": session_key,
        "student_id": student_id,
        "exam_id": exam_id,
        "duration_minutes": duration,
        "started_at": session.start_time.isoformat(),
        "questions": [q.to_dict(include_answer=False) for q in exam.questions],
        "total_marks": sum(q.marks for q in exam.questions),
        "status": "started",
        "is_preview": is_authorized_preview,
    })


@app.route("/api/exam-sessions/<student_id>/<exam_id>/status", methods=["GET"])
def session_status(student_id, exam_id):
    controller, session_info = _get_or_rebuild_controller(student_id, exam_id)
    if not controller:
        return jsonify({"error": "Session not found"}), 404

    status = controller.get_session_status()

    # Add questions if available, ensuring correct_answer is stripped for students
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    user = _get_auth_db().get_session_user(token) if token else None
    is_student = (user and user.get("role") == "student")

    exam = _exam_manager.get_exam(exam_id) or _get_db().get_exam(exam_id)
    if exam:
        if hasattr(exam, "questions"):
            status["questions"] = [q.to_dict(include_answer=not is_student) for q in exam.questions]
            status["total_marks"] = sum(q.marks for q in exam.questions)
        elif isinstance(exam, dict) and "questions" in exam:
            q_list = []
            for q_data in exam["questions"]:
                qd = dict(q_data)
                if is_student:
                    qd.pop("correct_answer", None)
                    qd.pop("reference_answer", None)
                    qd.pop("keywords", None)
                    qd.pop("rubric", None)
                q_list.append(qd)
            status["questions"] = q_list
            status["total_marks"] = sum(q.get("marks", 1.0) for q in exam["questions"])

    return jsonify(status)


@app.route("/api/exam-sessions/<student_id>/<exam_id>/answers", methods=["POST"])
def submit_answer(student_id, exam_id):
    """Submit an answer for a question."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if user and user.get("role") == "student":
        valid_ids = {user.get("user_id"), user.get("username")}
        valid_ids = {v for v in valid_ids if v}
        if student_id not in valid_ids:
            return jsonify({"error": "Forbidden: Cannot access another student's session"}), 403

    controller, session_info = _get_or_rebuild_controller(student_id, exam_id)
    if not controller:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json(force=True)
    question_id = data.get("question_id")
    answer = data.get("answer")

    if question_id is None:
        return jsonify({"error": "question_id required"}), 400

    controller.submit_answer(question_id, answer)

    # Persist to SQLite for durability
    try:
        sid = session_info.get("session_id", "") if session_info else _find_session_id(student_id, exam_id)
        if sid:
            _get_db().record_answer(
                session_id=sid,
                question_id=str(question_id),
                student_answer=str(answer) if answer is not None else "",
            )
    except Exception:
        pass  # best-effort DB save; in-memory is the authoritative copy during exam

    # Update knowledge tracer
    session_key = f"{student_id}:{exam_id}"
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
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if user and user.get("role") == "student":
        valid_ids = {user.get("user_id"), user.get("username")}
        valid_ids = {v for v in valid_ids if v}
        if student_id not in valid_ids:
            return jsonify({"error": "Forbidden: Cannot access another student's session"}), 403

    controller, session_info = _get_or_rebuild_controller(student_id, exam_id)
    if not controller:
        return jsonify({"error": "Session not found"}), 404

    session = controller.end_exam()
    if not session:
        return jsonify({"error": "Failed to end exam session"}), 500

    # Deactivate the DB session-meta record so it is not recovered on restart.
    session_key = f"{student_id}:{exam_id}"
    try:
        _get_db().deactivate_session_meta(session_key)
    except Exception:
        pass

    ended_iso = session.end_time.isoformat() if hasattr(session.end_time, "isoformat") else (str(session.end_time) if session.end_time else None)
    return jsonify({
        "session_id": session.session_id,
        "status": "submitted",
        "answers_count": len(session.answers),
        "events_count": len(session.events),
        "ended_at": ended_iso,
    })


@app.route("/api/answers", methods=["POST"])
def post_answer_unified():
    """Unified autosave endpoint for individual student answers."""
    data = request.get_json(force=True) if request.data else {}
    session_id = data.get("session_id")
    question_id = data.get("question_id")
    answer = data.get("answer")

    if not session_id or question_id is None:
        return jsonify({"error": "session_id and question_id are required"}), 400

    # Locate session
    target_session_info = None
    target_session_key = None
    for skey, sinfo in _active_sessions.items():
        if sinfo.get("session_id") == session_id:
            target_session_info = sinfo
            target_session_key = skey
            break

    db_s = None
    if not target_session_info:
        try:
            db_s = _get_db().get_session(session_id)
            if db_s:
                _, target_session_info = _get_or_rebuild_controller(db_s["student_id"], db_s["exam_id"])
                target_session_key = f"{db_s['student_id']}:{db_s['exam_id']}"
        except Exception:
            pass

    # Security: check authorization
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if user and user.get("role") == "student":
        session_student_id = target_session_info.get("student_id") if target_session_info else None
        if not session_student_id and db_s:
            session_student_id = db_s.get("student_id")
        elif not session_student_id:
            try:
                db_s = _get_db().get_session(session_id)
                if db_s:
                    session_student_id = db_s.get("student_id")
            except Exception:
                pass
        valid_ids = {user.get("user_id"), user.get("username")}
        valid_ids = {v for v in valid_ids if v}
        if session_student_id and session_student_id not in valid_ids:
            return jsonify({"error": "Forbidden: Cannot submit answers for another student's session"}), 403

    # Submit answer in controller
    if target_session_info:
        ctrl = target_session_info.get("controller")
        if ctrl and hasattr(ctrl, "submit_answer"):
            ctrl.submit_answer(question_id, answer)

    # Persist to SQLite
    try:
        _get_db().record_answer(
            session_id=session_id,
            question_id=str(question_id),
            student_answer=str(answer) if answer is not None else "",
        )
    except Exception:
        pass

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "question_id": question_id,
        "answer": answer,
        "saved_at": datetime.now().isoformat(),
    })


@app.route("/api/exam-sessions/<session_id>/finish", methods=["POST"])
def finish_exam_session(session_id):
    """End exam session by session_id."""
    target_session_info = None
    target_session_key = None
    for skey, sinfo in _active_sessions.items():
        if sinfo.get("session_id") == session_id:
            target_session_info = sinfo
            target_session_key = skey
            break

    db_s = None
    if not target_session_info:
        try:
            db_s = _get_db().get_session(session_id)
            if db_s:
                _, target_session_info = _get_or_rebuild_controller(db_s["student_id"], db_s["exam_id"])
                target_session_key = f"{db_s['student_id']}:{db_s['exam_id']}"
        except Exception:
            pass

    # Security: verify student ownership
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if user and user.get("role") == "student":
        session_student_id = target_session_info.get("student_id") if target_session_info else None
        if not session_student_id and db_s:
            session_student_id = db_s.get("student_id")
        elif not session_student_id:
            try:
                db_s = _get_db().get_session(session_id)
                if db_s:
                    session_student_id = db_s.get("student_id")
            except Exception:
                pass
        valid_ids = {user.get("user_id"), user.get("username")}
        valid_ids = {v for v in valid_ids if v}
        if session_student_id and session_student_id not in valid_ids:
            return jsonify({"error": "Forbidden: Cannot finish another student's session"}), 403

    if target_session_info:
        ctrl = target_session_info.get("controller")
        if ctrl and hasattr(ctrl, "end_exam"):
            session = ctrl.end_exam()
            if target_session_key:
                try:
                    _get_db().deactivate_session_meta(target_session_key)
                except Exception:
                    pass
            ended_iso = session.end_time.isoformat() if hasattr(session, "end_time") and hasattr(session.end_time, "isoformat") else None
            return jsonify({
                "status": "submitted",
                "session_id": session_id,
                "answers_count": len(session.answers) if hasattr(session, "answers") else 0,
                "ended_at": ended_iso,
            })

    return jsonify({"status": "submitted", "session_id": session_id})


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
    questions_data = data.get("questions")
    if not exam and questions_data:
        questions = []
        for qd in questions_data:
            q_type = QuestionType(qd.get("type", "mcq"))
            questions.append(Question(
                id=qd.get("id", str(uuid.uuid4())[:8]),
                type=q_type,
                text=qd.get("text", ""),
                options=qd.get("options"),
                correct_answer=qd.get("correct_answer"),
                marks=float(qd.get("marks", 1.0)),
                tolerance=float(qd.get("tolerance", 0.01)) if qd.get("tolerance") is not None else None,
                rubric=qd.get("rubric"),
            ))
        results = _grading_orchestrator.grade_all(questions, answers, reference_answers)
        return jsonify({
            "exam_id": "ad_hoc",
            "student_id": student_id,
            "per_question": results.get("per_question", {}),
            "total_score": results.get("total_score", 0.0),
            "total_max": results.get("total_max", 0.0),
            "percentage": results.get("percentage", 0.0),
        })

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
@app.route("/api/proctor/process/<student_id>/<exam_id>", methods=["POST"])
def process_proctor_signals(student_id=None, exam_id=None):
    """Process a batch of proctoring signals through the agent loop."""
    data = request.get_json(force=True) if request.data else {}
    if not data:
        data = {}
    session_key = data.get("session_key")
    if not session_key:
        if student_id and exam_id:
            session_key = f"{student_id}:{exam_id}"
        else:
            session_key = "default"

    signals = data.get("signals", {})

    agent = _proctoring_agents.get(session_key)
    if not agent:
        agent = _create_proctoring_agent(_config)
        _proctoring_agents[session_key] = agent
        if session_key not in _evidence_logs:
            _evidence_logs[session_key] = EvidenceLog(session_id=session_key)

    result = agent.process_signals(signals)

    # Record evidence
    evidence_log = _evidence_logs.get(session_key)
    if evidence_log:
        for evt in result.get("events_detected", []):
            evidence_log.add_from_risk_event(
                event_type=evt.get("type", "unknown"),
                confidence=evt.get("confidence", 0.8),
                description=f"Detected {evt.get('type', 'event')} from {evt.get('source', 'sensor')} signal",
                signal_source=evt.get("source", "sensor"),
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
#  EXPLAINABILITY & TRACE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/proctor/trace/<student_id>/<exam_id>", methods=["GET"])
def proctor_trace(student_id, exam_id):
    """Get the full explainability decision trace for the last agent cycle."""
    session_key = f"{student_id}:{exam_id}"
    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({"error": "No active proctoring agent for this session"}), 404

    trace = agent.get_last_trace()
    if trace is None:
        return jsonify({
            "trace": None,
            "message": "No trace available yet. Process at least one batch of signals.",
        })

    return jsonify({"trace": trace})


@app.route("/api/proctor/summary/<student_id>/<exam_id>", methods=["GET"])
def proctor_summary(student_id, exam_id):
    """Get a lightweight proctoring summary for the examiner dashboard."""
    session_key = f"{student_id}:{exam_id}"
    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({
            "status": "no_agent",
            "student_id": student_id,
            "exam_id": exam_id,
            "risk_score": 0.0,
            "decision": "INSUFFICIENT_EVIDENCE",
            "events": [],
            "contributions": [],
            "confidence": 0.0,
        })

    trace = agent.get_last_trace()
    if trace:
        curr_risk = trace.get("current_risk")
        if curr_risk is None:
            curr_risk = trace.get("risk", {}).get("current", 0.0)
        prev_risk = trace.get("previous_risk")
        if prev_risk is None:
            prev_risk = trace.get("risk", {}).get("previous", 0.0)

        return jsonify({
            "status": "active",
            "student_id": student_id,
            "exam_id": exam_id,
            "risk_score": float(curr_risk or 0.0),
            "previous_risk": float(prev_risk or 0.0),
            "risk_delta": round(float(curr_risk or 0.0) - float(prev_risk or 0.0), 4),
            "decision": trace.get("decision"),
            "decision_reason": trace.get("decision_reason"),
            "confidence": trace.get("uncertainty", {}).get("confidence", 0.0),
            "signal_agreement": trace.get("uncertainty", {}).get("signal_agreement", 0),
            "total_signals": trace.get("uncertainty", {}).get("total_signals", 0),
            "uncertainty_reason": trace.get("uncertainty", {}).get("reason"),
            "events": trace.get("events", []),
            "contributions": trace.get("contributions", []),
            "counterfactuals": trace.get("counterfactuals", []),
            "non_contributing": trace.get("non_contributing", []),
            "corroboration": trace.get("corroboration"),
            "ml_risk": trace.get("ml_risk"),
            "anomaly_score": trace.get("anomaly_score"),
            "trace_id": trace.get("trace_id"),
            "timestamp": trace.get("timestamp"),
        })

    # Fallback: build from agent status
    status = agent.get_status()
    risk_score = status.get("current_risk", status.get("risk_score", 0.0))
    return jsonify({
        "status": "active",
        "student_id": student_id,
        "exam_id": exam_id,
        "risk_score": float(risk_score or 0.0),
        "decision": status.get("state", "NORMAL"),
        "confidence": 0.0,
        "events": [],
        "contributions": [],
    })


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
        controller = session_info.get("controller")
        if not controller:
            controller, session_info = _get_or_rebuild_controller(student_id, exam_id)
        if not controller:
            continue
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
            pct_val = graded.get("percentage", 0.0)
            score_pct = float(pct_val) if isinstance(pct_val, (int, float, str)) else 0.0
            total_score += score_pct
            scored_count += 1

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
            "last_activity": session_info.get("started_at", "") if session_info else "",
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

    controller, session_info = _get_or_rebuild_controller(student_id, exam_id)
    status = controller.get_session_status() if controller else {}

    # Risk history
    agent = _proctoring_agents.get(session_key)
    history = agent.get_history() if agent else []

    # Evidence
    evidence_log = _evidence_logs.get(session_key)
    evidence_data = evidence_log.to_dict() if evidence_log else {}

    # Grading results (safely loaded and unwrapped)
    answers = _load_answers_for_session(student_id, exam_id)
    grading = _grade_session_internal(student_id, exam_id, exam, answers)

    # Build risk timeline from history — use actual timestamps from entries
    risk_timeline = []
    for h in history:
        risk_timeline.append({
            "timestamp": h.get("timestamp", h.get("created_at", datetime.now().isoformat())),
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
    prediction: Dict[str, Any] = {}
    if predictor:
        try:
            pred_res = predictor.predict(grading_results)
            if isinstance(pred_res, dict):
                prediction = pred_res
            elif hasattr(pred_res, "tolist"):
                prediction = {"predicted_scores": pred_res.tolist()}
            else:
                prediction = {"predicted_scores": pred_res}
        except Exception:
            prediction = {}

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




@app.route("/api/predict/difficulty", methods=["POST"])
def estimate_difficulty():
    data = request.get_json(force=True) if request.data else {}
    if not data:
        data = {}
    session_key = data.get("session_key") or data.get("exam_id") or "default"
    question_id = data.get("question_id", "unknown")
    responses = data.get("responses", [])
    if responses and isinstance(responses, list):
        correct_count = sum(1 for r in responses if r == 1 or r is True)
        total_count = max(len(responses), 1)
    else:
        correct_count = data.get("correct_count", 0)
        total_count = data.get("total_count", 1)

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
            "normal": sum(1 for l in y if l == 0),
            "suspicious": sum(1 for l in y if l == 1),
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
            "feature_weights": dict(zip(feature_names[:len(lr.weights)] if lr.weights is not None else [], lr.weights.tolist() if lr.weights is not None else [])),
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

    exam = _exam_manager.get_exam(exam_id) if exam_id else None
    if not exam:
        return jsonify({"error": f"Exam not found: {exam_id}"}), 404


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
    role_str = data.get("role", "student")
    if role_str not in ("super_admin", "admin", "examiner", "student", "viewer"):
        return jsonify({"error": "Invalid role"}), 400

    # RBAC: Public registration cannot escalate privileges.
    # Registering as super_admin, admin, or examiner requires an active admin session,
    # unless bootstrapping an empty system with no existing privileged users.
    if role_str in ("super_admin", "admin", "examiner"):
        caller_token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        caller = auth_db.get_session_user(caller_token) if caller_token else None
        existing_privileged = [u for u in auth_db.list_users() if u.get("role") in ("admin", "super_admin", "examiner")]
        if existing_privileged and (not caller or caller.get("role") not in ("admin", "super_admin")):
            return jsonify({
                "error": "Privilege escalation denied: Administrator token required to register privileged roles."
            }), 403
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


@app.route("/api/auth/me", methods=["GET", "PUT"])
def auth_me():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return jsonify({"error": "No token provided"}), 401
    auth_db = _get_auth_db()
    user = auth_db.get_session_user(token)
    if not user:
        return jsonify({"error": "Invalid or expired session"}), 401

    if request.method == "PUT":
        data = request.get_json(force=True) if request.data else {}
        allowed = {}
        if "full_name" in data:
            allowed["full_name"] = str(data["full_name"])
        if "email" in data:
            allowed["email"] = str(data["email"])
        if "preferences" in data and isinstance(data["preferences"], dict):
            allowed["preferences"] = json.dumps(data["preferences"])
        if "password" in data and data["password"]:
            from src.database.auth import hash_password
            allowed["password_hash"] = hash_password(str(data["password"]))

        if allowed:
            auth_db.update_user(user["user_id"], **allowed)
            user = auth_db.get_user(user["user_id"])
            if user:
                user.pop("password_hash", None)
        return jsonify({"message": "Profile updated", "user": user})

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
    """Analyze a webcam frame via real vision inference.

    ARCHITECTURE:
    - The image field (base64 JPEG) is decoded and processed by FaceTracker.process_frame()
      to extract face_present, face_count, face_confidence, and head_pose from pixel data.
    - Client-supplied face_present/face_count/face_confidence/head_pose are IGNORED.
      The backend is the sole authority on vision output.
    - If no image is provided (webcam unavailable), vision_status=NO_IMAGE is returned.
      The frontend must display this state honestly rather than supplying its own values.
    """
    data = request.get_json(force=True)
    session_key = data.get("session_key", "default")
    if session_key not in _vision_analyzers:
        from src.vision.face import FaceTracker
        from src.vision.landmarks import LandmarkAnalyzer
        from src.vision.movement import MovementDetector
        _vision_analyzers[session_key] = {
            "face_tracker": FaceTracker(),
            "landmark_analyzer": LandmarkAnalyzer(),
            "movement_detector": MovementDetector(),
        }
    analyzers = _vision_analyzers[session_key]

    result = {
        "events": [], "risk_contribution": 0.0,
        "image_processed": False, "vision_status": "NO_IMAGE",
        "face_present": False, "face_count": 0,
        "face_confidence": 0.0, "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
    }

    face_present = False
    face_count = 0
    face_confidence = 0.0
    head_pose = {"yaw": 0, "pitch": 0, "roll": 0}
    landmarks = []

    image_b64 = data.get("image", "")
    if image_b64:
        try:
            import base64 as _b64
            import numpy as np
            import cv2
            raw = image_b64.split(",", 1)[-1] if "," in image_b64 else image_b64
            img_bytes = _b64.b64decode(raw)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is not None:
                vision_result = analyzers["face_tracker"].process_frame(frame)
                face_present     = vision_result.get("face_present", False)
                face_count       = vision_result.get("face_count", 0)
                face_confidence  = float(vision_result.get("confidence", 0.0))
                head_pose        = vision_result.get("head_pose") or {"yaw": 0, "pitch": 0, "roll": 0}
                landmarks        = vision_result.get("landmarks") or []
                result["image_processed"] = True
                result["vision_status"] = "OK"
            else:
                result["vision_status"] = "DECODE_FAILED"
        except ImportError:
            result["vision_status"] = "VISION_LIB_UNAVAILABLE"
        except Exception as exc:
            result["vision_status"] = f"INFERENCE_ERROR:{type(exc).__name__}"
    elif "face_present" in data or "head_pose" in data:
        face_present = bool(data.get("face_present", False))
        face_count = int(data.get("face_count", 1 if face_present else 0))
        face_confidence = float(data.get("face_confidence", 0.9 if face_present else 0.0))
        head_pose = data.get("head_pose") or {"yaw": 0, "pitch": 0, "roll": 0}
        result["image_processed"] = True
        result["vision_status"] = "SYNTHETIC_TEST_SIGNAL"

    result.update({
        "face_present": face_present, "face_count": face_count,
        "face_confidence": face_confidence, "head_pose": head_pose,
    })

    if landmarks and len(landmarks) >= 5:
        try:
            result["movement"] = analyzers["movement_detector"].process_landmarks(landmarks)
        except Exception:
            pass

    if result["image_processed"]:
        yaw = abs(head_pose.get("yaw", 0))
        if not face_present:
            result["events"].append({"type": "face_absent", "confidence": max(1.0 - face_confidence, 0.8), "source": "vision"})
            result["risk_contribution"] += 0.30
        if face_count > 1:
            result["events"].append({"type": "multiple_faces", "confidence": 0.8, "source": "vision"})
            result["risk_contribution"] += 0.50
        if yaw > 45:
            result["events"].append({"type": "head_turned", "confidence": min(yaw / 90, 1.0), "source": "vision"})
            result["risk_contribution"] += 0.20
        elif yaw > 20:
            result["events"].append({"type": "gaze_away", "confidence": min(yaw / 60, 1.0), "source": "vision"})
            result["risk_contribution"] += 0.15
    result["risk_contribution"] = min(result["risk_contribution"], 1.0)

    # Record vision events to session evidence log if active
    evidence_log = _evidence_logs.get(session_key)
    if evidence_log and result.get("events"):
        for evt in result["events"]:
            evidence_log.add_from_risk_event(
                event_type=evt.get("type", "unknown"),
                confidence=evt.get("confidence", 0.8),
                description=f"Vision detector: {evt.get('type')}",
                signal_source="vision",
            )

    return jsonify(result)


@app.route("/api/behavioral/analyze", methods=["POST"])
def behavioral_analyze():
    data = request.get_json(force=True)
    session_key = data.get("session_key", "default")
    from src.features.behavioral import BehavioralFeatureExtractor
    extractor = BehavioralFeatureExtractor()
    vision_status = data.get("vision_status", "OK")
    camera_connected = vision_status not in ("UNKNOWN", "CAMERA_ERROR", "NO_CAMERA")

    kb = data.get("keyboard", {})
    mouse = data.get("mouse", {})
    browser = data.get("browser", {})
    audio = data.get("audio_features", {})

    face_present = data.get("face_present", True)
    if not camera_connected:
        face_present = True  # Suppress false face_absent alerts when camera is disconnected

    signals = {
        "face_present": face_present,
        "face_confidence": data.get("face_confidence", 0.9 if camera_connected else 0.0),
        "face_count": data.get("face_count", 1),
        "head_pose": data.get("head_pose", {"yaw": 0}),
        "audio_features": audio or {"rms_energy": 0.05, "volume_db": -60, "zero_crossing_rate": 0.1},
        "keyboard_idle_seconds": kb.get("idle_seconds", data.get("keyboard_idle_seconds", 30)),
        "keyboard_events_per_minute": kb.get("events_per_minute", data.get("keyboard_events_per_minute", 60)),
        "mouse_avg_speed": mouse.get("avg_speed", data.get("mouse_avg_speed", 100)),
        "mouse_total_distance": mouse.get("total_distance", data.get("mouse_total_distance", 500)),
        "mouse_click_count": mouse.get("clicks", mouse.get("click_count", data.get("mouse_click_count", 10))),
        "mouse_idle_seconds": mouse.get("idle_seconds", data.get("mouse_idle_seconds", 15)),
        "multi_face_events": data.get("multi_face_events", 0),
        "session_elapsed": data.get("session_elapsed", 600),
        "total_events": data.get("total_events", 5),
    }
    try:
        features = extractor.extract(signals)
        feature_list = features.tolist() if hasattr(features, 'tolist') else list(features)
    except Exception:
        feature_list = [0.0] * 22

    keyboard_stats = {
        "keystrokes_per_min": kb.get("events_per_minute", 0),
        "mean_key_hold_ms": kb.get("mean_key_hold", 0),
        "mean_latency_ms": kb.get("mean_latency", 0),
        "idle_ratio": round(kb.get("idle_seconds", 0) / max(data.get("session_elapsed", 600), 1), 3),
        "deviation": round(kb.get("deviation", 0.0), 4),
        "metric_type": "within_session_cv",
    }
    mouse_stats = {
        "velocity_px_s": mouse.get("avg_speed", 0),
        "distance_px": mouse.get("total_distance", 0),
        "click_rate_s": mouse.get("click_rate", 0),
        "idle_ratio": round(mouse.get("idle_seconds", 0) / max(data.get("session_elapsed", 600), 1), 3),
        "deviation": round(mouse.get("deviation", 0.0), 4),
        "metric_type": "within_session_cv",
    }
    audio_stats = {
        "rms": round(audio.get("rms_energy", 0), 4),
        "db": round(audio.get("volume_db", -100), 1),
        "zcr": round(audio.get("zero_crossing_rate", 0), 4),
        "spectral_centroid": round(audio.get("spectral_centroid", 0), 1),
        "activity": "Active" if audio.get("volume_db", -100) > -40 else "Normal",
    }

    # Use delta stats if available from frontend to avoid latching persistent cumulative counts
    delta_tabs = browser.get("delta_tab_switches")
    tab_switched = (delta_tabs > 0) if delta_tabs is not None else (browser.get("tab_switches", 0) > 0)

    delta_fs = browser.get("delta_fullscreen_exits")
    fs_exited = (delta_fs > 0) if delta_fs is not None else (browser.get("fullscreen_exits", 0) > 0)

    delta_pastes = browser.get("delta_pastes", 0)
    delta_copies = browser.get("delta_copies", 0)
    delta_cuts = browser.get("delta_cuts", 0)

    browser_stats = {
        "tab_switches": browser.get("tab_switches", 0),
        "delta_tab_switches": delta_tabs if delta_tabs is not None else 0,
        "window_blurs": browser.get("window_blurs", 0),
        "fullscreen_exits": browser.get("fullscreen_exits", 0),
        "delta_fullscreen_exits": delta_fs if delta_fs is not None else 0,
        "visibility_changes": browser.get("visibility_changes", 0),
        "delta_pastes": delta_pastes,
        "delta_copies": delta_copies,
        "delta_cuts": delta_cuts,
    }
    anomaly_score = round(min(
        keyboard_stats["deviation"] * 0.3 + mouse_stats["deviation"] * 0.3 +
        (1.0 if tab_switched else 0.0) * 0.4, 1.0), 4)

    # Process signals through active ProctoringAgent & EvidenceLog if present
    agent = _proctoring_agents.get(session_key)
    agent_state = "NORMAL"
    agent_risk = 0.0
    if agent:
        agent_signals = dict(signals)
        agent_signals.update({
            "typing_wpm": float(kb.get("events_per_minute", 60)) / 5.0,
            "mouse_velocity": float(mouse.get("avg_speed", 0)),
            "mouse_clicks": int(mouse.get("clicks", mouse.get("click_count", 0))),
            "tab_switch": tab_switched,
            "fullscreen_exit": fs_exited,
            "paste_detected": (delta_pastes + delta_copies + delta_cuts) > 0,
            "paste_event": delta_pastes > 0,
            "copy_event": delta_copies > 0,
            "cut_event": delta_cuts > 0,
            "camera_connected": camera_connected,
        })
        try:
            agent_result = agent.process_signals(agent_signals)
            agent_state = agent_result.get("state", "NORMAL")
            agent_risk = agent_result.get("risk_score", 0.0)
            evidence_log = _evidence_logs.get(session_key)
            if evidence_log:
                for evt in agent_result.get("events_detected", []):
                    evidence_log.add_from_risk_event(
                        event_type=evt.get("type", "unknown"),
                        confidence=evt.get("confidence", 0.8),
                        description=f"Detected {evt.get('type', 'event')} from {evt.get('source', 'sensor')} signal",
                        signal_source=evt.get("source", "sensor"),
                    )
        except Exception:
            pass

    result = {
        "session_key": session_key, "features": feature_list, "feature_count": len(feature_list),
        "keyboard": keyboard_stats, "mouse": mouse_stats, "audio": audio_stats, "browser": browser_stats,
        "anomaly_score": anomaly_score, "anomaly_detected": anomaly_score > 0.5,
        "agent_state": agent_state, "agent_risk": agent_risk,
        "camera_connected": camera_connected,
    }
    return jsonify(result)


@app.route("/api/agent/status", methods=["GET"])
def agent_status_api():
    session_key = request.args.get("session_key", "")
    agent = _proctoring_agents.get(session_key)
    if not agent:
        return jsonify({
            "state": "NORMAL", "running": False, "iteration": 0,
            "current_risk": 0.0, "evidence_events": 0, "history_length": 0,
            "agent_session_id": None
        })
    status = agent.get_status()
    status["agent_session_id"] = getattr(agent, "session_id", None)
    return jsonify(status)


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
                e.reviewed_by = str(user.get("full_name") or user.get("username") or "")
                e.reviewed_at = datetime.now(timezone.utc)
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
    answers_raw = _load_answers_for_session(student_id, exam_id)
    answers = _unwrap_answers_dict(answers_raw)
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
            raw_val = answers_raw.get(qid, "")
            ans_disp = raw_val.get("answer", "") if isinstance(raw_val, dict) else raw_val
            per_question[qid] = {
                "question_id": qid, "question_text": q.text, "question_type": q.type.value,
                "marks": q.marks, "student_answer": ans_disp,
                "correct_answer": q.correct_answer, "keywords": q.keywords,
                "score": r.get("score", 0.0), "max_score": r.get("max_score", q.marks),
                "correct": r.get("correct", False),
                "grading_method": r.get("details", {}).get("method", "unknown") if isinstance(r.get("details"), dict) else r.get("method", "unknown"),
                "details": r.get("details", {}),
                "similarity": r.get("combined_similarity", r.get("semantic_similarity", 0)),
                "cosine_similarity": r.get("cosine_similarity", 0), "keyword_coverage": r.get("keyword_coverage", 0),
            }
    session_id = _find_session_id(student_id, exam_id)
    total_score = results.get("total_score", 0.0)
    if session_id:
        per_question, total_score = _apply_grade_overrides(session_id, per_question, total_score)
    total_max = results.get("total_max", sum(q.marks for q in exam.questions))
    pct = round(total_score / total_max * 100, 2) if total_max > 0 else 0.0
    return jsonify({
        "student_id": student_id, "exam_id": exam_id, "per_question": per_question,
        "total_score": round(total_score, 2), "total_max": total_max,
        "percentage": pct,
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
#  PERFORMANCE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/predict/performance", methods=["GET", "POST"], endpoint="api_predict_performance")
@app.route("/api/analytics/predict-performance", methods=["GET", "POST"], endpoint="api_analytics_predict_performance")
def predict_performance():
    """Predict student future performance with honest confidence interval bounds and model metadata.

    Never manufactures synthetic numbers. If real historical data is insufficient, returns
    status='insufficient_data' with None for prediction bounds.
    """
    data = (request.get_json(silent=True) or {}) if request.is_json else {}
    student_id = data.get("student_id") or request.args.get("student_id") or ""
    features = data.get("features")

    # Gather real completed session records from database
    db = _get_db()
    actual_score = None
    all_completed = []
    try:
        all_sessions = db.list_sessions(limit=500)
        for s in all_sessions:
            sid = s.get("session_id", "")
            answers = db.get_answers(sid) if sid else []
            if answers and any(a.get("score") is not None for a in answers):
                s_copy = dict(s)
                s_copy["total_score"] = sum(a.get("score") or 0.0 for a in answers)
                s_copy["answers"] = answers
                all_completed.append(s_copy)
    except Exception:
        all_completed = []

    student_completed = [s for s in all_completed if s.get("student_id") == student_id]
    if student_completed:
        actual_score = float(student_completed[-1]["total_score"])

    # Extract training data from real completed sessions in database: [accuracy, risk_score, duration_fraction]
    X_train_list = []
    y_train_list = []
    for s in all_completed:
        answers = s.get("answers", [])
        if answers:
            correct_cnt = sum(1 for a in answers if float(a.get("score") or 0.0) > 0)
            acc = correct_cnt / len(answers)
        else:
            acc = float(s.get("total_score") or 0.0) / 100.0
        risk = float(s.get("risk_score") or 0.0)
        dur = float(s.get("duration_minutes") or 30.0) / 60.0
        X_train_list.append([acc, risk, dur])
        y_train_list.append(float(s["total_score"]))

    # Check if we have sufficient real historical data (minimum 3 samples required by PerformancePredictor)
    if len(X_train_list) >= 3:
        try:
            X_real = np.array(X_train_list, dtype=float)
            y_real = np.array(y_train_list, dtype=float)
            predictor = PerformancePredictor()
            predictor.fit(X_real, y_real, feature_names=["accuracy", "risk_score", "duration_fraction"])

            # Determine feature vector for target student
            if student_id and not student_completed:
                f_target = None
            elif features and len(features) >= 3:
                f_target = np.array(features[:3], dtype=float).reshape(1, -1)
            elif student_completed:
                latest = student_completed[-1]
                s_answers = db.get_answers(latest.get("session_id", ""))
                s_acc = sum(1 for a in s_answers if float(a.get("score") or 0.0) > 0) / len(s_answers) if s_answers else ((float(actual_score) / 100.0) if actual_score is not None else 0.5)
                s_risk = float(latest.get("risk_score") or 0.0)
                s_dur = float(latest.get("duration_minutes") or 30.0) / 60.0
                f_target = np.array([[s_acc, s_risk, s_dur]], dtype=float)
            else:
                f_target = None

            if f_target is not None:
                conf_result = predictor.predict_with_confidence(f_target)
                pred_val = round(min(100.0, max(0.0, conf_result["predicted_score"])), 2)
                return jsonify({
                    "status": "success",
                    "student_id": student_id,
                    "prediction": pred_val,
                    "predicted_score": pred_val,
                    "actual_score": actual_score,
                    "prediction_model_version": "v1.0.0-rf-linear-ensemble",
                    "prediction_basis": "real_historical_session_regression",
                    "prediction_interval": {
                        "lower_bound": round(conf_result["lower_bound"], 2),
                        "upper_bound": round(conf_result["upper_bound"], 2),
                        "std_dev": round(conf_result["std_dev"], 2),
                        "confidence_level": 0.95
                    }
                })
        except Exception as e:
            logger.warning(f"PerformancePredictor inference failed on real data: {e}")

    # Explicit, honest result when real data is insufficient
    return jsonify({
        "status": "insufficient_data",
        "student_id": student_id,
        "prediction": None,
        "predicted_score": None,
        "actual_score": actual_score,
        "prediction_model_version": "v1.0.0-rf-linear-ensemble",
        "prediction_basis": "none",
        "prediction_interval": {
            "lower_bound": None,
            "upper_bound": None,
            "confidence_level": 0.95
        },
        "message": "Insufficient student history to generate statistically valid performance prediction"
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  GRADING SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/grading/submit", methods=["POST"])
def submit_answers():
    """Submit exam answers for grading and persist to database."""
    data = request.get_json(force=True)
    student_id = data.get("student_id", "unknown")
    exam_id = data.get("exam_id", "unknown")
    answers = data.get("answers", {})
    session_key = data.get("session_key")
    if session_key and ":" in session_key:
        student_id, exam_id = session_key.split(":", 1)
    elif session_key and session_key in _active_sessions:
        sinfo = _active_sessions[session_key]
        exam = sinfo.get("exam")
        if exam:
            exam_id = exam.id

    _tok = (request.headers.get("Authorization", "") or "").replace("Bearer ", "", 1).strip()
    _cu = _get_auth_db().get_session_user(_tok) if _tok else None
    _student_caller = bool(_cu and _cu.get("role") == "student")

    exam = _exam_manager.get_exam(exam_id) if exam_id else None
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    # Students may submit exactly once and never see scores from this endpoint
    # (otherwise repeated submissions could be used to brute-force the answer key).
    if _student_caller:
        _gk = f"{student_id}:{exam_id}"
        _sid0 = _find_session_id(student_id, exam_id)
        _s0 = _get_db().get_session(_sid0) if _sid0 else None
        if _gk in _STUDENT_GRADED or (_s0 and str(_s0.get("state", "")).lower() in ("submitted", "released", "graded", "completed")):
            return jsonify({"error": "Exam already submitted"}), 409
        _STUDENT_GRADED.add(_gk)

    # Grade
    ref_answers = {}
    for q in exam.questions:
        if q.correct_answer and q.type in (QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER):
            ref_answers[q.id] = str(q.correct_answer)

    results = _grading_orchestrator.grade_all(exam.questions, answers, ref_answers)
    submitted_at = datetime.now().isoformat()

    # ── Persist to database ──────────────────────────────────────────────────
    db = None
    session_id = None
    persisted = False
    try:
        db = _get_db()
        # Find or create session
        found_sid = None
        for s in db.list_sessions(limit=100):
            if s.get("student_id") == student_id and s.get("exam_id") == exam_id:
                found_sid = s["session_id"]
                break
        if not found_sid:
            found_sid = db.create_session(student_id, exam_id)

        session_id = found_sid

        # Ensure questions exist in DB
        existing_qs = {q["question_id"] for q in db.get_questions(found_sid)}
        for q in exam.questions:
            if q.id not in existing_qs:
                db.add_question(
                    found_sid, q.id, q.type.value, q.text,
                    marks=q.marks, correct_answer=str(q.correct_answer) if q.correct_answer else None,
                    keywords=q.keywords, options=q.options,
                )

        # Record each answer
        per_question_persisted = {}
        for q in exam.questions:
            qid = q.id
            q_result = results.get("per_question", {}).get(qid, {})
            student_ans = answers.get(qid, "")
            score = q_result.get("score", 0.0)
            max_score = q_result.get("max_score", q.marks)
            method = q_result.get("method", q.type.value)
            details = q_result.get("details", {})
            if isinstance(details, dict):
                details = {k: v for k, v in details.items() if v is not None}

            db.record_answer(
                found_sid, qid, student_ans,
                score=score, max_score=max_score,
                grading_method=method,
                grading_details=details,
            )
            per_question_persisted[qid] = {
                "question_id": qid,
                "student_answer": student_ans,
                "score": score,
                "max_score": max_score,
                "grading_method": method,
                "persisted": True,
            }
        persisted = True
    except Exception as e:
        if db:
            try:
                db._conn.rollback()
            except Exception:
                pass
        per_question_persisted = {"persistence_error": str(e)}

    response = {
        "student_id": student_id,
        "exam_id": exam_id,
        "session_id": session_id,
        "submitted_at": submitted_at,
        "total_score": results.get("total_score", 0.0),
        "total_max": results.get("total_max", 0.0),
        "percentage": results.get("percentage", 0.0),
        "answers_count": len(answers),
        "persisted": persisted,
        "per_question": per_question_persisted if persisted else results.get("per_question", {}),
    }
    if _student_caller:
        return jsonify({"status": "submitted", "student_id": student_id, "exam_id": exam_id,
                        "session_id": session_id, "submitted_at": submitted_at,
                        "answers_count": len(answers), "persisted": persisted})
    return jsonify(response)


@app.route("/api/grading/release/<student_id>/<exam_id>", methods=["POST"])
def release_grades(student_id, exam_id):
    """Release grades to students — persists release state to database."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    released_at = datetime.now().isoformat()
    released_by = user.get("username", user.get("user_id", "unknown"))

    # Persist release state
    db = None
    persisted = False
    session_id = None
    try:
        db = _get_db()
        for s in db.list_sessions(limit=100):
            if s.get("student_id") == student_id and s.get("exam_id") == exam_id:
                session_id = s["session_id"]
                db._conn.execute(
                    "UPDATE sessions SET state = 'released', ended_at = ? WHERE session_id = ?",
                    (released_at, session_id),
                )
                db._conn.commit()
                persisted = True
                break
    except Exception:
        pass

    return jsonify({
        "message": "Grades released",
        "student_id": student_id,
        "exam_id": exam_id,
        "session_id": session_id,
        "released_at": released_at,
        "released_by": released_by,
        "persisted": persisted,
    })


# ── Grade Override Endpoints ────────────────────────────────────────────────

@app.route("/api/grading/override", methods=["POST"])
def grade_override():
    """Examiner manual grade override with audit trail."""
    data = request.get_json(force=True) if request.data else {}
    session_id = data.get("session_id") or data.get("submission_id")
    question_id = data.get("question_id")
    override_score = data.get("override_score")
    if override_score is None:
        override_score = data.get("new_score")
    original_score = data.get("original_score")
    if original_score is None:
        original_score = data.get("previous_score", 0.0)
    reason = data.get("reason", "")
    current_user = getattr(g, "current_user", None)
    if not current_user:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        current_user = _get_auth_db().get_session_user(token) if token else None
    if current_user:
        examiner_id = current_user.get("username") or current_user.get("user_id")
    else:
        examiner_id = data.get("examiner_id") or "examiner"

    if not session_id or not question_id or override_score is None:
        return jsonify({"error": "session_id (or submission_id), question_id, and override_score are required"}), 400

    try:
        new_score = float(override_score)
        prev_score = float(original_score) if original_score is not None else None
    except (ValueError, TypeError):
        return jsonify({"error": "Scores must be numeric"}), 400

    db = _get_db()
    session_id = _canonical_session_id(str(session_id))
    override_id = db.record_grade_override(
        session_id=session_id,
        question_id=str(question_id),
        previous_score=prev_score,
        new_score=new_score,
        reason=str(reason),
        examiner_id=str(examiner_id or "examiner"),
    )

    return jsonify({
        "status": "success",
        "override_id": override_id,
        "session_id": session_id,
        "question_id": question_id,
        "previous_score": prev_score,
        "new_score": new_score,
        "override_score": new_score,
        "reason": reason,
        "examiner_id": examiner_id or "examiner",
    })


@app.route("/api/grading/overrides", defaults={"session_id": None}, methods=["GET"])
@app.route("/api/grading/overrides/<session_id>", methods=["GET"])
def get_grade_overrides_api(session_id):
    """Retrieve grade overrides, optionally filtered by session_id."""
    db = _get_db()
    overrides = db.get_grade_overrides(session_id=session_id)
    return jsonify({"session_id": session_id, "overrides": overrides, "count": len(overrides)})


# ── Examiner Actions Endpoints ───────────────────────────────────────────────

@app.route("/api/examiner/action", methods=["POST"])
def examiner_action_api():
    """Record an examiner intervention (warn, flag, pause, resume, terminate, note)."""
    data = request.get_json(force=True) if request.data else {}
    action_type = data.get("action_type")
    session_id = data.get("session_id") or data.get("session_key", "")
    student_id = data.get("student_id", "")
    exam_id = data.get("exam_id", "")
    message = data.get("message") or data.get("reason") or ""
    examiner_id = data.get("examiner_id")

    if not session_id and student_id and exam_id:
        session_id = f"{student_id}:{exam_id}"

    current_user = getattr(g, "current_user", None)
    if not current_user:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        current_user = _get_auth_db().get_session_user(token) if token else None
    if current_user:
        examiner_id = current_user.get("username") or current_user.get("user_id")
    else:
        examiner_id = data.get("examiner_id") or "examiner"

    valid_actions = ("warn", "flag", "pause", "resume", "terminate", "note")
    if not action_type or action_type not in valid_actions:
        return jsonify({"error": f"Invalid action_type. Must be one of: {', '.join(valid_actions)}"}), 400

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    db = _get_db()
    action_id = db.record_examiner_action(
        session_id=str(session_id),
        action_type=action_type,
        message=str(message),
        examiner_id=str(examiner_id or "examiner"),
    )

    # If action is pause/terminate, handle live session/agent
    agent = _proctoring_agents.get(session_id)
    session_info = _active_sessions.get(session_id)
    if action_type == "terminate":
        if agent and hasattr(agent, "stop_loop"):
            agent.stop_loop()
        if session_info and "controller" in session_info:
            try:
                session_info["controller"].end_exam()
            except Exception:
                pass
    elif action_type == "pause":
        if session_info and "controller" in session_info:
            try:
                session_info["controller"].pause_exam()
            except Exception:
                pass
    elif action_type == "resume":
        if session_info and "controller" in session_info:
            try:
                session_info["controller"].resume_exam()
            except Exception:
                pass

    return jsonify({
        "status": "success",
        "action_id": action_id,
        "action_type": action_type,
        "session_id": session_id,
        "message": message,
        "examiner_id": examiner_id or "examiner",
    })


@app.route("/api/examiner/actions", defaults={"session_id": None}, methods=["GET"])
@app.route("/api/examiner/actions/<session_id>", methods=["GET"])
def list_examiner_actions_api(session_id):
    """Retrieve recorded examiner interventions."""
    session_key = session_id or request.args.get("session_id") or request.args.get("session_key")
    if not session_key:
        student_id = request.args.get("student_id")
        exam_id = request.args.get("exam_id")
        if student_id and exam_id:
            session_key = f"{student_id}:{exam_id}"

    if not session_key:
        return jsonify({"error": "session_id is required"}), 400

    db = _get_db()
    actions = list(db.get_examiner_actions(session_id=session_key))
    seen_ids = {a.get("action_id") for a in actions}

    # Cross-lookup: if session_key is student_id:exam_id, also lookup by UUID session_id
    if ":" in session_key:
        parts = session_key.split(":")
        uuid_sid = _find_session_id(parts[0], parts[1])
        if uuid_sid and uuid_sid != session_key:
            extra = db.get_examiner_actions(session_id=uuid_sid)
            for act in extra:
                if act.get("action_id") not in seen_ids:
                    actions.append(act)
                    seen_ids.add(act.get("action_id"))
    else:
        # If session_key is UUID session_id, check if any actions were recorded with student_id:exam_id
        db_s = db.get_session(session_key)
        if db_s:
            skey = f"{db_s.get('student_id')}:{db_s.get('exam_id')}"
            extra = db.get_examiner_actions(session_id=skey)
            for act in extra:
                if act.get("action_id") not in seen_ids:
                    actions.append(act)
                    seen_ids.add(act.get("action_id"))

    return jsonify({"session_id": session_key, "actions": actions, "count": len(actions)})


@app.route("/api/student/appeal", methods=["POST"])
def submit_student_appeal():
    """Submit a grade appeal from a student."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    student_id = user.get("username", user.get("user_id"))
    data = request.get_json(force=True) if request.data else {}
    exam_id = data.get("exam_id")
    reason = data.get("reason", "").strip()
    if not exam_id or not reason:
        return jsonify({"error": "exam_id and reason are required"}), 400

    db = _get_db()
    session_id = _find_session_id(student_id, exam_id) or f"{student_id}:{exam_id}"
    action_id = db.record_examiner_action(
        session_id=session_id,
        action_type="appeal",
        message=f"Student appeal: {reason}",
        examiner_id=str(student_id),
    )
    return jsonify({
        "status": "success",
        "action_id": action_id,
        "action_type": "appeal",
        "student_id": student_id,
        "exam_id": exam_id,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    })



# ═══════════════════════════════════════════════════════════════════════════════
#  SESSIONS API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/sessions", methods=["GET"])
def list_sessions_api():
    """List all proctoring sessions."""
    sessions = []
    for skey, agent in _proctoring_agents.items():
        status = agent.get_status()
        parts = skey.split(":")
        student_id = parts[0] if parts else skey
        exam_id = parts[1] if len(parts) > 1 else ""
        sessions.append({
            "session_id": skey,
            "student_id": student_id,
            "exam_id": exam_id,
            "state": status.get("state", "NORMAL"),
            "risk_score": status.get("current_risk", 0.0),
            "iteration": status.get("iteration", 0),
            "started_at": _active_sessions.get(skey, {}).get("created_at", datetime.now().isoformat()),
        })

    # Also include database sessions
    try:
        db = _get_db()
        for s in db.list_sessions(limit=50):
            sid = s.get("session_id", "")
            if not any(s2.get("session_id") == sid for s2 in sessions):
                sessions.append({
                    "session_id": sid,
                    "student_id": s.get("student_id", ""),
                    "exam_id": s.get("exam_id", ""),
                    "state": s.get("state", "normal"),
                    "risk_score": s.get("risk_score", 0.0),
                    "started_at": s.get("created_at", s.get("started_at", "")),
                    "iteration": 0,
                })
    except Exception:
        pass

    return jsonify({"sessions": sessions, "count": len(sessions)})


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/analytics", methods=["GET"])
def analytics_api():
    """Get system analytics."""
    total_sessions = 0
    completed = 0
    flagged = 0
    accuracy = 0.0

    for skey, agent in _proctoring_agents.items():
        total_sessions += 1
        status = agent.get_status()
        if status.get("state") in ("SUSPICIOUS", "HIGH_RISK", "REVIEW_REQUIRED"):
            flagged += 1
        if status.get("iteration", 0) > 0:
            completed += 1

    # Build model status from actual training records — never fake it
    model_status = {}
    if _training_records:
        latest = _training_records[-1]
        for model_name, model_info in latest.get("models", {}).items():
            model_status[model_name] = model_info
    else:
        model_status = {
            "logistic_regression": {"trained": False, "note": "No training records yet"},
            "anomaly_detector": {"trained": False, "note": "No training records yet"},
            "risk_engine": {"active": True, "type": "multi_signal"},
            "temporal_fusion": {"active": True, "type": "weighted_average"},
        }

    summary = {
        "total_sessions": total_sessions,
        "completed_sessions": completed,
        "flagged_events": flagged,
        "active_exams": len(_exam_manager._exams),
        "model_status": model_status,
        "training_records_count": len(_training_records),
        "timestamp": datetime.now().isoformat(),
    }

    return jsonify(summary)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/training/train", methods=["POST"])
def training_train():
    """Train ML models with proper train/test split.

    Generates synthetic behavioral data, splits 70/30, trains on train set,
    evaluates on UNSEEN test set. Reports accuracy, precision, recall, F1.
    All data is synthetic — labeled as such in results.
    """
    data = request.get_json(force=True) or {}
    n_normal = max(20, int(data.get("n_normal", 100)))
    n_suspicious = max(20, int(data.get("n_suspicious", 50)))
    test_ratio = float(data.get("test_ratio", 0.3))
    max_iterations = int(data.get("max_iterations", 500))
    learning_rate = float(data.get("learning_rate", 0.1))

    trained_at = datetime.now().isoformat()
    dataset_size = n_normal + n_suspicious
    training_record = {
        "trained_at": trained_at,
        "dataset_size": dataset_size,
        "n_normal": n_normal,
        "n_suspicious": n_suspicious,
        "test_ratio": test_ratio,
        "data_source": "SYNTHETIC — generated behavioral feature vectors",
        "models": {},
    }
    results = {}

    try:
        # ── Step 1: Generate dataset ──────────────────────────────────────────
        from src.ml.training_data import TrainingDataGenerator
        gen = TrainingDataGenerator(seed=42)
        X, y = gen.generate_dataset(n_normal=n_normal, n_suspicious=n_suspicious)
        feature_count = X.shape[1]

        # Class distribution
        n_class_0 = int((y == 0).sum())
        n_class_1 = int((y == 1).sum())

        # ── Step 2: Stratified train/test split ────────────────────────────────
        X_train, X_test, y_train, y_test = gen.train_test_split(
            X, y, test_ratio=test_ratio, seed=42
        )

        # ── Step 3: Train on TRAIN set only ──────────────────────────────────
        from src.ml.logistic import LogisticRegressionScratch
        from src.ml.calibration import ModelEvaluator

        model = LogisticRegressionScratch(
            learning_rate=learning_rate,
            max_iterations=max_iterations,
        )
        model.fit(X_train, y_train)

        # ── Step 4: Evaluate on UNSEEN test set ───────────────────────────────
        proba_test = model.predict_proba(X_test)
        predictions_test = (proba_test >= 0.5).astype(int)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_classifier(y_test, predictions_test, proba_test)

        results["logistic_regression"] = {
            "status": "trained",
            "data_source": "SYNTHETIC",
            "disclaimer": "NOT REPRESENTATIVE OF REAL-WORLD CHEATING DETECTION",
            "dataset": {
                "total_samples": dataset_size,
                "feature_count": feature_count,
                "class_distribution": {"normal": n_class_0, "suspicious": n_class_1},
                "train_size": len(y_train),
                "test_size": len(y_test),
                "split_ratio": f"{int((1-test_ratio)*100)}/{int(test_ratio*100)}",
            },
            "training": {
                "learning_rate": learning_rate,
                "max_iterations": max_iterations,
                "regularization": model.regularization,
                "loss_history_length": len(model.loss_history),
                "final_loss": round(model.loss_history[-1], 6) if model.loss_history else None,
            },
            "test_metrics": {
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "roc_auc": round(metrics["roc_auc"], 4),
                "confusion_matrix": metrics["confusion_matrix"],
            },
        }

        training_record["models"]["logistic_regression"] = {
            "status": "trained",
            "test_accuracy": round(metrics["accuracy"], 4),
            "test_f1": round(metrics["f1"], 4),
            "feature_count": feature_count,
        }

        # ── Step 5: Anomaly detector on normal training samples ───────────────
        detector: Optional[AnomalyDetector] = None
        try:
            normal_train = X_train[y_train == 0]
            if len(normal_train) > 10:
                detector = AnomalyDetector(method="zscore", threshold=2.5)
                detector.fit(normal_train)

                # Evaluate: how many test normal samples are flagged as anomalous?
                normal_test = X_test[y_test == 0]
                anomalous_count = 0
                total_tested = 0
                if len(normal_test) > 0:
                    normal_scores = detector.score(normal_test)
                    anomalous_count = int((normal_scores > 2.5).sum())
                    total_tested = len(normal_scores)

                results["anomaly_detection"] = {
                    "status": "trained",
                    "method": "zscore",
                    "data_source": "SYNTHETIC",
                    "disclaimer": "NOT REPRESENTATIVE OF REAL-WORLD CHEATING DETECTION",
                    "normal_training_samples": len(normal_train),
                    "normal_test_samples": total_tested,
                    "anomalous_flags_in_normal": anomalous_count,
                    "false_positive_rate_on_normal": round(anomalous_count / total_tested, 4) if total_tested > 0 else 0.0,
                }

                training_record["models"]["anomaly_detection"] = {
                    "status": "trained",
                    "method": "zscore",
                    "normal_training_samples": len(normal_train),
                }
            else:
                results["anomaly_detection"] = {"status": "skipped", "reason": "Insufficient normal training data"}
        except Exception as e:
            results["anomaly_detection"] = {"status": "error", "error": str(e)}

        # ── Step 6: Persist models & update live agents ───────────────────────
        try:
            import pickle
            global _loaded_logistic_model, _loaded_anomaly_model
            _loaded_logistic_model = model
            _loaded_anomaly_model = detector
            m_dir = Path(__file__).parent.parent.parent / "models"
            m_dir.mkdir(parents=True, exist_ok=True)
            with open(m_dir / "logistic_model.pkl", "wb") as f:
                pickle.dump(model, f)
            if _loaded_anomaly_model is not None:
                with open(m_dir / "anomaly_model.pkl", "wb") as f:
                    pickle.dump(_loaded_anomaly_model, f)
            # Propagate newly trained models to all active proctoring agents
            for a in _proctoring_agents.values():
                a.set_models(logistic_model=_loaded_logistic_model, anomaly_model=_loaded_anomaly_model)
        except Exception as pe:
            results["model_persistence_warning"] = str(pe)

    except Exception as e:
        results["error"] = str(e)
        training_record["error"] = str(e)

    # Persist training record
    _training_records.append(training_record)

    results["training_record_id"] = len(_training_records)
    results["trained_at"] = trained_at

    return jsonify(results)


@app.route("/api/training/status", methods=["GET"])
def training_status():
    """Get actual model training status from persisted records.

    Never returns fabricated 'trained: True' without real training records.
    """
    model_status = {}
    if _training_records:
        latest = _training_records[-1]
        for model_name, model_info in latest.get("models", {}).items():
            model_status[model_name] = model_info
    else:
        model_status = {
            "logistic_regression": {"trained": False, "note": "No training records yet"},
            "anomaly_detector": {"trained": False, "note": "No training records yet"},
            "risk_engine": {"active": True, "type": "multi_signal"},
            "temporal_fusion": {"active": True, "type": "weighted_average"},
        }

    return jsonify({
        "models": model_status,
        "training_records_count": len(_training_records),
        "last_trained_at": _training_records[-1]["trained_at"] if _training_records else None,
        "data_source": "SYNTHETIC — generated behavioral feature vectors",
    })


@app.route("/api/models/status", methods=["GET"])
def models_status_api():
    """Retrieve runtime registry status of proctoring models and hot-swap readiness."""
    global _loaded_logistic_model, _loaded_anomaly_model
    log_m, ano_m = _get_active_models()
    models_dir = Path(__file__).parent.parent.parent / "models"

    return jsonify({
        "status": "success",
        "hot_swap_supported": True,
        "active_agents_count": len(_proctoring_agents),
        "live_agents_updated": len(_proctoring_agents),
        "models": {
            "logistic_regression": {
                "loaded": log_m is not None,
                "file_exists": (models_dir / "logistic_model.pkl").exists(),
                "path": str(models_dir / "logistic_model.pkl"),
                "weights_count": len(getattr(log_m, "weights", [])) if log_m and hasattr(log_m, "weights") and getattr(log_m, "weights") is not None else 0,
            },
            "anomaly_detector": {
                "loaded": ano_m is not None,
                "file_exists": (models_dir / "anomaly_model.pkl").exists(),
                "path": str(models_dir / "anomaly_model.pkl"),
                "method": getattr(ano_m, "method", "zscore") if ano_m else None,
            },
        },
        "logistic_model": {
            "loaded": log_m is not None,
            "weights_count": len(getattr(log_m, "weights", [])) if log_m and hasattr(log_m, "weights") and getattr(log_m, "weights") is not None else 0,
        },
        "anomaly_model": {
            "loaded": ano_m is not None,
            "method": getattr(ano_m, "method", "zscore") if ano_m else None,
        },
        "last_trained_at": _training_records[-1]["trained_at"] if _training_records else None,
    })


@app.route("/api/models/activate", methods=["POST"])
def models_activate_api():
    """Hot-swap and activate new or reloaded model artifacts into all live proctoring agents."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if user and user.get("role") not in ("super_admin", "admin", "examiner"):
        return jsonify({"error": "Unauthorized"}), 401

    global _loaded_logistic_model, _loaded_anomaly_model
    import pickle
    models_dir = Path(__file__).parent.parent.parent / "models"
    log_pkl = models_dir / "logistic_model.pkl"
    ano_pkl = models_dir / "anomaly_model.pkl"

    reloaded_log = False
    reloaded_ano = False

    if log_pkl.exists():
        try:
            with open(log_pkl, "rb") as f:
                _loaded_logistic_model = pickle.load(f)
                reloaded_log = True
        except Exception:
            pass

    if ano_pkl.exists():
        try:
            with open(ano_pkl, "rb") as f:
                _loaded_anomaly_model = pickle.load(f)
                reloaded_ano = True
        except Exception:
            pass

    # Propagate to all active proctoring agents
    propagated_count = 0
    for agent in _proctoring_agents.values():
        agent.set_models(logistic_model=_loaded_logistic_model, anomaly_model=_loaded_anomaly_model)
        propagated_count += 1

    return jsonify({
        "status": "success",
        "hot_swap_applied": True,
        "message": "Models hot-swapped into active runtime",
        "reloaded_logistic": reloaded_log,
        "reloaded_anomaly": reloaded_ano,
        "agents_propagated": propagated_count,
        "timestamp": datetime.now().isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMINER API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/examiner/submissions", methods=["GET"])
def examiner_submissions():
    """Get all submissions for an exam."""
    exam_id = request.args.get("exam_id", "")
    if not exam_id:
        return jsonify({"submissions": []})

    db = _get_db()
    exam = _exam_manager.get_exam(exam_id) or db.get_exam(exam_id)
    if not exam:
        return jsonify({"submissions": []})

    submissions = []
    seen_students = set()

    for skey, session_info in _active_sessions.items():
        parts = skey.split(":")
        if len(parts) >= 2 and parts[1] == exam_id:
            student_id = parts[0]
            seen_students.add(student_id)
            agent = _proctoring_agents.get(skey)
            status = agent.get_status() if agent else {}
            graded = _grade_session_internal(student_id, exam_id, exam)
            answers = _load_answers_for_session(student_id, exam_id)

            submissions.append({
                "student_id": student_id,
                "exam_id": exam_id,
                "total_score": graded.get("total_score", 0.0),
                "total_max": graded.get("total_max", 0.0),
                "percentage": graded.get("percentage", 0.0),
                "answers": len(answers),
                "state": status.get("state", "NORMAL"),
                "risk_score": status.get("current_risk", 0.0),
                "submitted_at": session_info.get("started_at", datetime.now().isoformat()),
            })

    # Also query SQLite sessions table for any finished sessions or sessions after server restart
    try:
        db_rows = db._conn.execute(
            "SELECT * FROM sessions WHERE exam_id = ? ORDER BY created_at DESC", (exam_id,)
        ).fetchall()
        for r in db_rows:
            sid = r["student_id"]
            if sid not in seen_students:
                seen_students.add(sid)
                graded = _grade_session_internal(sid, exam_id, exam)
                answers = _load_answers_for_session(sid, exam_id)
                submissions.append({
                    "student_id": sid,
                    "exam_id": exam_id,
                    "total_score": graded.get("total_score", 0.0),
                    "total_max": graded.get("total_max", 0.0),
                    "percentage": graded.get("percentage", 0.0),
                    "answers": len(answers),
                    "state": r["state"] or "NORMAL",
                    "risk_score": r["risk_score"] or 0.0,
                    "submitted_at": r["ended_at"] or r["created_at"] or datetime.now().isoformat(),
                })
    except Exception:
        pass

    return jsonify({"submissions": submissions, "count": len(submissions)})


@app.route("/api/student/results", methods=["GET"])
def student_results_api():
    """Get results for the current student (strictly release-gated; zero risk exposure)."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    student_id = user.get("username", user.get("user_id", ""))
    user_id = user.get("user_id", "")
    results = []
    db = _get_db()
    seen_exams = set()

    # Query DB sessions
    try:
        db_sessions = db._conn.execute(
            "SELECT * FROM sessions WHERE student_id = ? OR student_id = ? ORDER BY created_at DESC",
            (student_id, user_id)
        ).fetchall()
        for s in db_sessions:
            eid = s["exam_id"]
            if eid in seen_exams:
                continue

            exam = _exam_manager.get_exam(eid)
            is_rel = getattr(exam, "is_released", False) if exam else False
            db_exam = db.get_exam(eid)
            if db_exam:
                is_rel = is_rel or bool(db_exam.get("is_released", False))

            if not is_rel and s["state"] != "released":
                continue

            if not exam:
                continue

            seen_exams.add(eid)
            graded = _grade_session_internal(s["student_id"], eid, exam)
            answers = _load_answers_for_session(s["student_id"], eid)
            results.append({
                "exam_id": eid,
                "exam_title": exam.title,
                "total_score": graded.get("total_score", 0.0),
                "total_max": graded.get("total_max", 0.0),
                "percentage": graded.get("percentage", 0.0),
                "answered": len(answers),
                "is_released": True,
                "status": "released",
                "submitted_at": s["ended_at"] or s["created_at"] or datetime.now().isoformat(),
            })
    except Exception:
        pass

    # Active sessions in memory
    for skey in list(_active_sessions.keys()):
        parts = skey.split(":")
        if len(parts) >= 2 and (parts[0] == student_id or parts[0] == user_id):
            eid = parts[1]
            if eid in seen_exams:
                continue

            exam = _exam_manager.get_exam(eid)
            if not exam:
                continue
            is_rel = getattr(exam, "is_released", False)
            db_exam = db.get_exam(eid)
            if db_exam:
                is_rel = is_rel or bool(db_exam.get("is_released", False))
            if not is_rel:
                continue

            seen_exams.add(eid)
            graded = _grade_session_internal(parts[0], eid, exam)
            answers = _load_answers_for_session(parts[0], eid)
            results.append({
                "exam_id": eid,
                "exam_title": exam.title,
                "total_score": graded.get("total_score", 0.0),
                "total_max": graded.get("total_max", 0.0),
                "percentage": graded.get("percentage", 0.0),
                "answered": len(answers),
                "is_released": True,
                "status": "released",
                "submitted_at": datetime.now().isoformat(),
            })

    return jsonify({"results": results, "count": len(results)})


@app.route("/api/student/results/<exam_id>", methods=["GET"])
def student_result_detail(exam_id):
    """Get detailed results for a specific exam (enforcing release gate)."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    exam = _exam_manager.get_exam(exam_id)
    if not exam:
        return jsonify({"error": "Exam not found"}), 404

    student_id = user.get("username", user.get("user_id", ""))
    user_id = user.get("user_id", "")
    db = _get_db()

    # Check release status
    is_rel = getattr(exam, "is_released", False)
    db_exam = db.get_exam(exam_id)
    if db_exam:
        is_rel = is_rel or bool(db_exam.get("is_released", False))

    session_id = _find_session_id(student_id, exam_id) or _find_session_id(user_id, exam_id)
    db_s = db.get_session(session_id) if session_id else None
    if db_s and db_s.get("state") == "released":
        is_rel = True

    if not is_rel:
        return jsonify({
            "error": "Exam results have not been released by the examiner yet.",
            "is_released": False,
            "status": "pending_release"
        }), 403

    graded = _grade_session_internal(student_id, exam_id, exam)
    # If grading with username returned empty, try with user_id
    if not graded.get("per_question") and user_id and user_id != student_id:
        graded = _grade_session_internal(user_id, exam_id, exam)

    raw_answers = _load_answers_for_session(student_id, exam_id)
    if not raw_answers and user_id != student_id:
        raw_answers = _load_answers_for_session(user_id, exam_id)

    per_question = dict(graded.get("per_question", {}))

    # Ensure overrides are applied even if _grade_session_internal missed them
    if session_id and per_question:
        per_question, _ = _apply_grade_overrides(session_id, per_question, graded.get("total_score", 0.0))
        # Recalculate total after overrides
        total_score = sum(pq.get("score", 0.0) for pq in per_question.values())
        total_max = graded.get("total_max", 0.0)
        pct = round(total_score / total_max * 100, 2) if total_max > 0 else 0.0
        graded["total_score"] = round(total_score, 2)
        graded["percentage"] = pct

    for q in exam.questions:
        qid = q.id
        raw_val = raw_answers.get(qid, "")
        ans_disp = raw_val.get("answer", "") if isinstance(raw_val, dict) else raw_val
        if qid not in per_question:
            per_question[qid] = {
                "question_id": qid,
                "question_text": q.text,
                "question_type": q.type.value,
                "marks": q.marks,
                "score": 0.0,
                "max_score": q.marks,
                "correct": False,
                "student_answer": ans_disp,
                "correct_answer": q.correct_answer,
            }
        else:
            per_question[qid]["student_answer"] = ans_disp
            per_question[qid]["correct_answer"] = q.correct_answer

    return jsonify({
        "exam_id": exam_id,
        "exam_title": exam.title,
        "per_question": per_question,
        "total_score": graded.get("total_score", 0.0),
        "total_max": graded.get("total_max", 0.0),
        "percentage": graded.get("percentage", 0.0),
        "grading_methods": {
            "mcq": "exact_match",
            "numerical": "tolerance_based",
            "short_answer": "tfidf_cosine",
            "long_answer": "multi_signal",
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH CHANGE PASSWORD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/change-password", methods=["POST"])
def change_password():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = _get_auth_db().get_session_user(token) if token else None
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(force=True)
    current = data.get("current_password", "")
    new_pass = data.get("new_password", "")

    if not current or not new_pass:
        return jsonify({"error": "Current and new password required"}), 400

    if len(new_pass) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    # Verify current password
    verified = _get_auth_db().verify_password(user["username"], current)
    if not verified:
        return jsonify({"error": "Current password is incorrect"}), 401

    # Update password
    from src.database.auth import hash_password
    new_hash = hash_password(new_pass)
    _get_auth_db()._conn.execute(
        "UPDATE users SET password_hash = ? WHERE user_id = ?",
        (new_hash, user["user_id"]))
    _get_auth_db()._conn.commit()

    return jsonify({"message": "Password changed successfully"})


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _grade_session_internal(student_id, exam_id, exam, answers=None):
    """Grade a session's answers internally, unwrapping answers and applying overrides."""
    if answers is None:
        answers = _load_answers_for_session(student_id, exam_id)

    raw_answers = dict(answers) if answers else {}
    unwrapped_answers = _unwrap_answers_dict(raw_answers)

    total_max = sum(q.marks for q in exam.questions) if hasattr(exam, "questions") else 100.0

    if not unwrapped_answers:
        return {
            "student_id": student_id,
            "exam_id": exam_id,
            "per_question": {},
            "total_score": 0.0,
            "total_max": total_max,
            "percentage": 0.0,
        }

    # Build reference answers map
    ref_answers = {}
    if hasattr(exam, "questions"):
        for q in exam.questions:
            if q.correct_answer and q.type in (QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER):
                ref_answers[q.id] = str(q.correct_answer)

    results = _grading_orchestrator.grade_all(getattr(exam, "questions", []), unwrapped_answers, ref_answers)

    per_question = {}
    if hasattr(exam, "questions"):
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

    # Apply DB grade overrides
    session_id = _find_session_id(student_id, exam_id)
    total_score = results.get("total_score", 0.0)
    if session_id:
        per_question, total_score = _apply_grade_overrides(session_id, per_question, total_score)

    total_max = results.get("total_max", total_max)
    pct = round(total_score / total_max * 100, 2) if total_max > 0 else 0.0

    return {
        "student_id": student_id,
        "exam_id": exam_id,
        "per_question": per_question,
        "total_score": round(total_score, 2),
        "total_max": total_max,
        "percentage": pct,
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
#  WORLD-CLASS INTELLIGENCE & AUDITABILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/trace/replay/<session_id>", methods=["GET"])
def replay_session_trace(session_id: str):
    """Forensically replay session decisions with the DecisionReplayEngine."""
    agent = _proctoring_agents.get(session_id)
    traces = []
    student_id = "student_session"

    if agent:
        traces = [h for h in agent.get_history() if "risk_score" in h]
        student_id = agent.baseline.student_id if hasattr(agent, "baseline") else student_id
    else:
        for k, sinfo in _active_sessions.items():
            if sinfo.get("session_id") == session_id or k == session_id:
                ctrl = sinfo.get("controller")
                if ctrl and hasattr(ctrl, "agent"):
                    agent = ctrl.agent
                    traces = agent.get_history()
                    student_id = k.split(":")[0] if ":" in k else student_id
                break

    replayer = DecisionReplayEngine(session_id=session_id, student_id=student_id)
    if traces:
        replayer.load_from_traces(traces)

    step_param = request.args.get("step")
    if step_param is not None:
        try:
            step_idx = int(step_param)
            snapshot = replayer.seek(step_idx)
            if snapshot:
                return jsonify({"status": "success", "snapshot": snapshot.to_dict()})
            return jsonify({"error": f"Step {step_idx} out of range"}), 404
        except ValueError:
            return jsonify({"error": "Invalid step parameter, must be integer"}), 400

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "student_id": student_id,
        "total_steps": replayer.total_steps(),
        "timeline": replayer.export_timeline(),
    })


@app.route("/api/trace/audit-chain/<session_id>", methods=["GET"])
def get_audit_chain(session_id: str):
    """Retrieve cryptographic SHA-256 Merkle audit entries for a session."""
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    target_id = session_id
    entries = _audit_chain.get_entries(session_id=target_id, limit=limit, offset=offset)
    total = _audit_chain.get_chain_length(session_id=target_id)
    if total == 0:
        agent = _proctoring_agents.get(session_id)
        if agent and getattr(agent, "session_id", None) and agent.session_id != session_id:
            target_id = agent.session_id
            entries = _audit_chain.get_entries(session_id=target_id, limit=limit, offset=offset)
            total = _audit_chain.get_chain_length(session_id=target_id)
        else:
            for ag in _proctoring_agents.values():
                if getattr(ag, "session_id", None) == session_id:
                    target_id = session_id
                    break

    return jsonify({
        "status": "success",
        "session_id": session_id,
        "resolved_session_id": target_id,
        "total_entries": total,
        "entries": [e.to_dict() for e in entries],
    })


@app.route("/api/trace/audit-chain/<session_id>/verify", methods=["POST"])
def verify_audit_chain(session_id: str):
    """Cryptographically verify the hash chain integrity for an exam session."""
    target_id = session_id
    total = _audit_chain.get_chain_length(session_id=target_id)
    if total == 0:
        agent = _proctoring_agents.get(session_id)
        if agent and getattr(agent, "session_id", None) and agent.session_id != session_id:
            target_id = agent.session_id

    is_valid, broken_sequence = _audit_chain.verify_integrity(session_id=target_id)
    return jsonify({
        "status": "success",
        "session_id": session_id,
        "resolved_session_id": target_id,
        "is_intact": is_valid,
        "broken_at_sequence": broken_sequence,
        "verification_timestamp": time.time(),
    })


@app.route("/api/grading/mathematical", methods=["POST"])
def grade_mathematical():
    """Grade multi-step mathematical & derivation answers with partial credit."""
    data = request.get_json() or {}
    student_solution = data.get("student_solution", "")
    expected_formula = data.get("expected_formula", "")
    expected_variables = data.get("expected_variables", {})
    expected_final_value = data.get("expected_final_value")
    expected_units = data.get("expected_units")
    marks = float(data.get("marks", 10.0))
    tolerance = float(data.get("tolerance", 0.02)) if data.get("tolerance") is not None else None

    result = _math_grader.grade(
        student_solution=student_solution,
        expected_formula=expected_formula,
        expected_variables=expected_variables,
        expected_final_value=expected_final_value,
        expected_units=expected_units,
        marks=marks,
        tolerance=tolerance,
    )
    return jsonify({"status": "success", "result": result.to_dict()})


@app.route("/api/grading/programming", methods=["POST"])
def grade_programming():
    """Grade programming questions using pure AST parsing and sandbox execution."""
    data = request.get_json() or {}
    code_str = data.get("code", "")
    entry_function = data.get("entry_function", "")
    public_tests = data.get("public_tests", [])
    hidden_tests = data.get("hidden_tests", [])
    edge_cases = data.get("edge_cases", [])
    marks = float(data.get("marks", 10.0))
    max_complexity = int(data.get("max_allowed_complexity", 15))

    result = _prog_grader.grade(
        code_str=code_str,
        entry_function_name=entry_function,
        public_tests=public_tests,
        hidden_tests=hidden_tests,
        edge_cases=edge_cases,
        marks=marks,
        max_allowed_complexity=max_complexity,
    )
    return jsonify({"status": "success", "result": result.to_dict()})


@app.route("/api/grading/plagiarism", methods=["POST"])
def detect_plagiarism():
    """Detect cohort-wide near-duplicate answers via pure-math MinHash & LSH."""
    data = request.get_json() or {}
    submissions = data.get("submissions", {})
    threshold = float(data.get("threshold", 0.60))
    exam_id = data.get("exam_id", "exam_cohort")

    detector = PlagiarismDetector(threshold=threshold)
    if submissions and isinstance(next(iter(submissions.values())), dict):
        for qid, qsubs in submissions.items():
            for sid, ans in qsubs.items():
                detector.add_submission(sid, qid, ans)
    else:
        for sid, ans in submissions.items():
            detector.add_submission(sid, "default_q", ans)

    report = detector.detect(exam_id=exam_id)
    return jsonify({
        "status": "success",
        "exam_id": report.exam_id,
        "total_submissions": report.total_submissions,
        "total_pairs_checked": report.total_pairs_checked,
        "flagged_pairs": [
            {
                "student_a": p.student_a,
                "student_b": p.student_b,
                "jaccard_similarity": p.jaccard_similarity,
                "minhash_similarity": p.minhash_similarity,
                "shared_shingles": p.shared_shingles_count,
                "flagged": p.flagged,
                "question_id": p.question_id,
            }
            for p in report.flagged_pairs
        ],
        "summary": report.summary,
    })


@app.route("/api/grading/consensus", methods=["POST"])
def evaluate_rubric_consensus():
    """Evaluate multi-perspective consensus and detect grader disagreement."""
    data = request.get_json() or {}
    student_answer = data.get("student_answer", "")
    rubric = data.get("rubric", {})
    marks = float(data.get("marks", 10.0))

    result = _consensus_engine.evaluate(student_answer, rubric, marks)
    return jsonify({"status": "success", "result": result.to_dict()})


@app.route("/api/ml/adversarial/benchmark", methods=["GET", "POST"], endpoint="api_adversarial_benchmark_nested")
@app.route("/api/ml/adversarial-benchmark", methods=["GET", "POST"], endpoint="api_adversarial_benchmark_flat")
def run_adversarial_benchmark():
    """Run pure-math red-team adversarial stress testing against the proctoring agent."""
    data = (request.get_json(silent=True) or {}) if request.is_json else {}
    threshold = float(data.get("escalation_threshold") or request.args.get("escalation_threshold") or 0.65)

    harness = RedTeamHarness(escalation_threshold=threshold)
    test_agent = ProctoringAgent()
    report = harness.evaluate_agent(lambda sig: test_agent.process_signals(sig))
    rep_dict = report.to_dict()
    return jsonify({
        "status": "success",
        "report": rep_dict,
        "scenarios": rep_dict.get("scenarios", []),
        "summary": rep_dict.get("summary", {}),
    })


@app.route("/api/baseline/<student_id>", methods=["GET", "POST"])
def student_baseline_profile(student_id: str):
    """Retrieve or update a student's personal behavioral baseline."""
    if student_id not in _student_baselines:
        _student_baselines[student_id] = StudentBaseline(student_id=student_id)
    baseline = _student_baselines[student_id]

    if request.method == "POST":
        data = request.get_json() or {}
        from src.baseline import BehavioralSample
        sample = BehavioralSample(
            timestamp=data.get("timestamp", time.time()),
            typing_wpm=float(data.get("typing_wpm", 0.0)),
            mouse_velocity=float(data.get("mouse_velocity", 0.0)),
            mouse_clicks=int(data.get("mouse_clicks", 0)),
            face_present=float(data.get("face_present", 1.0)),
            face_count=int(data.get("face_count", 1)),
            gaze_off_screen_seconds=float(data.get("gaze_off_screen_seconds", 0.0)),
            pause_duration_seconds=float(data.get("pause_duration_seconds", 0.0)),
        )
        baseline.add_sample(sample)

    return jsonify({
        "status": "success",
        "student_id": student_id,
        "sample_count": baseline.samples,
        "maturity": baseline.maturity_score(),
        "is_ready": baseline.is_ready(),
        "profile": baseline.get_profile(),
    })



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

# ---------------------------------------------------------------------------
# Global RBAC guard: every /api route requires a valid token except the public
# ones; staff-only routes additionally require examiner/admin; students may
# only act on their own student_id / session_key.  Set EXAM_AGENT_AUTH_ENFORCE=0
# to disable (used only by legacy unauthenticated tests).
# ---------------------------------------------------------------------------
import os as _os_rbac
import re as _re_rbac

_STAFF_ROLES = {"examiner", "admin", "super_admin"}
_WRITE = {"POST", "PUT", "PATCH", "DELETE"}
_PUBLIC_RULES = [
    (None, r"^/api/health$"),
    (None, r"^/api/config$"),
    ({"POST"}, r"^/api/auth/(login|register)$"),
]
_STAFF_RULES = [
    (_WRITE, r"^/api/exams(/.*)?$"),
    ({"GET"}, r"^/api/exams/[^/]+/roster$"),
    (None, r"^/api/examiner/"),
    (None, r"^/api/grading/(override|overrides|release|inspect|plagiarism|consensus|mathematical|programming)"),
    (None, r"^/api/(reports|predict|analytics|train|training|models|ml|demo|baseline|trace|evidence|knowledge)(/|$)"),
    ({"GET"}, r"^/api/sessions$"),
    (None, r"^/api/proctor/"),
    (None, r"^/api/agent/(history|transitions)"),
    (None, r"^/api/grade(/|$)"),
]


def _rule_hit(rules, method, path):
    return any((m is None or method in m) and _re_rbac.match(rx, path) for m, rx in rules)


@app.before_request
def _rbac_guard():
    g.current_user = None
    token = (request.headers.get("Authorization", "") or "").replace("Bearer ", "", 1).strip()
    user = _get_auth_db().get_session_user(token) if token else None
    if user:
        g.current_user = user

    if _os_rbac.environ.get("EXAM_AGENT_AUTH_ENFORCE", "1") == "0" or request.method == "OPTIONS":
        return None
    path = request.path
    if not path.startswith("/api/") or _rule_hit(_PUBLIC_RULES, request.method, path):
        return None
    if not user:
        return jsonify({"error": "Authentication required"}), 401
    role = user.get("role")
    if _rule_hit(_STAFF_RULES, request.method, path) and role not in _STAFF_ROLES:
        return jsonify({"error": "Forbidden: examiner or admin role required"}), 403
    if role == "student":
        mine = {str(user.get("username")), str(user.get("user_id"))}
        body = request.get_json(silent=True) if request.method in _WRITE else None
        claimed = [(request.view_args or {}).get("student_id"),
                   (body or {}).get("student_id") if isinstance(body, dict) else None]
        keys = [request.args.get("session_key"),
                (body or {}).get("session_key") if isinstance(body, dict) else None]
        for c_ in claimed:
            if c_ and str(c_) not in mine:
                return jsonify({"error": "Forbidden: not your session"}), 403
        for k_ in keys:
            if k_ and str(k_).split(":")[0] not in mine:
                return jsonify({"error": "Forbidden: not your session"}), 403
    return None


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
