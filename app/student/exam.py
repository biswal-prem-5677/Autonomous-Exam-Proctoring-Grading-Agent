"""Streamlit student exam UI — take an exam with live proctoring indicators."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.models import Question, QuestionType, ExamSession, AgentState
from src.exam.session import ExamSessionManager
from src.exam.exam import ExamController
from src.grading.scorer import GradingOrchestrator
from src.agent.controller import ProctoringAgent, AgentState as AGState
from src.agent.memory import EvidenceMemory
from src.agent.risk import RiskEngine
from src.reports.student_report import StudentReportGenerator
from src.reports.evidence import EvidenceLog
from data.sample_data import (
    load_sample_questions, get_all_student_ids,
    load_sample_events, save_questions_to_file,
)
import time
import uuid

st.set_page_config(page_title="Exam", page_icon="📝", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em;
        background: linear-gradient(135deg, #7C3AED, #A78BFA);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .risk-badge {
        padding: 8px 18px; border-radius: 24px; font-weight: 600;
        font-size: 0.82rem; letter-spacing: 0.02em;
    }
    .risk-normal { background: rgba(16, 185, 129, 0.12); color: #059669; border: 1px solid rgba(16,185,129,0.25); }
    .risk-suspicious { background: rgba(245, 158, 11, 0.12); color: #D97706; border: 1px solid rgba(245,158,11,0.25); }
    .risk-high { background: rgba(239, 68, 68, 0.12); color: #DC2626; border: 1px solid rgba(239,68,68,0.25); }

    .question-card {
        background: #1E1E2E; border: 1px solid #313244; border-radius: 14px;
        padding: 24px; margin: 12px 0;
    }
    .timer {
        font-size: 1.6rem; font-weight: 700; font-family: 'SF Mono', 'Fira Code', monospace;
        color: #CDD6F4;
    }
    .section-header {
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #A6ADC8;
    }
    .stat-card {
        background: #181825; border: 1px solid #313244; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #CDD6F4; }
    .stat-label { font-size: 0.72rem; color: #6C7086; text-transform: uppercase; letter-spacing: 0.05em; }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; transition: all 0.2s !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #8B5CF6, #7C3AED) !important;
        transform: translateY(-1px) !important;
    }

    div.stButton > button {
        background: #313244 !important; color: #CDD6F4 !important;
        border: 1px solid #45475A !important; border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    div.stButton > button:hover { background: #45475A !important; border-color: #585B70 !important; }

    [data-testid="stMetricValue"] { color: #CDD6F4 !important; font-weight: 700 !important; }

    .stRadio > div { gap: 8px; }
    [data-baseweb="radio"] > div:first-child > div {
        background-color: #313244 !important; border-color: #45475A !important;
    }
    [data-baseweb="radio"] input:checked + div > div:first-child {
        background-color: #7C3AED !important; border-color: #7C3AED !important;
    }
    input[type="number"], textarea, select, input[type="text"] {
        background: #1E1E2E !important; color: #CDD6F4 !important;
        border: 1px solid #313244 !important; border-radius: 8px !important;
    }

    .divider {
        height: 1px; background: linear-gradient(90deg, transparent, #313244, transparent);
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "student_id": None,
        "student_name": None,
        "exam_started": False,
        "exam_ended": False,
        "current_q_index": 0,
        "answers": {},
        "session_obj": None,
        "controller": None,
        "grading_results": None,
        "report": None,
        "events_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def risk_badge(state: str) -> str:
    cls = {"NORMAL": "risk-normal", "SUSPICIOUS": "risk-suspicious", "HIGH_RISK": "risk-high"}
    return f'<span class="risk-badge {cls.get(state, "risk-normal")}">{state}</span>'


def login_screen():
    st.title("📝 Student Exam Portal")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Select Your Identity")
        students = get_all_student_ids()
        names = {sid: sid.replace("_", " ").title() for sid in students}

        selected = st.selectbox("Student ID", students, format_func=lambda s: names.get(s, s))
        duration = st.slider("Exam Duration (minutes)", 10, 120, 30)

        if st.button("Start Exam", type="primary", use_container_width=True):
            # Save questions to file so QuestionBank can load them
            save_questions_to_file("data/questions.json")
            st.session_state.student_id = selected
            st.session_state.student_name = names.get(selected, selected)
            st.session_state.exam_duration = duration
            st.session_state.exam_started = True
            st.rerun()


def exam_screen():
    student_id = st.session_state.student_id
    student_name = st.session_state.student_name
    duration = st.session_state.get("exam_duration", 30)

    # Load questions once
    if "questions" not in st.session_state:
        qb_data = load_sample_questions()
        st.session_state.questions = [
            Question(
                id=q["id"],
                type=QuestionType(q["type"]),
                text=q["text"],
                options=q.get("options"),
                correct_answer=q.get("correct_answer"),
                marks=q.get("marks", 1.0),
                keywords=q.get("keywords", []),
                tolerance=q.get("tolerance", 0.02),
                metadata=q.get("metadata", {}),
            )
            for q in qb_data
        ]

    questions = st.session_state.questions
    total_qs = len(questions)

    # Initialize session on first render
    if st.session_state.session_obj is None:
        controller = ExamController(config={"risk": {"escalation_threshold": 0.7, "review_threshold": 0.5}})
        session = controller.create_session(student_id, "demo_exam_001", duration_minutes=duration)
        st.session_state.session_obj = session
        st.session_state.controller = controller
        controller.start_exam()

    session = st.session_state.session_obj
    controller = st.session_state.controller

    # ── Header bar ──────────────────────────────────────────────────────────
    top_col1, top_col2, top_col3 = st.columns([3, 2, 2])
    with top_col1:
        st.markdown(f"### 👤 {student_name}")
    with top_col2:
        elapsed = session.elapsed_seconds
        remaining = max(0, duration * 60 - elapsed)
        st.markdown(f'<div class="timer" style="text-align:center;">⏱️ {format_time(remaining)}</div>',
                    unsafe_allow_html=True)
    with top_col3:
        # Simulate proctoring signals
        event_types = ["NORMAL", "SUSPICIOUS", "HIGH_RISK"]
        sim_state = st.selectbox("Simulated Proctoring State", event_types, index=0,
                                 label_visibility="collapsed")
        st.markdown(f'<div style="text-align:center;">{risk_badge(sim_state)}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Auto-submit when time runs out ─────────────────────────────────────
    if remaining <= 0 and not st.session_state.exam_ended:
        st.session_state.exam_ended = True
        st.rerun()

    if st.session_state.exam_ended:
        show_results(questions, controller)
        return

    # ── Question navigation ─────────────────────────────────────────────────
    q_index = st.session_state.current_q_index
    q = questions[q_index]

    # Nav bar
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
    with nav_col1:
        if st.button("⬅️ Previous", disabled=(q_index == 0)):
            st.session_state.current_q_index = max(0, q_index - 1)
            st.rerun()
    with nav_col2:
        st.markdown(f"<div style='text-align:center; padding-top:8px;'><b>{q_index + 1} / {total_qs}</b></div>",
                    unsafe_allow_html=True)
    with nav_col3:
        answered = sum(1 for i in range(total_qs) if str(questions[i].id) in st.session_state.answers)
        st.markdown(f"<div style='text-align:center; padding-top:8px;'>✅ {answered}/{total_qs}</div>",
                    unsafe_allow_html=True)
    with nav_col4:
        if st.button("Next ➡️", disabled=(q_index == total_qs - 1)):
            st.session_state.current_q_index = min(total_qs - 1, q_index + 1)
            st.rerun()
    with nav_col5:
        if st.button("🏁 Submit", type="primary"):
            _submit_exam(questions, controller)

    st.markdown("---")

    # ── Question display ────────────────────────────────────────────────────
    st.markdown(f"#### Q{q_index + 1}. {q.text}")
    st.caption(f"*Type: {q.type.value.upper()} | Marks: {q.marks}*")

    q_id = str(q.id)
    saved_answer = st.session_state.answers.get(q_id, "")

    answer_key = f"ans_{q_id}"

    if q.type == QuestionType.MCQ:
        selected = st.radio("Select your answer:", q.options,
                            index=q.options.index(saved_answer) if saved_answer in q.options else 0,
                            key=answer_key, label_visibility="collapsed")
        st.session_state.answers[q_id] = selected

    elif q.type == QuestionType.TRUE_FALSE:
        selected = st.radio("Select:", q.options,
                            index=q.options.index(saved_answer) if saved_answer in q.options else 0,
                            key=answer_key, label_visibility="collapsed")
        st.session_state.answers[q_id] = selected

    elif q.type == QuestionType.NUMERICAL:
        val = st.number_input("Your answer:", value=float(saved_answer) if saved_answer else 0.0,
                              step=0.1, key=answer_key)
        st.session_state.answers[q_id] = val

    elif q.type == QuestionType.SHORT_ANSWER:
        val = st.text_area("Your answer:", value=saved_answer, height=150, key=answer_key,
                           max_chars=500)
        st.session_state.answers[q_id] = val

    # ── Question palette ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Question Palette**")
    palette_cols = st.columns(min(total_qs, 10))
    for i, question in enumerate(questions):
        col_idx = i % 10
        q_answered = str(question.id) in st.session_state.answers
        label = f"{'✅' if q_answered else '⬜'} {i + 1}"
        if st.button(label, key=f"pal_{question.id}", use_container_width=True):
            st.session_state.current_q_index = i
            st.rerun()


def _submit_exam(questions, controller):
    st.session_state.exam_ended = True
    session = st.session_state.session_obj

    # End session
    controller.end_exam()

    # Grade all answers
    orchestrator = GradingOrchestrator()
    answers = st.session_state.answers
    reference = {q.id: q.correct_answer for q in questions}

    grading = orchestrator.grade_all(questions, answers, reference)
    st.session_state.grading_results = grading

    # Simulate proctoring events
    events_data = load_sample_events(st.session_state.student_id)
    for evt in events_data.get("events", []):
        controller.record_proctoring_event(
            event_type=evt["type"],
            confidence=evt["confidence"],
            details={"timestamp": evt.get("timestamp", "")},
        )

    # Generate student report
    proctor_summary = controller.get_session_status()
    proctor_summary["student_id"] = st.session_state.student_id
    report_gen = StudentReportGenerator()
    report = report_gen.generate(
        grading_results=grading,
        proctoring_summary=proctor_summary,
    )
    st.session_state.report = report
    st.rerun()


def show_results(questions, controller):
    grading = st.session_state.get("grading_results")
    report = st.session_state.get("report")

    if grading is None:
        st.info("Grading your exam...")
        return

    st.title("📊 Exam Results")

    # Score summary
    col1, col2, col3 = st.columns(3)
    with col1:
        pct = grading.get("percentage", 0)
        st.metric("Score", f"{grading.get('total_score', 0):.1f} / {grading.get('total_max', 0):.1f}",
                  f"{pct:.1f}%")
    with col2:
        st.metric("Correct", f"{report.get('grade_summary', {}).get('correct_count', 0)} / {len(questions)}")
    with col3:
        status = controller.get_session_status()
        st.metric("Risk Score", f"{status.get('risk_score', 0):.4f}")

    # Risk state
    state = status.get("state", "NORMAL")
    st.markdown(f"**Proctoring State:** {risk_badge(state)}", unsafe_allow_html=True)

    st.markdown("---")

    # Per-question breakdown
    st.subheader("Question Breakdown")
    for q in questions:
        q_id = str(q.id)
        result = grading.get("per_question", {}).get(q_id, {})
        correct = result.get("correct", False)
        icon = "✅" if correct else "❌"
        score = result.get("score", 0)
        max_score = result.get("max_score", q.marks)

        with st.expander(f"{icon} Q{q.text[:50]}... — {score}/{max_score}", expanded=False):
            cols = st.columns(2)
            with cols[0]:
                st.write(f"**Type:** {q.type.value}")
                st.write(f"**Correct Answer:** {q.correct_answer}")
            with cols[1]:
                if result.get("details"):
                    st.write(f"**Method:** {result['details'].get('method', 'N/A')}")
                    if result["details"].get("matched_keywords"):
                        st.write(f"**Matched Keywords:** {', '.join(result['details']['matched_keywords'])}")

    # Recommendations
    st.markdown("---")
    st.subheader("Recommendations")
    for i, rec in enumerate(report.get("recommendations", []), 1):
        st.info(f"**{i}.** {rec}")

    st.markdown("---")
    if st.button("🔄 Start New Exam", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ["student_id", "student_name", "exam_duration"]:
                del st.session_state[key]
        st.session_state.exam_started = False
        st.session_state.exam_ended = False
        st.session_state.session_obj = None
        st.rerun()


def main():
    init_session_state()

    if not st.session_state.exam_started:
        login_screen()
    else:
        exam_screen()


if __name__ == "__main__":
    main()
