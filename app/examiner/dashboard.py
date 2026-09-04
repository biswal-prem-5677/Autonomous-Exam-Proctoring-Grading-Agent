"""Streamlit examiner dashboard — overview of all student exams and risk analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np
from src.reports.examiner_report import ExaminerReportGenerator
from src.reports.student_report import StudentReportGenerator
from src.reports.evidence import EvidenceLog
from data.sample_data import SAMPLE_STUDENT_ANSWERS, SAMPLE_PROCTORING_EVENTS, SAMPLE_QUESTIONS
from src.models.models import Question, QuestionType
from src.grading.scorer import GradingOrchestrator

st.set_page_config(page_title="Examiner Dashboard", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #CDD6F4;
        margin-bottom: 16px; letter-spacing: -0.01em;
    }
    .stat-card {
        background: #181825; border: 1px solid #313244; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .stat-value { font-size: 1.5rem; font-weight: 700; }
    .stat-label { font-size: 0.7rem; color: #6C7086; text-transform: uppercase; letter-spacing: 0.05em; }

    .flagged {
        background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 8px; padding: 12px; margin: 6px 0;
    }
    .clean {
        background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 8px; padding: 12px; margin: 6px 0;
    }

    [data-testid="stMetricValue"] {
        color: #CDD6F4 !important; font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6C7086 !important;
    }

    .stExpander > div:first-child {
        background: #1E1E2E; border: 1px solid #313244; border-radius: 10px;
    }
    .student-card {
        background: #181825; border: 1px solid #313244; border-radius: 10px;
        padding: 16px; margin: 8px 0;
    }

    .risk-bar {
        height: 6px; border-radius: 3px; background: #313244;
    }
    .risk-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }

    .section-divider {
        height: 1px; background: linear-gradient(90deg, transparent, #313244, transparent);
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_questions():
    return [
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
        for q in SAMPLE_QUESTIONS
    ]


def simulate_risk_score(events: list) -> float:
    """Simulate a risk score from event list."""
    if not events:
        return 0.0
    weights = {
        "face_absent": 0.30, "multiple_faces": 0.50, "head_turned": 0.20,
        "tab_switch": 0.25, "loud_audio": 0.10, "paste_detected": 0.40,
        "copy_paste": 0.40, "gaze_away": 0.15, "idle_keyboard": 0.05,
    }
    score = sum(weights.get(e.get("type", ""), 0.05) * e.get("confidence", 0.5)
                for e in events)
    return min(score, 1.0)


def run_grading_for_student(student_id: str, questions: list) -> dict:
    """Run grading pipeline for a single student."""
    answers_data = SAMPLE_STUDENT_ANSWERS.get(student_id, {})
    answers = answers_data.get("answers", {})
    reference = {q.id: q.correct_answer for q in questions}

    orchestrator = GradingOrchestrator()
    return orchestrator.grade_all(questions, answers, reference)


def build_student_reports(questions: list) -> list:
    """Build per-student reports for the examiner dashboard."""
    reports = []
    for sid, data in SAMPLE_STUDENT_ANSWERS.items():
        grading = run_grading_for_student(sid, questions)
        events = SAMPLE_PROCTORING_EVENTS.get(sid, {}).get("events", [])
        risk_score = simulate_risk_score(events)
        risk_level = "HIGH_RISK" if risk_score >= 0.7 else ("SUSPICIOUS" if risk_score >= 0.5 else "NORMAL")

        reports.append({
            "student_id": sid,
            "name": data.get("name", sid),
            "grade_summary": {
                "percentage": grading.get("percentage", 0.0),
                "total_score": grading.get("total_score", 0.0),
                "total_max": grading.get("total_max", 0.0),
                "correct_count": sum(
                    1 for r in grading.get("per_question", {}).values()
                    if r.get("correct", False)
                ),
            },
            "proctoring_summary": {
                "risk_score": round(risk_score, 4),
                "state": risk_level,
                "events_count": len(events),
                "event_types": [e["type"] for e in events],
            },
        })
    return reports


def main():
    st.title("🎓 Examiner Dashboard")
    st.markdown("---")

    questions = load_questions()
    student_reports = build_student_reports(questions)

    # ── Examiner report ─────────────────────────────────────────────────────
    examiner = ExaminerReportGenerator()
    report = examiner.generate("demo_exam_001", student_reports)
    stats = report["statistics"]
    risk = report["risk_analysis"]

    # ── Top metrics ─────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Students", report["total_students"])
    with m2:
        st.metric("Mean Score", f"{stats.get('mean_score', 0):.1f}%")
    with m3:
        st.metric("Pass Rate", f"{stats.get('pass_count', 0)}/{report['total_students']}")
    with m4:
        st.metric("High Risk", risk.get("high_risk_count", 0))
    with m5:
        st.metric("Mean Risk", f"{risk.get('mean_risk', 0):.4f}")

    st.markdown("---")

    # ── Score distribution ──────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Score Distribution")
        scores = [r["grade_summary"]["percentage"] for r in student_reports]
        names = [r.get("name", r["student_id"]) for r in student_reports]
        colors = ["#d4edda" if s >= 40 else "#f8d7da" for s in scores]
        st.bar_chart({name: score for name, score in zip(names, scores)})

        st.subheader("Grade Statistics")
        st.write(f"**Mean:** {stats.get('mean_score', 0):.1f}%")
        st.write(f"**Median:** {stats.get('median_score', 0):.1f}%")
        st.write(f"**Std Dev:** {stats.get('std_score', 0):.1f}%")
        st.write(f"**Range:** {stats.get('min_score', 0):.1f}% – {stats.get('max_score', 0):.1f}%")
        st.write(f"**Passed:** {stats.get('pass_count', 0)} | **Failed:** {stats.get('fail_count', 0)}")

    with col_right:
        st.subheader("🚨 Risk Analysis")
        st.write(f"**Mean Risk Score:** {risk.get('mean_risk', 0):.4f}")
        st.write(f"**Max Risk:** {risk.get('max_risk', 0):.4f}")
        st.write(f"**High Risk:** {risk.get('high_risk_count', 0)} students")
        st.write(f"**Suspicious:** {risk.get('suspicious_count', 0)} students")
        st.write(f"**Normal:** {risk.get('normal_count', 0)} students")

        st.subheader("Risk Distribution")
        risk_chart_data = {
            "Normal": risk.get("normal_count", 0),
            "Suspicious": risk.get("suspicious_count", 0),
            "High Risk": risk.get("high_risk_count", 0),
        }
        st.bar_chart(risk_chart_data)

    st.markdown("---")

    # ── Student table ───────────────────────────────────────────────────────
    st.subheader("📋 Student Details")

    for sr in student_reports:
        risk_val = sr["proctoring_summary"]["risk_score"]
        risk_level = sr["proctoring_summary"]["state"]
        pct = sr["grade_summary"]["percentage"]
        score_color = "🟢" if pct >= 70 else ("🟡" if pct >= 40 else "🔴")
        risk_color = "🔴" if risk_level == "HIGH_RISK" else ("🟡" if risk_level == "SUSPICIOUS" else "🟢")

        with st.expander(
            f"{score_color} {sr['name']} — {pct:.1f}% | {risk_color} Risk: {risk_val:.4f} ({risk_level})"
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**Grade**")
                st.write(f"Score: {sr['grade_summary']['total_score']:.1f} / {sr['grade_summary']['total_max']:.1f}")
                st.write(f"Percentage: {pct:.1f}%")
                st.write(f"Correct: {sr['grade_summary']['correct_count']} / {len(questions)}")
            with c2:
                st.write("**Proctoring**")
                st.write(f"State: {risk_level}")
                st.write(f"Risk Score: {risk_val:.4f}")
                st.write(f"Events: {sr['proctoring_summary']['events_count']}")
                if sr["proctoring_summary"].get("event_types"):
                    st.write(f"Event types: {', '.join(set(sr['proctoring_summary']['event_types']))}")
            with c3:
                st.write("**Status**")
                if risk_level == "HIGH_RISK":
                    st.error("⚠️ Requires immediate review")
                elif risk_level == "SUSPICIOUS":
                    st.warning("⚠️ Flagged for review")
                else:
                    st.success("✅ No concerns")

    st.markdown("---")

    # ── Flagged students ────────────────────────────────────────────────────
    flagged = report.get("flagged_students", [])
    st.subheader(f"🚩 Flagged for Review ({len(flagged)})")
    if flagged:
        for f in flagged:
            st.markdown(
                f'<div class="flagged"><b>{f["student_id"]}</b> — '
                f'risk={f["risk_score"]:.4f} | {f["reason"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No students flagged for review.")

    # ── Full text report ────────────────────────────────────────────────────
    with st.expander("📄 Full Text Report"):
        st.text(examiner.to_text())

    st.markdown("---")
    st.caption("⚠️ Risk scores are evidence indicators, NOT proof of cheating. "
               "Review flagged cases before taking any action.")


if __name__ == "__main__":
    main()
