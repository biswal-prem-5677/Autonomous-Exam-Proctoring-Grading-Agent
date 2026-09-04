"""Main entry point — CLI and programmatic API for the exam proctoring agent."""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_demo(args):
    """Run a full demo: load sample data, grade answers, generate reports."""
    from src.models.models import Question, QuestionType
    from src.grading.scorer import GradingOrchestrator
    from src.agent.controller import ProctoringAgent, AgentPhase
    from src.reports.student_report import StudentReportGenerator
    from src.reports.examiner_report import ExaminerReportGenerator
    from src.reports.evidence import EvidenceLog
    from data.sample_data import (
        SAMPLE_QUESTIONS, SAMPLE_STUDENT_ANSWERS,
        SAMPLE_PROCTORING_EVENTS, load_sample_questions,
    )

    print("=" * 60)
    print("  AUTONOMOUS EXAM PROCTORING & GRADING AGENT")
    print("  Demo Mode")
    print("=" * 60)

    # Build questions
    questions = [
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

    # Build reference answers
    reference = {q.id: q.correct_answer for q in questions}

    # Initialize grader
    orchestrator = GradingOrchestrator()

    # Initialize agent
    agent = ProctoringAgent(config={
        "risk": {
            "escalation_threshold": 0.7,
            "review_threshold": 0.5,
            "multi_signal_required": 2,
        },
        "temporal_window": 30,
    })

    student_reports = []

    for student_id, data in SAMPLE_STUDENT_ANSWERS.items():
        print(f"\n--- Processing: {data.get('name', student_id)} ---")

        answers = data.get("answers", {})

        # Grade
        results = orchestrator.grade_all(questions, answers, reference)
        pct = results.get("percentage", 0.0)
        print(f"  Score: {results['total_score']:.1f} / {results['total_max']:.1f} ({pct:.1f}%)")

        # Simulate proctoring signals
        events_data = SAMPLE_PROCTORING_EVENTS.get(student_id, {}).get("events", [])
        signals = _events_to_signals(events_data)
        agent_result = agent.process_signals(signals)
        risk_score = agent_result["risk_score"]
        state = agent_result["state"]
        print(f"  Risk: {risk_score:.4f} ({state})")
        print(f"  Events: {len(agent_result['events_detected'])}")

        # Generate student report
        report_gen = StudentReportGenerator()
        report = report_gen.generate(
            grading_results=results,
            proctoring_summary={
                "student_id": student_id,
                "risk_score": risk_score,
                "state": state,
                "events_count": len(events_data),
            },
        )

        student_reports.append({
            "student_id": student_id,
            "name": data.get("name", student_id),
            "grade_summary": {
                "percentage": pct,
                "total_score": results["total_score"],
                "total_max": results["total_max"],
                "correct_count": sum(
                    1 for r in results.get("per_question", {}).values()
                    if r.get("correct", False)
                ),
            },
            "proctoring_summary": {
                "risk_score": risk_score,
                "state": state,
                "events_count": len(events_data),
                "event_types": [e["type"] for e in events_data],
            },
        })

        agent.reset()

    # Examiner report
    print("\n" + "=" * 60)
    print("  EXAMINER REPORT")
    print("=" * 60)

    examiner = ExaminerReportGenerator()
    exam_report = examiner.generate("demo_exam_001", student_reports)

    stats = exam_report["statistics"]
    risk = exam_report["risk_analysis"]

    print(f"\n  Total Students: {exam_report['total_students']}")
    print(f"  Mean Score:     {stats['mean_score']:.1f}%")
    print(f"  Pass Rate:      {stats['pass_count']}/{exam_report['total_students']}")
    print(f"  High Risk:      {risk['high_risk_count']}")
    print(f"  Mean Risk:      {risk['mean_risk']:.4f}")

    flagged = exam_report.get("flagged_students", [])
    if flagged:
        print(f"\n  Flagged ({len(flagged)}):")
        for f in flagged:
            print(f"    - {f['student_id']}: risk={f['risk_score']:.4f} | {f['reason']}")

    print("\n" + examiner.to_text())

    # Save reports to file
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "exam_report": exam_report,
                "student_reports": student_reports,
            }, f, indent=2, default=str)
        print(f"\nReports saved to {out_path}")


def cmd_grade(args):
    """Grade a single answer file."""
    from src.models.models import Question, QuestionType
    from src.grading.scorer import GradingOrchestrator
    from data.sample_data import SAMPLE_QUESTIONS, load_sample_questions

    questions = [
        Question(
            id=q["id"],
            type=QuestionType(q["type"]),
            text=q["text"],
            options=q.get("options"),
            correct_answer=q.get("correct_answer"),
            marks=q.get("marks", 1.0),
            keywords=q.get("keywords", []),
            tolerance=q.get("tolerance", 0.02),
        )
        for q in SAMPLE_QUESTIONS
    ]

    orchestrator = GradingOrchestrator()
    reference = {q.id: q.correct_answer for q in questions}

    # Load student answers from JSON
    ans_path = Path(args.answers)
    if not ans_path.exists():
        print(f"Error: {ans_path} not found.")
        sys.exit(1)

    with open(ans_path) as f:
        answers = json.load(f)

    results = orchestrator.grade_all(questions, answers, reference)
    print(json.dumps(results, indent=2, default=str))


def cmd_test(args):
    """Run the test suite."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=False,
    )
    sys.exit(result.returncode)


def _events_to_signals(events: list) -> dict:
    """Convert proctoring event list to signal dict for the agent."""
    signals = {
        "face_present": True,
        "face_confidence": 1.0,
        "face_count": 1,
        "keyboard_idle_seconds": 0,
        "mouse_idle_seconds": 0,
        "tab_switch": False,
        "paste_detected": False,
        "audio_features": {"volume_db": -100, "rms_energy": 0.0},
        "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
    }
    for evt in events:
        etype = evt.get("type", "")
        if etype == "face_absent":
            signals["face_present"] = False
            signals["face_confidence"] = evt.get("confidence", 0.0)
        elif etype == "multiple_faces":
            signals["face_count"] = 2
        elif etype == "tab_switch":
            signals["tab_switch"] = True
        elif etype == "paste_detected":
            signals["paste_detected"] = True
        elif etype == "loud_audio":
            signals["audio_features"]["volume_db"] = -25
            signals["audio_features"]["rms_energy"] = 0.5
        elif etype == "head_turned":
            signals["head_pose"]["yaw"] = 50
        elif etype == "copy_paste":
            signals["paste_detected"] = True
    return signals


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Exam Proctoring & Grading Agent",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_demo = sub.add_parser("demo", help="Run full demo with sample data")
    p_demo.add_argument("--output", "-o", help="Save reports to JSON file")

    p_grade = sub.add_parser("grade", help="Grade answers from a JSON file")
    p_grade.add_argument("answers", help="Path to answers JSON file")

    p_test = sub.add_parser("test", help="Run test suite")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "grade":
        cmd_grade(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
