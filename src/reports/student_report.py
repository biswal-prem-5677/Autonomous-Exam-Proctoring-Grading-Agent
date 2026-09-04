"""Student report generator — produces individual exam reports."""

from typing import Dict, Any, List, Optional
from datetime import datetime


class StudentReportGenerator:
    """Generates individual student exam reports.

    Includes:
    - Score breakdown by question type
    - Proctoring risk summary
    - Performance prediction
    - Improvement suggestions
    """

    def __init__(self):
        self.report_data: Dict[str, Any] = {}

    def generate(self, grading_results: Dict[str, Any],
                 proctoring_summary: Dict[str, Any] = None,
                 prediction: Dict[str, Any] = None,
                 knowledge_state: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate a comprehensive student report.

        Args:
            grading_results: Output from GradingOrchestrator.grade_all()
            proctoring_summary: Risk scores and evidence summary
            prediction: Performance prediction results
            knowledge_state: Per-topic mastery levels

        Returns:
            Structured report dictionary
        """
        student_id = proctoring_summary.get("student_id", "Unknown") if proctoring_summary else "Unknown"

        report = {
            "report_type": "student_report",
            "generated_at": datetime.now().isoformat(),
            "student_id": student_id,
            "grade_summary": self._build_grade_summary(grading_results),
            "proctoring_summary": proctoring_summary or {},
            "prediction": prediction or {},
            "knowledge_state": knowledge_state or {},
            "recommendations": self._build_recommendations(
                grading_results, proctoring_summary, knowledge_state
            ),
        }
        self.report_data = report
        return report

    def _build_grade_summary(self, grading_results: Dict) -> Dict[str, Any]:
        """Build grade summary section."""
        return {
            "total_score": grading_results.get("total_score", 0.0),
            "total_max": grading_results.get("total_max", 0.0),
            "percentage": round(grading_results.get("percentage", 0.0), 2),
            "per_question": grading_results.get("per_question", {}),
            "correct_count": sum(
                1 for r in grading_results.get("per_question", {}).values()
                if r.get("correct", False)
            ),
        }

    def _build_recommendations(self, grading_results: Dict,
                               proctoring_summary: Dict = None,
                               knowledge_state: Dict = None) -> List[str]:
        """Generate actionable improvement recommendations."""
        recs = []
        percentage = grading_results.get("percentage", 0.0)

        if percentage < 40:
            recs.append("Review fundamental concepts before the next exam.")
        elif percentage < 70:
            recs.append("Focus on weak areas identified in the question breakdown.")

        if knowledge_state:
            weak = [t for t, m in knowledge_state.items() if m < 0.4]
            if weak:
                recs.append(f"Prioritize studying: {', '.join(weak)}")

        if proctoring_summary:
            risk = proctoring_summary.get("risk_score", 0.0)
            if risk >= 0.5:
                recs.append("Proctoring alerts were raised — ensure exam conditions are optimal.")

        if not recs:
            recs.append("Good performance. Continue current study practices.")

        return recs

    def to_text(self) -> str:
        """Generate a human-readable text report."""
        if not self.report_data:
            return "No report generated."

        lines = [
            "=" * 60,
            "           EXAMINATION REPORT",
            "=" * 60,
            f"Generated: {self.report_data.get('generated_at', 'N/A')}",
            f"Student: {self.report_data.get('student_id', 'N/A')}",
            "-" * 60,
            "",
            "GRADE SUMMARY",
            f"  Score: {self.report_data['grade_summary']['total_score']:.1f} / "
            f"{self.report_data['grade_summary']['total_max']:.1f}",
            f"  Percentage: {self.report_data['grade_summary']['percentage']:.1f}%",
            "",
            "RECOMMENDATIONS",
        ]

        for i, rec in enumerate(self.report_data.get("recommendations", []), 1):
            lines.append(f"  {i}. {rec}")

        lines.extend([
            "",
            "=" * 60,
            "  Risk score is NOT proof of cheating.",
            "  It indicates confidence level for examiner review.",
            "=" * 60,
        ])

        return "\n".join(lines)
