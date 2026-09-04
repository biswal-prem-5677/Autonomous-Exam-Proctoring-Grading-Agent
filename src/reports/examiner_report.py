"""Examiner report generator — produces administration-level reports."""

from typing import Dict, Any, List, Optional
from datetime import datetime


class ExaminerReportGenerator:
    """Generates examiner-level reports for multiple students.

    Includes:
    - Aggregate statistics
    - Risk distribution across students
    - Anomaly flags
    - Evidence timeline
    - Cheating risk indicators (for review, not accusation)
    """

    def __init__(self):
        self.report_data: Dict[str, Any] = {}

    def generate(self, exam_id: str, student_reports: List[Dict[str, Any]],
                 exam_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate an examiner report for an entire exam session.

        Args:
            exam_id: Identifier for the exam
            student_reports: List of per-student report data
            exam_config: Exam configuration details

        Returns:
            Structured examiner report
        """
        stats = self._compute_statistics(student_reports)
        risk_analysis = self._analyze_risk(student_reports)

        report = {
            "report_type": "examiner_report",
            "exam_id": exam_id,
            "generated_at": datetime.now().isoformat(),
            "exam_config": exam_config or {},
            "total_students": len(student_reports),
            "statistics": stats,
            "risk_analysis": risk_analysis,
            "flagged_students": self._flag_students(student_reports),
            "student_summaries": [
                {
                    "student_id": sr.get("student_id"),
                    "score": sr.get("grade_summary", {}).get("percentage", 0.0),
                    "risk_score": sr.get("proctoring_summary", {}).get("risk_score", 0.0),
                    "state": sr.get("proctoring_summary", {}).get("state", "NORMAL"),
                }
                for sr in student_reports
            ],
        }
        self.report_data = report
        return report

    def _compute_statistics(self, student_reports: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        if not student_reports:
            return {"mean_score": 0, "median_score": 0, "std_score": 0,
                    "min_score": 0, "max_score": 0}

        scores = [sr.get("grade_summary", {}).get("percentage", 0.0)
                  for sr in student_reports]
        scores_sorted = sorted(scores)

        n = len(scores)
        median = scores_sorted[n // 2] if n % 2 == 1 else (
            (scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2
        )

        return {
            "mean_score": round(sum(scores) / n, 2),
            "median_score": round(median, 2),
            "std_score": round(
                (sum((s - sum(scores) / n) ** 2 for s in scores) / n) ** 0.5, 2
            ) if n > 0 else 0,
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
            "pass_count": sum(1 for s in scores if s >= 40),
            "fail_count": sum(1 for s in scores if s < 40),
        }

    def _analyze_risk(self, student_reports: List[Dict]) -> Dict[str, Any]:
        """Analyze proctoring risk distribution."""
        risk_scores = [
            sr.get("proctoring_summary", {}).get("risk_score", 0.0)
            for sr in student_reports
        ]

        if not risk_scores:
            return {"mean_risk": 0, "high_risk_count": 0}

        return {
            "mean_risk": round(sum(risk_scores) / len(risk_scores), 4),
            "max_risk": round(max(risk_scores), 4),
            "high_risk_count": sum(1 for r in risk_scores if r >= 0.7),
            "suspicious_count": sum(1 for r in risk_scores if 0.5 <= r < 0.7),
            "normal_count": sum(1 for r in risk_scores if r < 0.5),
        }

    def _flag_students(self, student_reports: List[Dict]) -> List[Dict[str, Any]]:
        """Flag students requiring human review."""
        flagged = []
        for sr in student_reports:
            risk = sr.get("proctoring_summary", {}).get("risk_score", 0.0)
            if risk >= 0.5:
                flagged.append({
                    "student_id": sr.get("student_id"),
                    "risk_score": risk,
                    "reason": self._flag_reason(sr),
                })
        return flagged

    def _flag_reason(self, student_report: Dict) -> str:
        """Determine why a student was flagged."""
        risk = student_report.get("proctoring_summary", {}).get("risk_score", 0.0)
        if risk >= 0.7:
            return "High cheating risk — multiple strong signals detected"
        state = student_report.get("proctoring_summary", {}).get("state", "")
        if state == "HIGH_RISK":
            return "Escalated state — requires immediate review"
        return "Suspicious behavior detected — multi-signal convergence"

    def to_text(self) -> str:
        """Generate human-readable examiner report."""
        if not self.report_data:
            return "No report generated."

        stats = self.report_data.get("statistics", {})
        risk = self.report_data.get("risk_analysis", {})

        lines = [
            "=" * 60,
            "       EXAMINER REPORT",
            f"       Exam: {self.report_data.get('exam_id', 'N/A')}",
            f"  Generated: {self.report_data.get('generated_at', 'N/A')}",
            "=" * 60,
            "",
            "OVERVIEW",
            f"  Total Students: {self.report_data.get('total_students', 0)}",
            "",
            "GRADE STATISTICS",
            f"  Mean Score:   {stats.get('mean_score', 0):.1f}%",
            f"  Median Score: {stats.get('median_score', 0):.1f}%",
            f"  Std Dev:      {stats.get('std_score', 0):.1f}%",
            f"  Range:        {stats.get('min_score', 0):.1f}% - {stats.get('max_score', 0):.1f}%",
            f"  Passed:       {stats.get('pass_count', 0)}",
            f"  Failed:       {stats.get('fail_count', 0)}",
            "",
            "RISK ANALYSIS",
            f"  Mean Risk Score:    {risk.get('mean_risk', 0):.4f}",
            f"  High Risk:          {risk.get('high_risk_count', 0)}",
            f"  Suspicious:         {risk.get('suspicious_count', 0)}",
            f"  Normal:             {risk.get('normal_count', 0)}",
            "",
            "FLAGGED FOR REVIEW",
        ]

        flagged = self.report_data.get("flagged_students", [])
        if flagged:
            for f in flagged:
                lines.append(f"  - {f['student_id']}: risk={f['risk_score']:.4f} | {f['reason']}")
        else:
            lines.append("  None")

        lines.extend([
            "",
            "=" * 60,
            "  Risk scores are evidence indicators, NOT proof of cheating.",
            "  Review flagged cases before taking any action.",
            "=" * 60,
        ])

        return "\n".join(lines)
