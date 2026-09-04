"""Numerical answer grading with tolerance-based evaluation."""

from typing import Dict, Any, Optional
from src.models.models import Question, QuestionType


class NumericalGrader:
    """Grades numerical answers with configurable tolerance."""

    def __init__(self, default_tolerance: float = 0.02):
        self.default_tolerance = default_tolerance

    def grade(self, question: Question, student_answer: Any) -> Dict:
        """Grade a numerical answer with tolerance check.

        Uses question tolerance if explicitly set, otherwise grader default.
        When tolerance is 0, requires exact match.

        Args:
            question: The question with correct_answer and tolerance
            student_answer: Student's numerical answer

        Returns:
            Dict with score, details
        """
        correct = question.correct_answer
        marks = question.marks
        tolerance = question.tolerance if question.tolerance is not None else self.default_tolerance

        if student_answer is None or student_answer == "":
            return {
                "score": 0.0,
                "max_score": marks,
                "correct": False,
                "details": {"reason": "No answer provided"},
            }

        try:
            student_val = float(student_answer)
            correct_val = float(correct)
        except (ValueError, TypeError):
            return {
                "score": 0.0,
                "max_score": marks,
                "correct": False,
                "details": {
                    "reason": "Non-numeric answer",
                    "student_answer": str(student_answer),
                },
            }

        # Compute relative or absolute tolerance
        if tolerance == 0:
            is_correct = student_val == correct_val
        elif correct_val != 0:
            relative_error = abs(student_val - correct_val) / abs(correct_val)
            is_correct = relative_error <= tolerance
        else:
            is_correct = abs(student_val) <= tolerance

        if is_correct:
            return {
                "score": marks,
                "max_score": marks,
                "correct": True,
                "details": {"reason": "Correct"},
            }

        # Partial credit for near-miss (within 2x tolerance)
        if tolerance > 0 and correct_val != 0:
            relative_error = abs(student_val - correct_val) / abs(correct_val)
            if relative_error <= tolerance * 2:
                partial = marks * 0.5
                return {
                    "score": partial,
                    "max_score": marks,
                    "correct": False,
                    "details": {"reason": "Near miss (partial credit)"},
                }

        return {
            "score": 0.0,
            "max_score": marks,
            "correct": False,
            "details": {"reason": "Outside tolerance"},
        }

    def grade_batch(self, questions: list,
                    answers: Dict[str, Any]) -> Dict:
        """Grade a batch of numerical answers."""
        results = {}
        total_score = 0.0
        total_max = 0.0

        for q in questions:
            if q.type != QuestionType.NUMERICAL:
                continue
            answer = answers.get(q.id)
            result = self.grade(q, answer)
            results[q.id] = result
            total_score += result["score"]
            total_max += result["max_score"]

        return {
            "per_question": results,
            "total_score": total_score,
            "total_max": total_max,
            "percentage": (total_score / total_max * 100) if total_max > 0 else 0.0,
        }
