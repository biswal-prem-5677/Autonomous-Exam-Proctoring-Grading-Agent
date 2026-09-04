"""Multiple-choice question grading."""

import numpy as np
from typing import Dict, List, Optional, Any
from src.models.models import Question, QuestionType


class MCQGrader:
    """Grades multiple-choice questions with support for single and multi-select."""

    def __init__(self, partial_credit: bool = False):
        self.partial_credit = partial_credit

    def grade(self, question: Question, student_answer: Any) -> Dict:
        """Grade an MCQ answer.

        Args:
            question: The question object with correct_answer
            student_answer: Student's selected option(s)

        Returns:
            Dict with 'score', 'correct', 'details'
        """
        correct = question.correct_answer
        marks = question.marks

        if student_answer is None or student_answer == "":
            return {
                "score": 0.0,
                "max_score": marks,
                "correct": False,
                "details": {"reason": "No answer provided"},
            }

        student_answer = self._normalize(student_answer)
        correct = self._normalize(correct)

        # Multi-select MCQ
        if isinstance(correct, list) or isinstance(student_answer, list):
            student_set = set(student_answer) if isinstance(student_answer, list) else {student_answer}
            correct_set = set(correct) if isinstance(correct, list) else {correct}

            if student_set == correct_set:
                return {
                    "score": marks,
                    "max_score": marks,
                    "correct": True,
                    "details": {"reason": "All options correct"},
                }

            if self.partial_credit:
                overlap = len(student_set & correct_set)
                total = len(correct_set)
                partial = overlap / total * marks
                return {
                    "score": partial,
                    "max_score": marks,
                    "correct": False,
                    "details": {
                        "reason": f"Partial match: {overlap}/{total}",
                        "selected": list(student_set),
                        "correct_options": list(correct_set),
                    },
                }

            return {
                "score": 0.0,
                "max_score": marks,
                "correct": False,
                "details": {
                    "reason": "Incorrect selection",
                    "selected": list(student_set),
                    "correct_options": list(correct_set),
                },
            }

        # Single-select MCQ
        if str(student_answer).lower() == str(correct).lower():
            return {
                "score": marks,
                "max_score": marks,
                "correct": True,
                "details": {"reason": "Correct"},
            }

        return {
            "score": 0.0,
            "max_score": marks,
            "correct": False,
            "details": {
                "reason": "Incorrect",
                "selected": student_answer,
                "correct_answer": correct,
            },
        }

    def _normalize(self, value: Any) -> Any:
        """Normalize answer value for comparison."""
        if isinstance(value, str):
            return value.strip()
        return value

    def grade_batch(self, questions: List[Question],
                    answers: Dict[str, Any]) -> Dict:
        """Grade a batch of MCQ answers.

        Args:
            questions: List of MCQ questions
            answers: Dict mapping question_id -> student answer

        Returns:
            Dict with total score, per-question results
        """
        results = {}
        total_score = 0.0
        total_max = 0.0

        for q in questions:
            if q.type not in (QuestionType.MCQ, QuestionType.TRUE_FALSE):
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
