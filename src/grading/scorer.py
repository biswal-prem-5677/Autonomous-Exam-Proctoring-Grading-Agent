"""Unified grading orchestrator — routes questions to appropriate graders."""

from typing import Dict, Any, List, Optional
from src.models.models import Question, QuestionType
from src.grading.mcq import MCQGrader
from src.grading.numerical import NumericalGrader
from src.grading.tfidf import TFIDFScorer, KeywordCoverageScorer


class GradingOrchestrator:
    """Routes each question to its appropriate grader and aggregates results."""

    def __init__(self):
        self.mcq_grader = MCQGrader(partial_credit=True)
        self.numerical_grader = NumericalGrader(default_tolerance=0.02)
        self.tfidf_scorer = TFIDFScorer()
        self.keyword_scorer = KeywordCoverageScorer(partial_threshold=0.6)

    def grade_all(self, questions: List[Question],
                  answers: Dict[str, Any],
                  reference_answers: Dict[str, str] = None) -> Dict[str, Any]:
        """Grade all answers across all question types.

        Args:
            questions: List of all questions
            answers: Dict mapping question_id -> student answer
            reference_answers: Reference answers for short-answer TF-IDF (optional)

        Returns:
            Aggregated grading results
        """
        mcq_results = self.grade_mcq(questions, answers)
        numerical_results = self.grade_numerical(questions, answers)
        short_answer_results = self.grade_short_answer(questions, answers, reference_answers)

        return self._aggregate([mcq_results, numerical_results, short_answer_results])

    def grade_mcq(self, questions: List[Question],
                  answers: Dict[str, Any]) -> Dict[str, Any]:
        """Grade MCQ and true/false questions."""
        mcq_questions = [q for q in questions if q.type in (QuestionType.MCQ, QuestionType.TRUE_FALSE)]
        return self.mcq_grader.grade_batch(mcq_questions, answers)

    def grade_numerical(self, questions: List[Question],
                        answers: Dict[str, Any]) -> Dict[str, Any]:
        """Grade numerical questions."""
        num_questions = [q for q in questions if q.type == QuestionType.NUMERICAL]
        return self.numerical_grader.grade_batch(num_questions, answers)

    def grade_short_answer(self, questions: List[Question],
                           answers: Dict[str, Any],
                           reference_answers: Dict[str, str] = None) -> Dict[str, Any]:
        """Grade short-answer questions using TF-IDF + keyword coverage."""
        short_qs = [q for q in questions if q.type == QuestionType.SHORT_ANSWER]
        if not short_qs:
            return {"per_question": {}, "total_score": 0.0, "total_max": 0.0, "percentage": 0.0}

        # Build reference corpus for TF-IDF
        refs = []
        if reference_answers:
            refs = [reference_answers.get(q.id, q.text) for q in short_qs]
        else:
            refs = [q.text for q in short_qs]

        self.tfidf_scorer.fit(refs)

        per_question = {}
        total_score = 0.0
        total_max = 0.0

        for q in short_qs:
            student_ans = answers.get(q.id, "")
            if isinstance(student_ans, dict):
                student_ans = student_ans.get("answer", "")
            student_str = str(student_ans)

            ref_answer = reference_answers.get(q.id, q.text) if reference_answers else q.text

            # TF-IDF cosine similarity against reference
            cosine_sim = self.tfidf_scorer.cosine_similarity(student_str, ref_answer)

            # Keyword coverage
            kw_result = self.keyword_scorer.score(student_str, q.keywords)
            kw_coverage = kw_result.get("coverage", 0.0)

            # Combined score
            combined_sim = 0.6 * cosine_sim + 0.4 * kw_coverage
            score = combined_sim * q.marks

            per_question[q.id] = {
                "score": round(score, 4),
                "max_score": q.marks,
                "correct": combined_sim >= 0.8,
                "cosine_similarity": round(cosine_sim, 4),
                "keyword_coverage": round(kw_coverage, 4),
                "combined_similarity": round(combined_sim, 4),
                "details": {
                    "method": "tfidf_keyword",
                    "reference": ref_answer[:100],
                    "matched_keywords": kw_result.get("matched", []),
                },
            }
            total_score += score
            total_max += q.marks

        return {
            "per_question": per_question,
            "total_score": total_score,
            "total_max": total_max,
            "percentage": (total_score / total_max * 100) if total_max > 0 else 0.0,
        }

    def _aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        """Merge multiple grading result dicts into one."""
        combined = {
            "per_question": {},
            "total_score": 0.0,
            "total_max": 0.0,
            "breakdown": {},
        }
        for r in results:
            combined["per_question"].update(r.get("per_question", {}))
            combined["total_score"] += r.get("total_score", 0.0)
            combined["total_max"] += r.get("total_max", 0.0)
        combined["percentage"] = (
            combined["total_score"] / combined["total_max"] * 100
            if combined["total_max"] > 0 else 0.0
        )
        return combined
