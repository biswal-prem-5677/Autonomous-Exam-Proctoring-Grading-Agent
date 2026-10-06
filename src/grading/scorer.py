"""Unified grading orchestrator — routes questions to appropriate graders."""

from typing import Dict, Any, List
from src.models.models import Question, QuestionType
from src.grading.mcq import MCQGrader
from src.grading.numerical import NumericalGrader
from src.grading.tfidf import TFIDFScorer, KeywordCoverageScorer
from src.grading.long_answer import LongAnswerGrader
from src.grading.mathematical import MathematicalGrader
from src.grading.programming import ProgrammingGrader
from src.grading.plagiarism import PlagiarismDetector
from src.grading.consensus import GraderDisagreementEngine


class GradingOrchestrator:
    """Routes each question to its appropriate grader and aggregates results."""

    def __init__(self):
        self.mcq_grader = MCQGrader(partial_credit=True)
        self.numerical_grader = NumericalGrader(default_tolerance=0.02)
        self.tfidf_scorer = TFIDFScorer()
        self.keyword_scorer = KeywordCoverageScorer(partial_threshold=0.6)
        self.long_answer_grader = LongAnswerGrader()
        self.mathematical_grader = MathematicalGrader()
        self.programming_grader = ProgrammingGrader()
        self.plagiarism_detector = PlagiarismDetector()
        self.consensus_engine = GraderDisagreementEngine()

    def grade_all(self, questions: List[Question],
                  answers: Dict[str, Any],
                  reference_answers: Dict[str, Any] = None) -> Dict[str, Any]:
        """Grade all answers across all question types.

        Routes to:
        - MCQ / True-False → exact match
        - Numerical → tolerance-based
        - Short Answer → TF-IDF + keyword coverage
        - Long/Descriptive Answer → multi-signal (semantic + keyword + structure + coverage)
        - Mathematical → step-by-step derivation + formula + units + partial credit
        - Programming → AST sandbox + security analysis + test-suite evaluation
        """
        mcq_results = self.grade_mcq(questions, answers)
        numerical_results = self.grade_numerical(questions, answers)
        short_answer_results = self.grade_short_answer(questions, answers, reference_answers)
        long_answer_results = self.grade_long_answer(questions, answers, reference_answers)
        math_results = self.grade_mathematical(questions, answers, reference_answers)
        prog_results = self.grade_programming(questions, answers)

        return self._aggregate([
            mcq_results, numerical_results, short_answer_results,
            long_answer_results, math_results, prog_results
        ])

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

    def grade_long_answer(self, questions: List[Question],
                          answers: Dict[str, Any],
                          reference_answers: Dict[str, str] = None) -> Dict[str, Any]:
        """Grade long/descriptive answer questions using multi-signal scoring.

        Score = w1 * S_semantic + w2 * S_keyword + w3 * S_structure + w4 * S_coverage
        """
        long_qs = [q for q in questions if q.type == QuestionType.LONG_ANSWER]
        if not long_qs:
            return {"per_question": {}, "total_score": 0.0, "total_max": 0.0, "percentage": 0.0}

        per_question = {}
        total_score = 0.0
        total_max = 0.0

        for q in long_qs:
            student_ans = answers.get(q.id, "")
            if isinstance(student_ans, dict):
                student_ans = student_ans.get("answer", "")
            student_str = str(student_ans)

            ref_answer = reference_answers.get(q.id, q.text) if reference_answers else q.text

            result = self.long_answer_grader.grade(
                student_answer=student_str,
                reference_answer=ref_answer,
                keywords=q.keywords or [],
                marks=q.marks,
                question_text=q.text,
            )

            per_question[q.id] = {
                "score": round(result.final_score, 4),
                "max_score": result.max_score,
                "semantic_similarity": round(result.semantic_score, 4),
                "keyword_coverage": round(result.keyword_score, 4),
                "structure_score": round(result.structure_score, 4),
                "coverage_score": round(result.coverage_score, 4),
                "details": result.details,
            }
            total_score += result.final_score
            total_max += result.max_score

        return {
            "per_question": per_question,
            "total_score": total_score,
            "total_max": total_max,
            "percentage": (total_score / total_max * 100) if total_max > 0 else 0.0,
        }

    def grade_mathematical(self, questions: List[Question],
                           answers: Dict[str, Any],
                           reference_answers: Dict[str, Any] = None) -> Dict[str, Any]:
        """Grade multi-step mathematical derivation questions with partial credit."""
        math_qs = [q for q in questions if q.type == QuestionType.MATHEMATICAL]
        if not math_qs:
            return {"per_question": {}, "total_score": 0.0, "total_max": 0.0, "percentage": 0.0}

        per_question = {}
        total_score = 0.0
        total_max = 0.0

        for q in math_qs:
            student_ans = answers.get(q.id, "")
            if isinstance(student_ans, dict):
                student_ans = student_ans.get("answer", "")
            student_str = str(student_ans)

            meta = q.metadata or {}
            ref = (reference_answers.get(q.id) if reference_answers else None) or {}
            if isinstance(ref, str):
                ref_dict = {"expected_formula": ref, "final_value": q.correct_answer}
            else:
                ref_dict = ref or {}

            expected_formula = ref_dict.get("expected_formula", meta.get("governing_formula", meta.get("formula")))
            required_params = ref_dict.get("required_params", meta.get("required_params", meta.get("parameters", [])))
            derivation_steps = ref_dict.get("expected_derivation_steps", meta.get("expected_derivation_steps", 2))
            final_val = ref_dict.get("final_value", q.correct_answer if isinstance(q.correct_answer, (int, float)) else meta.get("final_value"))
            units = ref_dict.get("units", meta.get("units"))
            tolerance = q.tolerance if q.tolerance is not None else 0.02

            result = self.mathematical_grader.grade(
                student_solution=student_str,
                governing_formula=expected_formula,
                required_parameters=required_params,
                expected_derivation_steps=derivation_steps,
                expected_final_value=final_val,
                expected_units=units,
                tolerance=tolerance,
                marks=q.marks,
            )

            per_question[q.id] = {
                "score": round(result.score, 4),
                "max_score": result.max_score,
                "percentage": round(result.percentage, 2),
                "formula_credit": round(result.formula_credit, 2),
                "substitution_credit": round(result.substitution_credit, 2),
                "intermediate_credit": round(result.intermediate_credit, 2),
                "final_answer_credit": round(result.final_answer_credit, 2),
                "units_credit": round(result.units_credit, 2),
                "explanation": result.explanation,
                "details": result.to_dict(),
            }
            total_score += result.score
            total_max += result.max_score

        return {
            "per_question": per_question,
            "total_score": round(total_score, 4),
            "total_max": round(total_max, 4),
            "percentage": (total_score / total_max * 100) if total_max > 0 else 0.0,
        }

    def grade_programming(self, questions: List[Question],
                          answers: Dict[str, Any],
                          test_suites: Dict[str, List[Any]] = None) -> Dict[str, Any]:
        """Grade programming questions with AST sandbox, vulnerability detection & execution."""
        prog_qs = [q for q in questions if q.type == QuestionType.PROGRAMMING]
        if not prog_qs:
            return {"per_question": {}, "total_score": 0.0, "total_max": 0.0, "percentage": 0.0}

        per_question = {}
        total_score = 0.0
        total_max = 0.0

        for q in prog_qs:
            student_code = answers.get(q.id, "")
            if isinstance(student_code, dict):
                student_code = student_code.get("code", student_code.get("answer", ""))
            student_str = str(student_code)

            meta = q.metadata or {}
            func_name = meta.get("function_name", meta.get("target_function", "solution"))
            tests = (test_suites.get(q.id) if test_suites else None) or meta.get("tests", [])

            result = self.programming_grader.grade(
                student_code=student_str,
                target_function_name=func_name,
                test_cases=tests,
                marks=q.marks,
            )

            per_question[q.id] = {
                "score": round(result.score, 4),
                "max_score": result.max_score,
                "percentage": round(result.percentage, 2),
                "syntax_valid": result.syntax_valid,
                "is_safe": result.security.is_safe,
                "passed_tests": result.passed_tests,
                "total_tests": result.total_tests,
                "cyclomatic_complexity": result.cyclomatic_complexity,
                "explanation": result.explanation,
                "details": result.to_dict(),
            }
            total_score += result.score
            total_max += result.max_score

        return {
            "per_question": per_question,
            "total_score": round(total_score, 4),
            "total_max": round(total_max, 4),
            "percentage": (total_score / total_max * 100) if total_max > 0 else 0.0,
        }

    def check_plagiarism(self, submissions: Dict[str, str],
                         question_id: str = "q_general",
                         threshold: float = 0.6):
        """Check plagiarism across student submissions using MinHash & LSH."""
        return self.plagiarism_detector.detect_plagiarism(
            submissions=submissions,
            question_id=question_id,
            threshold=threshold,
        )

    def evaluate_consensus(self, question_text: str,
                           student_answer: str,
                           rubric: Dict[str, Any],
                           marks: float = 10.0):
        """Evaluate multi-perspective consensus on a student answer."""
        return self.consensus_engine.evaluate_perspectives(
            question_text=question_text,
            student_answer=student_answer,
            rubric=rubric,
            marks=marks,
        )

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
