"""Tests for long-answer / descriptive grading engine."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLongAnswerGrader:
    def test_identical_answers_high_score(self):
        from src.grading.long_answer import LongAnswerGrader
        grader = LongAnswerGrader()
        ref = "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration."
        result = grader.grade(student_answer=ref, reference_answer=ref, marks=10.0)
        assert result.final_score > 7.0
        assert result.semantic_score > 0.8

    def test_empty_answer_zero_score(self):
        from src.grading.long_answer import LongAnswerGrader
        grader = LongAnswerGrader()
        result = grader.grade(
            student_answer="",
            reference_answer="The mitochondria produces ATP.",
            marks=10.0,
        )
        assert result.final_score == 0.0

    def test_keyword_coverage_impacts_score(self):
        from src.grading.long_answer import LongAnswerGrader
        grader = LongAnswerGrader()
        ref = "Photosynthesis converts light energy into chemical energy using chlorophyll."
        student = "Photosynthesis uses chlorophyll."
        result = grader.grade(
            student_answer=student,
            reference_answer=ref,
            keywords=["photosynthesis", "chlorophyll", "light energy", "chemical energy"],
            marks=10.0,
        )
        assert result.keyword_score >= 0.3  # At least 2 of 4 keywords
        assert result.details["matched_keywords"] == ["photosynthesis", "chlorophyll"]

    def test_structure_score_short_answer(self):
        """Short answers with matching sentence count get decent structure score;
        very terse answers with far fewer sentences get penalized."""
        from src.grading.long_answer import LongAnswerGrader
        grader = LongAnswerGrader()
        ref = "Newton's first law states that an object at rest stays at rest. An object in motion stays in motion. This is the law of inertia."
        result = grader.grade(
            student_answer="Inertia.",
            reference_answer=ref,
            marks=10.0,
        )
        # Student has 1 sentence vs reference's 3 — ratio 1/3 < 0.5 → sent_score = ratio * 2 ≈ 0.67
        assert result.structure_score < 0.7

    def test_weights_sum_to_one(self):
        from src.grading.long_answer import LongAnswerGrader
        grader = LongAnswerGrader(
            w_semantic=0.4, w_keyword=0.3, w_structure=0.2, w_coverage=0.1
        )
        result = grader.grade(
            student_answer="Test answer about science.",
            reference_answer="A comprehensive answer about science and technology.",
            marks=10.0,
        )
        assert result.final_score <= 10.0
        assert result.final_score >= 0.0

    def test_invalid_weights_raises(self):
        from src.grading.long_answer import LongAnswerGrader
        with pytest.raises(ValueError):
            LongAnswerGrader(w_semantic=0.5, w_keyword=0.5, w_structure=0.5, w_coverage=0.5)

    def test_batch_grading(self):
        from src.grading.long_answer import LongAnswerGrader
        grader = LongAnswerGrader()
        answers = {
            "q1": "The heart pumps blood through the body.",
            "q2": "Photosynthesis occurs in chloroplasts.",
        }
        refs = {
            "q1": "The heart is a muscular organ that pumps blood throughout the circulatory system.",
            "q2": "Photosynthesis is the process by which plants convert light energy into chemical energy in chloroplasts.",
        }
        kws = {
            "q1": ["heart", "blood", "pump"],
            "q2": ["photosynthesis", "chloroplasts", "light energy"],
        }
        result = grader.grade_batch(answers, refs, keywords_map=kws, marks_map={"q1": 10.0, "q2": 10.0})
        assert "q1" in result["per_question"]
        assert "q2" in result["per_question"]
        assert result["total_max"] == 20.0

    def test_descriptive_grader_routes_correctly(self):
        from src.grading.long_answer import DescriptiveGrader
        from src.models.models import Question, QuestionType
        grader = DescriptiveGrader()

        q = Question(
            id="q1", type=QuestionType.LONG_ANSWER,
            text="Explain the theory of relativity.",
            marks=10.0,
            keywords=["relativity", "speed of light", "time dilation"],
        )
        result = grader.grade_question(
            question=q,
            student_answer="Einstein's theory of relativity states that the speed of light is constant and time dilation occurs at high velocities.",
        )
        assert result["type"] == "long_answer"
        assert result["score"] >= 0.0
        assert result["max_score"] == 10.0
        assert "semantic_similarity" in result

    def test_descriptive_grader_routes_mcq(self):
        from src.grading.long_answer import DescriptiveGrader
        from src.models.models import Question, QuestionType
        grader = DescriptiveGrader()

        q = Question(
            id="q1", type=QuestionType.MCQ,
            text="What is 2+2?",
            options=["3", "4", "5", "6"],
            correct_answer="4",
            marks=1.0,
        )
        result = grader.grade_question(question=q, student_answer="4")
        assert result["type"] == "mcq"
        assert result["correct"] is True

    def test_descriptive_grader_routes_numerical(self):
        from src.grading.long_answer import DescriptiveGrader
        from src.models.models import Question, QuestionType
        grader = DescriptiveGrader()

        q = Question(
            id="q1", type=QuestionType.NUMERICAL,
            text="What is 3.14 * 2?",
            correct_answer=6.28,
            marks=2.0,
        )
        result = grader.grade_question(question=q, student_answer=6.28)
        assert result["type"] == "numerical"
        assert result["correct"] is True

    def test_descriptive_grader_routes_short_answer(self):
        from src.grading.long_answer import DescriptiveGrader
        from src.models.models import Question, QuestionType
        grader = DescriptiveGrader()

        q = Question(
            id="q1", type=QuestionType.SHORT_ANSWER,
            text="Define osmosis.",
            keywords=["osmosis", "semipermeable", "concentration"],
            marks=5.0,
        )
        result = grader.grade_question(
            question=q,
            student_answer="Osmosis is the movement of water across a semipermeable membrane.",
        )
        assert result["type"] == "short_answer"
        assert result["score"] >= 0.0

    def test_coverage_score_reference_words(self):
        from src.grading.long_answer import LongAnswerGrader
        grader = LongAnswerGrader()
        ref = "The water cycle includes evaporation, condensation, and precipitation."
        # Student answer that shares some words with reference
        student = "Evaporation and condensation are part of the water cycle."
        result = grader.grade(
            student_answer=student,
            reference_answer=ref,
            marks=10.0,
        )
        assert result.coverage_score > 0.2

    def test_long_answer_in_orchestrator(self):
        """Verify the orchestrator now routes long-answer questions."""
        from src.grading.scorer import GradingOrchestrator
        from src.models.models import Question, QuestionType

        orchestrator = GradingOrchestrator()
        questions = [
            Question(
                id="q1", type=QuestionType.LONG_ANSWER,
                text="Explain how neural networks learn.",
                marks=10.0,
                keywords=["neural", "weights", "backpropagation", "gradient"],
            ),
        ]
        answers = {
            "q1": "Neural networks learn by adjusting weights through backpropagation and gradient descent to minimize error.",
        }
        results = orchestrator.grade_all(questions, answers)
        assert "q1" in results["per_question"]
        assert results["per_question"]["q1"]["score"] > 0.0
        assert "semantic_similarity" in results["per_question"]["q1"]


class TestQuestionTypeEnum:
    def test_long_answer_enum_exists(self):
        from src.models.models import QuestionType
        assert QuestionType.LONG_ANSWER.value == "long_answer"

    def test_all_question_types(self):
        from src.models.models import QuestionType
        types = [qt.value for qt in QuestionType]
        assert "mcq" in types
        assert "numerical" in types
        assert "short_answer" in types
        assert "long_answer" in types
