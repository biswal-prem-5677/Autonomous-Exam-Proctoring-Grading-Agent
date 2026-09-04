"""Long-answer / descriptive grading — structure, concept coverage, and coherence."""

import re
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.grading.tfidf import TFIDFScorer, KeywordCoverageScorer


@dataclass
class LongAnswerScore:
    """Score breakdown for a long-answer question."""
    semantic_score: float = 0.0        # TF-IDF cosine similarity
    keyword_score: float = 0.0         # Concept coverage
    structure_score: float = 0.0       # Organization quality
    coverage_score: float = 0.0        # Breadth of concepts
    final_score: float = 0.0
    max_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class LongAnswerGrader:
    """Multi-signal long-answer grading.

    Formula (per the architecture plan):
        Score = w1 * S_semantic + w2 * S_keyword + w3 * S_structure + w4 * S_coverage

    All weights sum to 1.0 and are configurable.
    """

    def __init__(
        self,
        w_semantic: float = 0.35,
        w_keyword: float = 0.25,
        w_structure: float = 0.20,
        w_coverage: float = 0.20,
    ):
        total = w_semantic + w_keyword + w_structure + w_coverage
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.w_semantic = w_semantic
        self.w_keyword = w_keyword
        self.w_structure = w_structure
        self.w_coverage = w_coverage
        self._tfidf = TFIDFScorer()
        self._kw_scorer = KeywordCoverageScorer()

    def grade(
        self,
        student_answer: str,
        reference_answer: str,
        keywords: List[str] = None,
        marks: float = 10.0,
        question_text: str = "",
        concept_weights: Dict[str, float] = None,
    ) -> LongAnswerScore:
        """Grade a long-answer response.

        Args:
            student_answer: The student's full response text.
            reference_answer: The model/expected answer text.
            keywords: Key concepts that should appear in the answer.
            marks: Maximum marks for this question.
            question_text: The question itself (used to fit TF-IDF).
            concept_weights: Optional dict of concept -> importance weight.

        Returns:
            LongAnswerScore with per-component breakdown.
        """
        if not student_answer or not student_answer.strip():
            # Complete blank answer
            return LongAnswerScore(
                semantic_score=0.0,
                keyword_score=0.0,
                structure_score=0.0,
                coverage_score=0.0,
                final_score=0.0,
                max_score=marks,
                details={"method": "multi_signal", "reason": "empty_answer"},
            )

        keywords = keywords or []
        concept_weights = concept_weights or {}

        # ── 1. Semantic similarity (TF-IDF cosine) ───────────────────────
        corpus = [reference_answer]
        if question_text:
            corpus.insert(0, question_text)
        self._tfidf.fit(corpus)
        semantic_sim = self._tfidf.cosine_similarity(student_answer, reference_answer)

        # ── 2. Keyword / concept coverage ───────────────────────────────
        kw_result = self._kw_scorer.score(student_answer, keywords)
        base_coverage = kw_result.get("coverage", 0.0)

        # Weighted concept coverage (if importance weights provided)
        if concept_weights and keywords:
            weighted_matched = sum(
                concept_weights.get(kw, 1.0) for kw in kw_result.get("matched", [])
            )
            weighted_total = sum(concept_weights.get(kw, 1.0) for kw in keywords)
            weighted_coverage = weighted_matched / weighted_total if weighted_total > 0 else 0.0
            keyword_score = 0.5 * base_coverage + 0.5 * weighted_coverage
        else:
            keyword_score = base_coverage

        # ── 3. Structure score ──────────────────────────────────────────
        structure_score = self._score_structure(student_answer, reference_answer)

        # ── 4. Concept breadth coverage ─────────────────────────────────
        coverage_score = self._score_coverage(student_answer, reference_answer, keywords)

        # ── Weighted combination ────────────────────────────────────────
        combined = (
            self.w_semantic * semantic_sim
            + self.w_keyword * keyword_score
            + self.w_structure * structure_score
            + self.w_coverage * coverage_score
        )

        result = LongAnswerScore(
            semantic_score=semantic_sim,
            keyword_score=keyword_score,
            structure_score=structure_score,
            coverage_score=coverage_score,
            final_score=combined * marks,
            max_score=marks,
            details={
                "method": "multi_signal",
                "weights": {
                    "semantic": self.w_semantic,
                    "keyword": self.w_keyword,
                    "structure": self.w_structure,
                    "coverage": self.w_coverage,
                },
                "semantic_raw": round(semantic_sim, 4),
                "keyword_raw": round(keyword_score, 4),
                "structure_raw": round(structure_score, 4),
                "coverage_raw": round(coverage_score, 4),
                "combined_raw": round(combined, 4),
                "matched_keywords": kw_result.get("matched", []),
                "missing_keywords": kw_result.get("missing", []),
                "sentence_count": self._count_sentences(student_answer),
                "word_count": len(student_answer.split()),
            },
        )
        return result

    def grade_batch(
        self,
        student_answers: Dict[str, str],
        reference_answers: Dict[str, str],
        keywords_map: Dict[str, List[str]] = None,
        marks_map: Dict[str, float] = None,
        question_texts: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Grade multiple long-answer questions."""
        keywords_map = keywords_map or {}
        marks_map = marks_map or {}
        question_texts = question_texts or {}

        per_question = {}
        total_score = 0.0
        total_max = 0.0

        for qid, student_ans in student_answers.items():
            ref_ans = reference_answers.get(qid, "")
            kws = keywords_map.get(qid, [])
            mks = marks_map.get(qid, 10.0)
            qtext = question_texts.get(qid, "")

            result = self.grade(
                student_answer=student_ans,
                reference_answer=ref_ans,
                keywords=kws,
                marks=mks,
                question_text=qtext,
            )
            per_question[qid] = {
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

    # ─── Private structure/coverage helpers ──────────────────────────────────

    def _score_structure(self, answer: str, reference: str) -> float:
        """Evaluate answer structure quality.

        Measures:
        - Sentence count (should be proportional to reference)
        - Sentence length variance (diversity of expression)
        - Paragraph presence (if answer is long enough)
        """
        if not answer.strip():
            return 0.0

        student_sents = self._count_sentences(answer)
        ref_sents = self._count_sentences(reference)
        if ref_sents == 0:
            ref_sents = 1

        # Sentence count similarity (log-scaled)
        ratio = student_sents / ref_sents
        if ratio <= 0:
            sent_score = 0.0
        elif ratio < 0.5:
            sent_score = ratio * 2  # 0 → 1.0 as ratio goes 0 → 0.5
        elif ratio <= 2.0:
            sent_score = 1.0  # Full marks for proportional length
        else:
            sent_score = max(0.0, 1.0 - (ratio - 2.0) * 0.3)

        # Sentence length variance (measure of expression diversity)
        sentences = self._split_sentences(answer)
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            mean_len = sum(lengths) / len(lengths)
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            # Moderate variance is good (not all same-length sentences)
            variance_score = min(1.0, variance / (mean_len ** 2 + 1))
        else:
            variance_score = 0.5  # Single sentence — partial credit

        return 0.6 * sent_score + 0.4 * variance_score

    def _score_coverage(self, answer: str, reference: str, keywords: List[str]) -> float:
        """Evaluate breadth of concept coverage beyond keyword matching.

        Compares unique content words in student vs reference answers.
        """
        if not answer.strip() or not reference.strip():
            return 0.0

        student_tokens = set(self._tokenize(answer))
        ref_tokens = set(self._tokenize(reference))

        if not ref_tokens:
            return 0.0

        # Jaccard overlap of content words
        intersection = student_tokens & ref_tokens
        coverage = len(intersection) / len(ref_tokens)

        # Penalty for very short answers
        if len(student_tokens) < 5:
            coverage *= 0.5

        return min(1.0, coverage)

    def _count_sentences(self, text: str) -> int:
        """Count sentences in a text."""
        if not text.strip():
            return 0
        sentences = self._split_sentences(text)
        return len(sentences)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'\b(e\.g|i\.e|etc|vs|Dr|Mr|Mrs)\.', r'\1<DOT>', text)
        parts = re.split(r'[.!?]+', text)
        sentences = [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]
        return sentences

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize: lowercase, remove punctuation, split."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 2]  # Skip short tokens


class DescriptiveGrader:
    """High-level interface for grading all answer types including descriptive.

    This is the unified grading interface that combines:
    - MCQ (exact match)
    - Numerical (tolerance-based)
    - Short answer (TF-IDF + keywords)
    - Long/descriptive answer (multi-signal: semantic + keyword + structure + coverage)
    """

    def __init__(self):
        self.mcq_grader = None  # Lazy import to avoid circular
        self.numerical_grader = None
        self.short_answer_grader = None
        self.long_answer_grader = LongAnswerGrader()

    def grade_question(
        self,
        question: Any,  # Question model
        student_answer: Any,
        reference_answer: str = None,
    ) -> Dict[str, Any]:
        """Grade a single question based on its type."""
        from src.grading.mcq import MCQGrader
        from src.grading.numerical import NumericalGrader
        from src.grading.tfidf import TFIDFScorer, KeywordCoverageScorer

        qtype = question.type.value if hasattr(question.type, 'value') else str(question.type)
        qtype_lower = qtype.lower()
        max_marks = question.marks

        if qtype_lower in ("mcq", "true_false"):
            if self.mcq_grader is None:
                self.mcq_grader = MCQGrader()
            result = self.mcq_grader.grade(question, student_answer)
            return {
                "type": "mcq",
                "score": result.get("score", 0.0),
                "max_score": max_marks,
                "correct": result.get("correct", False),
                "method": "exact_match",
            }

        elif qtype_lower == "numerical":
            if self.numerical_grader is None:
                self.numerical_grader = NumericalGrader()
            result = self.numerical_grader.grade(question, student_answer)
            return {
                "type": "numerical",
                "score": result.get("score", 0.0),
                "max_score": max_marks,
                "correct": result.get("correct", False),
                "method": "tolerance_match",
                "error": result.get("error"),
            }

        elif qtype_lower == "short_answer":
            ref = reference_answer or question.text or ""
            scorer = TFIDFScorer()
            scorer.fit([ref])
            cosine = scorer.cosine_similarity(str(student_answer), ref)
            kw_scorer = KeywordCoverageScorer()
            kw = kw_scorer.score(str(student_answer), question.keywords or [])
            combined = 0.6 * cosine + 0.4 * kw.get("coverage", 0.0)
            score = combined * max_marks
            return {
                "type": "short_answer",
                "score": round(score, 4),
                "max_score": max_marks,
                "correct": combined >= 0.8,
                "method": "tfidf_keyword",
                "cosine_similarity": round(cosine, 4),
                "keyword_coverage": round(kw.get("coverage", 0.0), 4),
            }

        elif qtype_lower in ("long_answer", "descriptive", "essay"):
            ref = reference_answer or question.text or ""
            result = self.long_answer_grader.grade(
                student_answer=str(student_answer),
                reference_answer=ref,
                keywords=question.keywords or [],
                marks=max_marks,
                question_text=question.text,
            )
            return {
                "type": "long_answer",
                "score": round(result.final_score, 4),
                "max_score": result.max_score,
                "semantic_similarity": round(result.semantic_score, 4),
                "keyword_coverage": round(result.keyword_score, 4),
                "structure_score": round(result.structure_score, 4),
                "coverage_score": round(result.coverage_score, 4),
                "details": result.details,
            }

        else:
            # Default: treat as short answer
            ref = reference_answer or question.text or ""
            scorer = TFIDFScorer()
            scorer.fit([ref])
            cosine = scorer.cosine_similarity(str(student_answer), ref)
            score = cosine * max_marks
            return {
                "type": qtype_lower,
                "score": round(score, 4),
                "max_score": max_marks,
                "correct": cosine >= 0.8,
                "method": "tfidf_fallback",
            }
