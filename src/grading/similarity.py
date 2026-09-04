"""Answer similarity detection — checks for answer copying between students."""

import re
from typing import Dict, List, Any
import numpy as np

from src.grading.tfidf import TFIDFScorer


class AnswerSimilarityDetector:
    """Detects unusually similar answers between two students' submissions."""

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self._tfidf = TFIDFScorer()

    def compare_answers(self, answer1: str, answer2: str) -> Dict[str, Any]:
        """Compare two text answers and return similarity metrics."""
        if not answer1 or not answer2:
            return {"similarity": 0.0, "suspicious": False, "reason": "Empty answer(s)"}

        # Clean answers
        a1 = self._clean(answer1)
        a2 = self._clean(answer2)

        cosine_sim = self._tfidf.cosine_similarity(a1, a2)
        jaccard_sim = self._tfidf.jaccard_similarity(a1, a2)

        # Length ratio (very different lengths = less suspicious)
        len_ratio = min(len(a1), len(a2)) / max(len(a1), len(a2), 1)

        # Combined score
        combined = 0.7 * cosine_sim + 0.3 * jaccard_sim

        # Adjust for length ratio
        if len_ratio < 0.5:
            combined *= len_ratio * 2

        suspicious = combined >= self.similarity_threshold

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "jaccard_similarity": round(jaccard_sim, 4),
            "length_ratio": round(len_ratio, 4),
            "combined_similarity": round(combined, 4),
            "suspicious": suspicious,
            "threshold": self.similarity_threshold,
        }

    def compare_batch(self, answers_a: Dict[str, str],
                      answers_b: Dict[str, str]) -> Dict[str, Any]:
        """Compare all answers between two students."""
        results = {}
        suspicious_count = 0
        question_ids = set(answers_a.keys()) | set(answers_b.keys())

        for qid in question_ids:
            a = answers_a.get(qid, "")
            b = answers_b.get(qid, "")
            if isinstance(a, dict):
                a = a.get("answer", "")
            if isinstance(b, dict):
                b = b.get("answer", "")
            result = self.compare_answers(str(a), str(b))
            results[qid] = result
            if result["suspicious"]:
                suspicious_count += 1

        total = len(question_ids)
        return {
            "per_question": results,
            "suspicious_count": suspicious_count,
            "total_questions": total,
            "suspicious_ratio": suspicious_count / total if total > 0 else 0.0,
            "flagged": suspicious_count > 0,
        }

    def _clean(self, text: str) -> str:
        """Basic text cleaning."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
