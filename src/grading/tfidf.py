"""TF-IDF vectorizer and short-answer grading using keyword matching."""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Any
import numpy as np


class TFIDFScorer:
    """Self-implemented TF-IDF for short-answer evaluation."""

    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._vocabulary: set = set()
        self._corpus_count: int = 0

    def fit(self, documents: List[str]) -> 'TFIDFScorer':
        """Compute IDF from a corpus of reference answers."""
        self._corpus_count = len(documents)
        doc_freq: Counter = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            self._vocabulary.update(tokens)
            doc_freq.update(set(tokens))

        for word in self._vocabulary:
            df = doc_freq.get(word, 0)
            self._idf[word] = math.log(self._corpus_count / (1 + df))

        return self

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, remove punctuation, split."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def _compute_tfidf(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF vector for a document."""
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        tf = Counter(tokens)
        total_terms = len(tokens)

        vector = {}
        for term, count in tf.items():
            idf_val = self._idf.get(term, math.log(self._corpus_count + 1))
            vector[term] = (count / total_terms) * idf_val

        return vector

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using TF-IDF."""
        vec1 = self._compute_tfidf(text1)
        vec2 = self._compute_tfidf(text2)

        if not vec1 or not vec2:
            return 0.0

        all_terms = set(vec1) | set(vec2)
        v1 = np.array([vec1.get(t, 0.0) for t in all_terms])
        v2 = np.array([vec2.get(t, 0.0) for t in all_terms])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between token sets."""
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union


class KeywordCoverageScorer:
    """Evaluates short answers by checking keyword coverage."""

    def __init__(self, partial_threshold: float = 0.6):
        self.partial_threshold = partial_threshold

    def score(self, *args, **kwargs) -> Dict:
        """Score an answer based on keyword coverage.
        
        Supports:
            score(student_answer, keywords)
            score(reference_answer, student_answer, keywords)
        """
        if len(args) >= 3:
            student_answer = args[1]
            keywords = args[2]
        elif len(args) == 2:
            student_answer = args[0]
            keywords = args[1]
        else:
            student_answer = kwargs.get("student_answer", "")
            keywords = kwargs.get("keywords", [])

        keywords = keywords or []
        if not keywords:
            return {
                "coverage": 1.0,
                "matched": [],
                "found": [],
                "missing": [],
                "score": 1.0,
                "total_keywords": 0,
                "matched_count": 0,
            }

        answer_lower = str(student_answer).lower()
        matched = []
        missing = []

        for kw in keywords:
            if str(kw).lower() in answer_lower:
                matched.append(kw)
            else:
                missing.append(kw)

        coverage = len(matched) / len(keywords)

        return {
            "coverage": coverage,
            "matched": matched,
            "found": matched,
            "missing": missing,
            "score": coverage,
            "total_keywords": len(keywords),
            "matched_count": len(matched),
        }
