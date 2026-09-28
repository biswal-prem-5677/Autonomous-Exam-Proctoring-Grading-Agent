"""Plagiarism detection via MinHash + Locality-Sensitive Hashing (LSH).

Detects near-duplicate answers across student submissions without any
external APIs — pure mathematical implementation.

Mathematical foundation:
    1. k-gram shingling: Convert text into overlapping character n-grams
       S(text) = {text[i:i+k] for i in range(len(text)-k+1)}

    2. MinHash signatures: Approximate Jaccard similarity using hash functions
       J(A,B) = |A ∩ B| / |A ∪ B|
       Pr[min(h(A)) = min(h(B))] = J(A,B)

       We use n_hash independent hash functions to build a signature vector.
       Signature comparison gives an unbiased estimate of Jaccard similarity.

    3. LSH banding: Divide signature into b bands of r rows each.
       Two documents are candidate pairs if they agree in ALL rows of ANY band.
       Probability of becoming a candidate: P(s) = 1 - (1 - s^r)^b
       where s = true Jaccard similarity.

    4. Pairwise similarity matrix: For flagged pairs, compute exact
       Jaccard similarity for confirmation.

References:
    - Broder (1997) "On the resemblance and containment of documents"
    - Leskovec, Rajaraman, Ullman — "Mining of Massive Datasets" Ch. 3
"""

import hashlib
import re
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any


@dataclass
class PlagiarismResult:
    """Result of plagiarism detection for a pair of submissions."""
    student_a: str
    student_b: str
    jaccard_similarity: float
    minhash_similarity: float
    shared_shingles_count: int
    total_shingles_a: int
    total_shingles_b: int
    flagged: bool
    question_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlagiarismReport:
    """Full plagiarism report for an exam."""
    exam_id: str
    total_submissions: int
    total_pairs_checked: int
    flagged_pairs: List[PlagiarismResult]
    similarity_matrix: Dict[str, Dict[str, float]]
    summary: Dict[str, Any] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _create_shingles(text: str, k: int = 5) -> Set[str]:
    """Create k-gram character shingles from text.

    A k-shingle (or k-gram) is a contiguous subsequence of k characters.
    Example: "abcde" with k=3 → {"abc", "bcd", "cde"}

    Args:
        text: Normalized text to shingle
        k: Shingle size (default 5 for character-level)

    Returns:
        Set of k-character shingles
    """
    text = _normalize_text(text)
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _create_word_shingles(text: str, k: int = 3) -> Set[str]:
    """Create k-gram word shingles from text.

    Word-level shingles are better for longer text where paraphrasing
    might change individual characters but preserve word sequences.

    Args:
        text: Text to shingle
        k: Number of consecutive words per shingle

    Returns:
        Set of word k-gram shingles
    """
    text = _normalize_text(text)
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _hash_shingle(shingle: str, seed: int) -> int:
    """Hash a shingle with a given seed for MinHash.

    Uses SHA-256 truncated to 32-bit integer for each (shingle, seed) pair.
    Different seeds produce independent hash functions.
    """
    payload = f"{seed}|{shingle}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:8], 16)  # First 32 bits


class MinHasher:
    """Compute MinHash signatures for sets.

    A MinHash signature is a compact representation of a set that allows
    fast Jaccard similarity estimation.

    For n_hash hash functions, the expected error of Jaccard estimation is:
        error ≈ 1 / sqrt(n_hash)

    With n_hash=128: error ≈ 0.088 (8.8%)
    With n_hash=200: error ≈ 0.071 (7.1%)
    """

    def __init__(self, n_hash: int = 128):
        """Initialize with number of hash functions.

        Args:
            n_hash: Number of independent hash functions. More = more accurate
                    but slower. 128 gives ~8.8% error, 200 gives ~7.1%.
        """
        self.n_hash = n_hash

    def compute_signature(self, shingle_set: Set[str]) -> List[int]:
        """Compute the MinHash signature for a set of shingles.

        For each of the n_hash hash functions, we compute the minimum hash
        value across all shingles. This gives us our signature vector.

        The probability that two signatures agree at any position equals
        the true Jaccard similarity of the underlying sets.
        """
        if not shingle_set:
            return [0] * self.n_hash

        signature = []
        for seed in range(self.n_hash):
            min_hash = min(_hash_shingle(s, seed) for s in shingle_set)
            signature.append(min_hash)

        return signature

    def estimate_similarity(self, sig_a: List[int], sig_b: List[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures.

        Similarity = (number of agreeing positions) / n_hash
        """
        if not sig_a or not sig_b:
            return 0.0
        if len(sig_a) != len(sig_b):
            raise ValueError("Signatures must have the same length")

        agreements = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return agreements / len(sig_a)


class LSH:
    """Locality-Sensitive Hashing for fast candidate pair detection.

    Divides MinHash signatures into bands. Two documents become a candidate
    pair if they are identical in at least one complete band.

    The S-curve probability of two documents with similarity s becoming
    a candidate pair is:
        P(s) = 1 - (1 - s^r)^b

    where b = number of bands, r = rows per band.

    Example with b=20, r=5 (100 hashes total):
        s=0.8: P ≈ 0.9996 (99.96% chance of detection)
        s=0.5: P ≈ 0.4700 (47% chance)
        s=0.2: P ≈ 0.0064 (0.64% chance — correctly ignored)
    """

    def __init__(self, n_bands: int = 20, rows_per_band: int = 5):
        """Initialize LSH with banding parameters.

        Args:
            n_bands: Number of bands to divide the signature into
            rows_per_band: Number of rows per band
        """
        self.n_bands = n_bands
        self.rows_per_band = rows_per_band
        self._buckets: List[Dict[int, List[str]]] = [
            defaultdict(list) for _ in range(n_bands)
        ]

    @property
    def signature_length(self) -> int:
        """Required MinHash signature length."""
        return self.n_bands * self.rows_per_band

    def threshold_probability(self, similarity: float) -> float:
        """Compute the probability that two documents with given similarity
        are identified as a candidate pair.

        P(s) = 1 - (1 - s^r)^b
        """
        r = self.rows_per_band
        b = self.n_bands
        return 1.0 - (1.0 - similarity ** r) ** b

    def index(self, doc_id: str, signature: List[int]) -> None:
        """Index a document's MinHash signature for LSH lookup.

        The signature is divided into bands, and each band is hashed
        to a bucket. Documents sharing a bucket in any band are candidates.
        """
        if len(signature) < self.signature_length:
            signature = signature + [0] * (self.signature_length - len(signature))

        for band_idx in range(self.n_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            bucket_key = hash(band)
            self._buckets[band_idx][bucket_key].append(doc_id)

    def find_candidates(self) -> Set[Tuple[str, str]]:
        """Find all candidate pairs that share at least one LSH bucket.

        Returns:
            Set of (doc_id_a, doc_id_b) pairs
        """
        candidates = set()
        for band_buckets in self._buckets:
            for bucket_docs in band_buckets.values():
                if len(bucket_docs) > 1:
                    for i in range(len(bucket_docs)):
                        for j in range(i + 1, len(bucket_docs)):
                            pair = tuple(sorted([bucket_docs[i], bucket_docs[j]]))
                            candidates.add(pair)
        return candidates

    def clear(self) -> None:
        """Clear all indexed documents."""
        self._buckets = [defaultdict(list) for _ in range(self.n_bands)]


def compute_exact_jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute exact Jaccard similarity between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class PlagiarismDetector:
    """Full plagiarism detection pipeline.

    Usage:
        detector = PlagiarismDetector(threshold=0.6)

        # Add student submissions
        detector.add_submission("student_001", "q1", "My answer about photosynthesis...")
        detector.add_submission("student_002", "q1", "My answer about photosynthesis...")
        detector.add_submission("student_003", "q1", "A completely different answer...")

        # Run detection
        report = detector.detect("exam_001")
        print(report.flagged_pairs)
    """

    def __init__(
        self,
        threshold: float = 0.6,
        shingle_size: int = 5,
        use_word_shingles: bool = True,
        word_shingle_size: int = 3,
        n_hash: int = 128,
        n_bands: int = 20,
        rows_per_band: int = 5,
    ):
        """Initialize the plagiarism detector.

        Args:
            threshold: Jaccard similarity threshold for flagging (0.0-1.0)
            shingle_size: Character k-gram size
            use_word_shingles: Use word-level shingles (better for essays)
            word_shingle_size: Word k-gram size
            n_hash: Number of MinHash functions
            n_bands: Number of LSH bands
            rows_per_band: Rows per LSH band
        """
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.use_word_shingles = use_word_shingles
        self.word_shingle_size = word_shingle_size

        self.hasher = MinHasher(n_hash=n_hash)
        self.lsh = LSH(n_bands=n_bands, rows_per_band=rows_per_band)

        # Storage: {question_id: {student_id: (shingles, signature)}}
        self._submissions: Dict[str, Dict[str, Tuple[Set[str], List[int]]]] = defaultdict(dict)

    def add_submission(self, student_id: str, question_id: str, answer_text: str) -> None:
        """Add a student's answer for plagiarism analysis.

        Args:
            student_id: Unique student identifier
            question_id: Question identifier
            answer_text: The student's answer text
        """
        if not answer_text or not answer_text.strip():
            return

        # Create shingles
        if self.use_word_shingles:
            shingles = _create_word_shingles(answer_text, k=self.word_shingle_size)
        else:
            shingles = _create_shingles(answer_text, k=self.shingle_size)

        if not shingles:
            return

        # Compute MinHash signature
        signature = self.hasher.compute_signature(shingles)

        # Store
        self._submissions[question_id][student_id] = (shingles, signature)

    def detect(self, exam_id: str = "") -> PlagiarismReport:
        """Run plagiarism detection across all submitted answers.

        Pipeline:
        1. Build LSH index from all MinHash signatures
        2. Find candidate pairs via LSH
        3. Compute exact Jaccard similarity for candidates
        4. Flag pairs exceeding threshold

        Returns:
            PlagiarismReport with flagged pairs and similarity matrix
        """
        all_flagged: List[PlagiarismResult] = []
        similarity_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        total_pairs = 0

        for question_id, submissions in self._submissions.items():
            if len(submissions) < 2:
                continue

            # Reset LSH for this question
            self.lsh.clear()

            # Index all signatures
            for student_id, (shingles, signature) in submissions.items():
                doc_key = f"{question_id}::{student_id}"
                self.lsh.index(doc_key, signature)

            # Find candidate pairs via LSH
            candidates = self.lsh.find_candidates()

            # Also do brute-force for small submission counts (< 50)
            if len(submissions) < 50:
                student_ids = list(submissions.keys())
                for i in range(len(student_ids)):
                    for j in range(i + 1, len(student_ids)):
                        pair = tuple(sorted([
                            f"{question_id}::{student_ids[i]}",
                            f"{question_id}::{student_ids[j]}",
                        ]))
                        candidates.add(pair)

            # Evaluate candidates
            for doc_a, doc_b in candidates:
                # Parse student IDs
                parts_a = doc_a.split("::")
                parts_b = doc_b.split("::")
                if len(parts_a) != 2 or len(parts_b) != 2:
                    continue

                q_a, s_a = parts_a
                q_b, s_b = parts_b

                if q_a != q_b:
                    continue  # Different questions

                if s_a not in submissions or s_b not in submissions:
                    continue

                shingles_a, sig_a = submissions[s_a]
                shingles_b, sig_b = submissions[s_b]

                # MinHash estimated similarity
                minhash_sim = self.hasher.estimate_similarity(sig_a, sig_b)

                # Exact Jaccard similarity for confirmation
                exact_jaccard = compute_exact_jaccard(shingles_a, shingles_b)
                shared = len(shingles_a & shingles_b)

                total_pairs += 1

                # Store in similarity matrix
                pair_key = f"{s_a}|{s_b}"
                similarity_matrix[s_a][s_b] = exact_jaccard
                similarity_matrix[s_b][s_a] = exact_jaccard

                # Flag if above threshold
                flagged = exact_jaccard >= self.threshold
                result = PlagiarismResult(
                    student_a=s_a,
                    student_b=s_b,
                    jaccard_similarity=round(exact_jaccard, 4),
                    minhash_similarity=round(minhash_sim, 4),
                    shared_shingles_count=shared,
                    total_shingles_a=len(shingles_a),
                    total_shingles_b=len(shingles_b),
                    flagged=flagged,
                    question_id=question_id,
                    details={
                        "threshold": self.threshold,
                        "shingle_type": "word" if self.use_word_shingles else "character",
                        "shingle_size": self.word_shingle_size if self.use_word_shingles else self.shingle_size,
                    },
                )

                if flagged:
                    all_flagged.append(result)

        # Sort flagged pairs by similarity (highest first)
        all_flagged.sort(key=lambda r: r.jaccard_similarity, reverse=True)

        total_submissions = sum(len(subs) for subs in self._submissions.values())

        return PlagiarismReport(
            exam_id=exam_id,
            total_submissions=total_submissions,
            total_pairs_checked=total_pairs,
            flagged_pairs=all_flagged,
            similarity_matrix=dict(similarity_matrix),
            summary={
                "flagged_count": len(all_flagged),
                "threshold": self.threshold,
                "max_similarity": max(
                    (r.jaccard_similarity for r in all_flagged), default=0.0
                ),
                "questions_analyzed": len(self._submissions),
            },
        )

    def clear(self) -> None:
        """Clear all submissions."""
        self._submissions.clear()
        self.lsh.clear()
