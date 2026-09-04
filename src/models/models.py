"""Data models for questions, sessions, and exam events."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class QuestionType(Enum):
    MCQ = "mcq"
    NUMERICAL = "numerical"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    TRUE_FALSE = "true_false"


class AgentState(Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass
class Question:
    """Represents a single exam question."""
    id: str
    type: QuestionType
    text: str
    options: Optional[List[str]] = None          # For MCQ
    correct_answer: Any = None                   # Answer key
    marks: float = 1.0
    keywords: List[str] = field(default_factory=list)  # For short answer
    tolerance: Optional[float] = None                      # For numerical (None = use grader default)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "text": self.text,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "marks": self.marks,
            "keywords": self.keywords,
            "tolerance": self.tolerance,
            "metadata": self.metadata,
        }


@dataclass
class ProctoringEvent:
    """A single proctoring event (face absent, head turned, etc.)."""
    timestamp: datetime
    event_type: str
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Exam:
    """Represents an exam with questions."""
    exam_id: str
    title: str
    duration_minutes: int
    questions: List[Question] = field(default_factory=list)
    topic: str = ""
    difficulty: str = "medium"
    created_at: datetime = field(default_factory=datetime.now)

    def add_question(self, qtype: str, text: str, **kwargs):
        """Add a question to the exam."""
        # Extract fields that belong on Exam, not Question
        meta = kwargs.pop("metadata", {})
        for key in ("topic", "difficulty"):
            if key in kwargs:
                meta[key] = kwargs.pop(key)
        if kwargs.get("reference_answer"):
            meta["reference_answer"] = kwargs.pop("reference_answer")
        if meta:
            kwargs["metadata"] = meta
        q = Question(
            id=f"q{len(self.questions) + 1}",
            type=QuestionType(qtype),
            text=text,
            **kwargs
        )
        self.questions.append(q)
        return q


@dataclass
class ExamSession:
    """Represents an active exam session."""
    session_id: str
    student_id: str
    exam_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_minutes: int = 60
    state: AgentState = AgentState.NORMAL
    risk_score: float = 0.0
    events: List[ProctoringEvent] = field(default_factory=list)
    answers: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
