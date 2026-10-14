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
    MATHEMATICAL = "mathematical"
    PROGRAMMING = "programming"


class AgentState(Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    TECHNICAL_EVENT = "TECHNICAL_EVENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MONITOR = "MONITOR"


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
    rubric: Optional[Dict[str, Any]] = None               # Optional rubric for grading
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
            "rubric": self.rubric,
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
    exam_id: str = ""
    title: str = "Untitled Exam"
    duration_minutes: int = 60
    questions: List[Question] = field(default_factory=list)
    topic: str = ""
    difficulty: str = "medium"
    instructions: str = ""
    passing_score: float = 40.0
    is_released: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __init__(
        self,
        exam_id: str = "",
        title: str = "Untitled Exam",
        duration_minutes: int = 60,
        questions: Optional[List[Question]] = None,
        topic: str = "",
        difficulty: str = "medium",
        instructions: str = "",
        passing_score: float = 40.0,
        is_released: bool = False,
        settings: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        **kwargs
    ):
        self.exam_id = exam_id or kwargs.get("id", "")
        self.title = title
        self.duration_minutes = duration_minutes
        self.questions = questions if questions is not None else []
        self.topic = topic
        self.difficulty = difficulty
        self.instructions = instructions
        self.passing_score = float(passing_score)
        self.is_released = is_released
        self.settings = settings if settings is not None else {}
        self.created_at = created_at if created_at is not None else datetime.now()

    @property
    def id(self) -> str:
        """Alias for exam_id for compatibility."""
        return self.exam_id

    def to_dict(self) -> dict:
        return {
            "exam_id": self.exam_id,
            "id": self.exam_id,
            "title": self.title,
            "duration_minutes": self.duration_minutes,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "instructions": self.instructions,
            "passing_score": self.passing_score,
            "is_released": self.is_released,
            "settings": self.settings,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, "isoformat") else str(self.created_at),
            "questions": [q.to_dict() for q in self.questions],
        }

    def add_question(self, qtype: str, text: str, **kwargs):
        """Add a question to the exam."""
        # Normalize type string: handle mixed-case inputs from frontend
        qtype_normalized = qtype.strip().lower().replace(" ", "_")
        meta = dict(kwargs.pop("metadata", {}) or {})

        options = kwargs.pop("options", None)
        correct_answer = kwargs.pop("correct_answer", None)
        marks = float(kwargs.pop("marks", 1.0))
        keywords = kwargs.pop("keywords", []) or []
        tolerance = kwargs.pop("tolerance", None)
        rubric = kwargs.pop("rubric", None)
        qid = kwargs.pop("id", kwargs.pop("question_id", f"q{len(self.questions) + 1}"))

        # Map programming fields
        tests = kwargs.pop("tests", kwargs.pop("test_cases", None))
        if tests is not None:
            meta["tests"] = tests
        hidden_tests = kwargs.pop("hidden_tests", None)
        if hidden_tests is not None:
            meta["hidden_tests"] = hidden_tests
        edge_cases = kwargs.pop("edge_cases", None)
        if edge_cases is not None:
            meta["edge_cases"] = edge_cases
        func_name = kwargs.pop("function_name", kwargs.pop("target_function", kwargs.pop("entry_function_name", None)))
        if func_name is not None:
            meta["function_name"] = func_name
        language = kwargs.pop("language", None)
        if language is not None:
            meta["language"] = language
        starter_code = kwargs.pop("starter_code", None)
        if starter_code is not None:
            meta["starter_code"] = starter_code

        # Map mathematical fields
        formula = kwargs.pop("formula", kwargs.pop("formula_hint", kwargs.pop("governing_formula", None)))
        if formula is not None:
            meta["governing_formula"] = formula
            meta["formula"] = formula
        variables = kwargs.pop("variables", kwargs.pop("expected_variables", kwargs.pop("required_params", None)))
        if variables is not None:
            meta["variables"] = variables
            meta["expected_variables"] = variables
            meta["required_params"] = variables
        units = kwargs.pop("units", kwargs.pop("expected_units", None))
        if units is not None:
            meta["units"] = units
        final_value = kwargs.pop("final_value", kwargs.pop("expected_final_value", None))
        if final_value is not None:
            meta["final_value"] = final_value

        # Map model / reference answer
        model_answer = kwargs.pop("model_answer", kwargs.pop("reference_answer", None))
        if model_answer is not None:
            meta["model_answer"] = model_answer
            meta["reference_answer"] = model_answer
            if correct_answer is None:
                correct_answer = model_answer

        # Put any remaining kwargs into meta
        for k, v in kwargs.items():
            meta[k] = v

        q = Question(
            id=str(qid),
            type=QuestionType(qtype_normalized),
            text=text,
            options=options,
            correct_answer=correct_answer,
            marks=marks,
            keywords=keywords,
            tolerance=tolerance,
            rubric=rubric,
            metadata=meta,
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
