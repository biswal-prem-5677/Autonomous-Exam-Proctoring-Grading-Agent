"""Context-aware risk engine — adjusts risk based on exam context.

The same behavior means different things depending on context:

- Looking away during reading → normal
- Looking away during multiple-face + audio → significant
- Fast typing during essay → normal
- Fast typing during MCQ → irrelevant
- Mouse movement during interactive graph → normal
- Mouse movement with no expected interaction → interesting

This module provides exam policy, question context, timing awareness,
and imports the personal behavioral baseline for context-aware risk.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum


class QuestionType(Enum):
    MCQ = "mcq"
    NUMERICAL = "numerical"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    TRUE_FALSE = "true_false"


@dataclass
class ExamPolicy:
    """Configurable exam security policy.

    Defines what behaviors are relevant for each exam type.
    Different exams need different rules.
    """
    camera_required: bool = True
    microphone_required: bool = True
    keyboard_required: bool = True
    mouse_required: bool = True

    allow_external_sites: bool = False
    allow_calculator: bool = False
    allow_ide: bool = False
    allow_documentation: bool = False

    multiple_face_enabled: bool = True
    audio_enabled: bool = True
    browser_events_enabled: bool = True
    temporal_fusion_enabled: bool = True
    keystroke_enabled: bool = True
    mouse_enabled: bool = True

    weight_vision: float = 1.0
    weight_audio: float = 0.8
    weight_keyboard: float = 0.7
    weight_mouse: float = 0.6
    weight_browser: float = 1.0
    weight_interaction: float = 0.9

    escalation_threshold: float = 0.7
    review_threshold: float = 0.5
    multi_signal_required: int = 2

    def as_dict(self) -> Dict:
        return {
            "camera_required": self.camera_required,
            "microphone_required": self.microphone_required,
            "keyboard_required": self.keyboard_required,
            "mouse_required": self.mouse_required,
            "allow_external_sites": self.allow_external_sites,
            "allow_calculator": self.allow_calculator,
            "allow_ide": self.allow_ide,
            "allow_documentation": self.allow_documentation,
            "multiple_face_enabled": self.multiple_face_enabled,
            "audio_enabled": self.audio_enabled,
            "browser_events_enabled": self.browser_events_enabled,
            "temporal_fusion_enabled": self.temporal_fusion_enabled,
            "keystroke_enabled": self.keystroke_enabled,
            "mouse_enabled": self.mouse_enabled,
            "weight_vision": self.weight_vision,
            "weight_audio": self.weight_audio,
            "weight_keyboard": self.weight_keyboard,
            "weight_mouse": self.weight_mouse,
            "weight_browser": self.weight_browser,
            "weight_interaction": self.weight_interaction,
            "escalation_threshold": self.escalation_threshold,
            "review_threshold": self.review_threshold,
            "multi_signal_required": self.multi_signal_required,
        }


@dataclass
class QuestionContext:
    """Context about the current question being answered."""
    question_id: str
    question_type: str
    question_text: str
    expected_interaction: str  # "typing", "selecting", "navigating", "reading"
    marks: float = 1.0
    time_elapsed_on_question: float = 0.0
    expects_typing: bool = False
    expects_mouse: bool = False
    expects_navigation: bool = False


@dataclass
class ExamContext:
    """Full exam context for risk adjustment."""
    policy: ExamPolicy
    current_question: Optional[QuestionContext] = None
    exam_duration_seconds: float = 3600.0
    exam_elapsed_seconds: float = 0.0

    def exam_progress(self) -> float:
        if self.exam_duration_seconds <= 0:
            return 0.0
        return min(self.exam_elapsed_seconds / self.exam_duration_seconds, 1.0)

    def time_remaining(self) -> float:
        return max(0.0, self.exam_duration_seconds - self.exam_elapsed_seconds)


# Import the full baseline implementation from src.baseline
from src.baseline import (  # noqa: E402
    StudentBaseline as _StudentBaseline,
    BehavioralSample,
    BaselineManager,
)


class StudentBaseline(_StudentBaseline):
    """Extended student baseline (delegates to src.baseline implementation)."""
    pass


def get_default_policy(exam_type: str = "standard") -> ExamPolicy:
    """Return a default exam policy for a given exam type."""
    policies = {
        "standard": ExamPolicy(),
        "programming": ExamPolicy(
            allow_ide=True, allow_documentation=True, weight_mouse=0.8,
        ),
        "mathematics": ExamPolicy(allow_calculator=True, weight_keyboard=0.9),
        "open_book": ExamPolicy(
            allow_external_sites=True, allow_documentation=True,
            weight_browser=0.3,
        ),
        "essay": ExamPolicy(weight_keyboard=0.95, weight_interaction=0.95),
    }
    return policies.get(exam_type, policies["standard"])


def compute_context_weight(
    event_type: str,
    context: Optional[ExamContext] = None,
    modality: str = "vision",
) -> float:
    """Compute context-adjusted weight for an event."""
    if context is None or context.current_question is None:
        return 1.0

    q = context.current_question
    policy = context.policy

    if event_type in ("rapid_typing", "idle_keyboard"):
        return 0.3 if q.expects_typing else 0.9

    if event_type in ("rapid_movement", "idle_mouse"):
        return 0.3 if q.expects_mouse else 0.9

    if event_type == "tab_switch":
        return 0.2 if policy.allow_external_sites else 1.0

    modality_weight_map = {
        "vision": policy.weight_vision,
        "audio": policy.weight_audio,
        "keyboard": policy.weight_keyboard,
        "mouse": policy.weight_mouse,
        "browser": policy.weight_browser,
    }
    return modality_weight_map.get(modality, 1.0)
