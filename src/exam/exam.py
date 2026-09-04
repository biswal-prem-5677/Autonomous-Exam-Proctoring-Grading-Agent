"""Exam controller — orchestrates exam flow and proctoring integration."""

import uuid
from datetime import datetime
from typing import Optional, List, Any
from src.models.models import ExamSession, AgentState, ProctoringEvent, Question
from src.exam.session import ExamSessionManager
from src.exam.questions import QuestionBank
from src.agent.risk import RiskEngine
from src.agent.memory import EvidenceMemory


class ExamController:
    """Main controller for an exam session, integrating all subsystems."""

    def __init__(self, config: dict):
        self.config = config
        self.session_id = str(uuid.uuid4())[:8]
        self.question_bank = QuestionBank()
        self.risk_engine = RiskEngine(config.get("risk", {}))
        self.evidence_memory = EvidenceMemory(
            window_seconds=config.get("risk", {}).get("evidence_window_seconds", 300)
        )
        self.session_manager: Optional[ExamSessionManager] = None
        self.session: Optional[ExamSession] = None

    def create_session(self, student_id: str, exam_id: str,
                       duration_minutes: int = 60) -> ExamSession:
        """Create and initialize a new exam session."""
        self.session = ExamSession(
            session_id=self.session_id,
            student_id=student_id,
            exam_id=exam_id,
            duration_minutes=duration_minutes,
        )
        self.session_manager = ExamSessionManager(
            session=self.session,
            on_warning=self._on_time_warning,
            on_expire=self._on_session_expire,
        )
        return self.session

    def start_exam(self) -> None:
        """Begin the exam session."""
        if self.session_manager:
            self.session_manager.start()

    def record_proctoring_event(self, event_type: str, confidence: float,
                                details: dict = None) -> None:
        """Record a proctoring event and update risk score."""
        event = ProctoringEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            confidence=confidence,
            details=details or {},
        )

        if self.session_manager:
            self.session_manager.add_event(event)

        self.evidence_memory.add_event(event)

        # Update risk score
        risk_score = self.risk_engine.compute_risk(
            self.evidence_memory.get_recent_events()
        )
        self.session.risk_score = risk_score

        # Update agent state based on risk
        new_state = self._determine_state(risk_score)
        self.session.state = new_state
        if self.session_manager:
            self.session_manager.set_state(new_state)

    def _determine_state(self, risk_score: float) -> AgentState:
        """Map risk score to agent state."""
        thresholds = self.config.get("risk", {})
        if risk_score >= thresholds.get("escalation_threshold", 0.7):
            return AgentState.HIGH_RISK
        elif risk_score >= thresholds.get("review_threshold", 0.5):
            return AgentState.SUSPICIOUS
        return AgentState.NORMAL

    def submit_answer(self, question_id: str, answer: Any) -> None:
        """Submit an answer for a question."""
        if self.session_manager:
            self.session_manager.submit_answer(question_id, answer)

    def end_exam(self) -> ExamSession:
        """End the exam session."""
        if self.session_manager:
            self.session_manager.stop()
        return self.session

    def _on_time_warning(self, remaining_seconds: int) -> None:
        """Callback for time warning."""
        print(f"[WARNING] {remaining_seconds}s remaining in exam")

    def _on_session_expire(self) -> None:
        """Callback for session expiration."""
        print("[INFO] Exam session expired — auto-submitting")

    def get_session_status(self) -> dict:
        """Get current session status for UI."""
        if not self.session or not self.session_manager:
            return {}
        return {
            "session_id": self.session.session_id,
            "state": self.session.state.value,
            "risk_score": round(self.session.risk_score, 4),
            "elapsed_seconds": self.session_manager.get_elapsed(),
            "remaining_seconds": self.session_manager.get_remaining(),
            "events_count": len(self.session.events),
            "answers_count": len(self.session.answers),
        }


class ExamManager:
    """High-level exam manager for frontend - wraps ExamController for UI convenience."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._controllers: dict[str, ExamController] = {}
        self._exams: dict[str, object] = {}

    def get_exam(self, exam_id: str):
        """Get an exam by ID."""
        return self._exams.get(exam_id)

    def create_exam(self, exam_id: str, title: str, duration_minutes: int,
                    questions: list) -> object:
        """Create a new exam."""
        from src.models.models import Exam
        exam = Exam(
            exam_id=exam_id,
            title=title,
            duration_minutes=duration_minutes,
            questions=questions,
        )
        self._exams[exam_id] = exam
        return exam

    def get_or_create_controller(self, student_id: str, exam_id: str,
                                  duration_minutes: int = 60) -> ExamController:
        """Get or create an exam controller for a student."""
        key = f"{student_id}:{exam_id}"
        if key not in self._controllers:
            self._controllers[key] = ExamController(self.config)
            self._controllers[key].create_session(student_id, exam_id, duration_minutes)
        return self._controllers[key]

    def get_controller(self, student_id: str, exam_id: str) -> Optional[ExamController]:
        """Get existing controller for a student."""
        key = f"{student_id}:{exam_id}"
        return self._controllers.get(key)
