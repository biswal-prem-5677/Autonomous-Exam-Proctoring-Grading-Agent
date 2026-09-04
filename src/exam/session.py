"""Exam session management — timer, state tracking, answer collection."""

import time
import threading
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from src.models.models import ExamSession, AgentState, ProctoringEvent


class ExamSessionManager:
    """Manages an active exam session with timer and state tracking."""

    def __init__(self, session: ExamSession, on_warning: Optional[Callable] = None,
                 on_expire: Optional[Callable] = None):
        self.session = session
        self.on_warning = on_warning
        self.on_expire = on_expire
        self._timer_thread: Optional[threading.Thread] = None
        self._running = False
        self._warning_sent = False

    def start(self) -> None:
        """Start the exam session and timer."""
        self.session.start_time = datetime.now()
        self._running = True
        self._start_timer()

    def _start_timer(self) -> None:
        """Start the countdown timer in a background thread."""
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()

    def _timer_loop(self) -> None:
        """Countdown loop that checks for warnings and expiration."""
        duration = timedelta(minutes=self.session.duration_minutes)

        while self._running:
            elapsed = datetime.now() - self.session.start_time
            remaining = duration - elapsed

            if remaining.total_seconds() <= 0:
                self._expire()
                return

            # Warning at 5 minutes remaining
            if not self._warning_sent and remaining <= timedelta(minutes=5):
                self._warning_sent = True
                if self.on_warning:
                    self.on_warning(int(remaining.total_seconds()))

            time.sleep(1)

    def _expire(self) -> None:
        """Handle session expiration."""
        self.session.end_time = datetime.now()
        self._running = False
        if self.on_expire:
            self.on_expire()

    def submit_answer(self, question_id: str, answer: Any) -> None:
        """Record a student's answer."""
        self.session.answers[question_id] = {
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        }

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.session.elapsed_seconds

    def get_remaining(self) -> float:
        """Get remaining time in seconds."""
        duration = timedelta(minutes=self.session.duration_minutes)
        remaining = duration - timedelta(seconds=self.session.elapsed_seconds)
        return max(0, remaining.total_seconds())

    def add_event(self, event: ProctoringEvent) -> None:
        """Add a proctoring event to the session."""
        self.session.events.append(event)

    def set_state(self, state: AgentState) -> None:
        """Update the agent state."""
        self.session.state = state

    def stop(self) -> None:
        """Stop the session timer."""
        self._running = False
        self.session.end_time = datetime.now()

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return self.session.elapsed_seconds >= self.session.duration_minutes * 60
