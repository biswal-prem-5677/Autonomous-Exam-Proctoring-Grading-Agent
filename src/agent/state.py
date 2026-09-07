"""Agent state machine — manages state transitions with validation."""

from typing import Optional
from src.models.models import AgentState


class AgentStateMachine:
    """State machine for the proctoring agent with valid transitions.

    Valid transitions:
    NORMAL → MONITOR → SUSPICIOUS → HIGH_RISK → REVIEW_REQUIRED
    NORMAL → TECHNICAL_EVENT → NORMAL (technical issue resolved)
    SUSPICIOUS → INSUFFICIENT_EVIDENCE → NORMAL
    Any → NORMAL (reset)
    """

    VALID_TRANSITIONS = {
        AgentState.NORMAL: [AgentState.MONITOR, AgentState.TECHNICAL_EVENT,
                            AgentState.SUSPICIOUS],
        AgentState.MONITOR: [AgentState.NORMAL, AgentState.SUSPICIOUS],
        AgentState.SUSPICIOUS: [AgentState.NORMAL, AgentState.HIGH_RISK,
                               AgentState.INSUFFICIENT_EVIDENCE],
        AgentState.HIGH_RISK: [AgentState.REVIEW_REQUIRED, AgentState.SUSPICIOUS],
        AgentState.REVIEW_REQUIRED: [AgentState.NORMAL, AgentState.SUSPICIOUS],
        AgentState.TECHNICAL_EVENT: [AgentState.NORMAL],
        AgentState.INSUFFICIENT_EVIDENCE: [AgentState.NORMAL, AgentState.MONITOR],
    }

    def __init__(self, initial_state: AgentState = AgentState.NORMAL):
        self._state = initial_state
        self._transition_history = []
        self._state_duration = 0.0

    @property
    def state(self) -> AgentState:
        return self._state

    def can_transition(self, new_state: AgentState) -> bool:
        """Check if a state transition is valid."""
        return new_state in self.VALID_TRANSITIONS.get(self._state, [])

    def transition(self, new_state: AgentState) -> bool:
        """Attempt to transition to a new state.

        Returns True if transition was valid and executed.
        """
        if self.can_transition(new_state):
            old = self._state
            self._state = new_state
            self._transition_history.append({
                "from": old.value,
                "to": new_state.value,
            })
            self._state_duration = 0.0
            return True
        return False

    def force_state(self, new_state: AgentState) -> None:
        """Force state change regardless of valid transitions (for resets)."""
        self._state = new_state
        self._state_duration = 0.0

    def tick(self, seconds: float = 1.0) -> None:
        """Advance state duration timer."""
        self._state_duration += seconds

    @property
    def duration(self) -> float:
        """Time spent in current state (seconds)."""
        return self._state_duration

    def get_transition_count(self) -> int:
        """Get total number of state transitions."""
        return len(self._transition_history)

    def reset(self) -> None:
        """Reset to NORMAL state."""
        self._state = AgentState.NORMAL
        self._transition_history.clear()
        self._state_duration = 0.0
