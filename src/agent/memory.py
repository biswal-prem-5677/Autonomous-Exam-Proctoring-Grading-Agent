"""Evidence memory with temporal sliding window."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class EvidenceEntry:
    """A single evidence entry with timestamp and decayed weight."""
    timestamp: float
    event_type: str
    confidence: float
    decayed_score: float = 0.0

    def age(self, now: float) -> float:
        """Return age in seconds."""
        return now - self.timestamp


class EvidenceMemory:
    """Temporal evidence memory with sliding window and risk decay.

    Stores proctoring events in a sliding time window, applying
    exponential decay: E_t = alpha * E_{t-1} + (1 - alpha) * new_event
    """

    def __init__(self, window_seconds: float = 300.0, alpha: float = 0.8):
        self.window_seconds = window_seconds
        self.alpha = alpha
        self._entries: deque = deque()
        self._cumulative_risk: float = 0.0

    def add_event(self, event_type: str, confidence: float,
                  timestamp: Optional[float] = None) -> None:
        """Add a new proctoring event with decay."""
        now = timestamp or time.time()

        # Apply decay to existing cumulative risk
        self._cumulative_risk *= self.alpha

        # Add new event score
        event_score = (1 - self.alpha) * confidence
        self._cumulative_risk += event_score

        entry = EvidenceEntry(
            timestamp=now,
            event_type=event_type,
            confidence=confidence,
            decayed_score=event_score,
        )
        self._entries.append(entry)
        self._prune(now)

    def add_events_batch(self, events: list) -> None:
        """Add multiple events at once."""
        for evt in events:
            ts = evt.timestamp if hasattr(evt, 'timestamp') else evt.get('timestamp', time.time())
            etype = evt.event_type if hasattr(evt, 'event_type') else evt['event_type']
            conf = evt.confidence if hasattr(evt, 'confidence') else evt['confidence']
            self.add_event(etype, conf, ts.timestamp() if hasattr(ts, 'timestamp') else ts)

    def _prune(self, now: float) -> None:
        """Remove entries outside the sliding window."""
        cutoff = now - self.window_seconds
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()

    def get_recent_events(self, since: Optional[float] = None) -> List[EvidenceEntry]:
        """Get events within the window."""
        now = since or time.time()
        self._prune(now)
        cutoff = now - self.window_seconds
        return [e for e in self._entries if e.timestamp >= cutoff]

    def get_recent_by_type(self, event_type: str,
                           since: Optional[float] = None) -> List[EvidenceEntry]:
        """Get recent events filtered by type."""
        return [e for e in self.get_recent_events(since) if e.event_type == event_type]

    def get_current_risk(self) -> float:
        """Get the current decayed risk score."""
        return min(self._cumulative_risk, 1.0)

    def get_event_counts(self) -> dict:
        """Get counts of each event type in the current window."""
        counts: dict = {}
        for entry in self.get_recent_events():
            counts[entry.event_type] = counts.get(entry.event_type, 0) + 1
        return counts

    def get_signal_summary(self) -> dict:
        """Get a summary of independent signal categories and their counts."""
        signal_map = {
            'vision': ['face_absent', 'multiple_faces', 'head_turned', 'gaze_away'],
            'audio': ['loud_audio', 'speech_detected'],
            'keyboard': ['rapid_typing', 'idle_keyboard', 'copy_paste'],
            'mouse': ['rapid_movement', 'idle_mouse', 'right_click_spam'],
            'temporal': ['tab_switch', 'window_blur', 'fullscreen_exit'],
        }
        counts = self.get_event_counts()
        summary = {}
        for signal, events in signal_map.items():
            summary[signal] = sum(counts.get(e, 0) for e in events)
        return summary

    def count_active_signals(self, threshold: int = 1) -> int:
        """Count how many independent signal categories have events above threshold."""
        summary = self.get_signal_summary()
        return sum(1 for count in summary.values() if count >= threshold)

    def clear(self) -> None:
        """Reset all memory."""
        self._entries.clear()
        self._cumulative_risk = 0.0
