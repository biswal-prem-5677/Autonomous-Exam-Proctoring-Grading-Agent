"""Evidence log — structured record of all proctoring events and findings."""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class EvidenceSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class EvidenceEntry:
    """A single evidence log entry."""
    timestamp: float
    event_type: str
    severity: str
    confidence: float
    description: str
    signal_source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d


class EvidenceLog:
    """Structured evidence log for all proctoring events.

    Maintains a chronologically ordered, JSON-serializable record
    of every signal event with severity classification.
    """

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self._entries: List[EvidenceEntry] = []

    def add(self, event_type: str, severity: str, confidence: float,
            description: str, signal_source: str = "",
            metadata: Dict = None) -> EvidenceEntry:
        """Add a new evidence entry."""
        # Auto-classify severity if not provided
        if not severity:
            severity = self._classify_severity(event_type, confidence)
        entry = EvidenceEntry(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            confidence=confidence,
            description=description,
            signal_source=signal_source,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        return entry

    def add_from_risk_event(self, event_type: str, confidence: float,
                            description: str, signal_source: str = "") -> EvidenceEntry:
        """Add an entry with severity determined from event type."""
        severity = self._classify_severity(event_type, confidence)
        return self.add(event_type, severity, confidence, description, signal_source)

    def _classify_severity(self, event_type: str, confidence: float) -> str:
        """Classify severity based on event type and confidence."""
        critical_events = {
            "face_absent", "multiple_faces", "copy_paste", "paste_detected",
            "fullscreen_exit", "speech_detected",
        }
        if event_type in critical_events:
            return "critical" if confidence >= 0.5 else "warning"
        if confidence >= 0.6:
            return "warning"
        return "info"

    def get_entries(self, severity: Optional[str] = None,
                    event_type: Optional[str] = None) -> List[EvidenceEntry]:
        """Filter evidence entries."""
        results = self._entries
        if severity:
            results = [e for e in results if e.severity == severity]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        return results

    def get_critical_count(self) -> int:
        """Count critical-severity events."""
        return sum(1 for e in self._entries if e.severity == "critical")

    def get_timeline(self) -> List[Dict]:
        """Get evidence as a chronological timeline for reports."""
        return [e.to_dict() for e in self._entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "total_events": len(self._entries),
            "critical_count": self.get_critical_count(),
            "warning_count": sum(1 for e in self._entries if e.severity == "warning"),
            "info_count": sum(1 for e in self._entries if e.severity == "info"),
            "timeline": self.get_timeline(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def clear(self) -> None:
        """Reset the evidence log."""
        self._entries.clear()
