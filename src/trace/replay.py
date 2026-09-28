"""Decision Replay Engine — deterministic reconstruction of exam sessions.

Enables human examiners and auditors to inspect any past examination session
tick-by-tick (like a flight data recorder):
    09:31:02 — Face detected (baseline: normal)
    09:31:04 — Tab switch detected
    09:31:05 — Clipboard event (paste 84 chars)
    09:31:06 — Rapid keyboard burst (92 WPM vs baseline 46 WPM)
    09:31:07 — Risk: 0.31 → 0.48 (Δ = +0.17)
    09:31:08 — Cross-modal corroboration: 3 independent channels aligned within 3.2s
    09:31:09 — Risk: 0.48 → 0.76 (Review threshold 0.50 exceeded)
    09:31:10 — Agent Decision: REVIEW_REQUIRED (Audit Entry #104 signed)

Inspect why every parameter changed at any microsecond of the exam.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class ReplaySnapshot:
    """Complete system state snapshot at a specific point in time."""
    step_index: int
    timestamp: float
    iso_time: str
    session_id: str
    student_id: str
    events_at_step: List[Dict[str, Any]]
    active_modalities: List[str]
    corroboration_label: str
    corroboration_score: float
    previous_risk: float
    current_risk: float
    risk_delta: float
    confidence: float
    uncertainty_reason: str
    primary_causal_factor: str
    agent_decision: str
    agent_actions: List[str]
    audit_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "iso_time": self.iso_time,
            "session_id": self.session_id,
            "student_id": self.student_id,
            "events": self.events_at_step,
            "active_modalities": self.active_modalities,
            "corroboration": {
                "label": self.corroboration_label,
                "score": round(self.corroboration_score, 4),
            },
            "risk": {
                "previous": round(self.previous_risk, 4),
                "current": round(self.current_risk, 4),
                "delta": round(self.risk_delta, 4),
            },
            "uncertainty": {
                "confidence": round(self.confidence, 4),
                "reason": self.uncertainty_reason,
            },
            "causal_factor": self.primary_causal_factor,
            "decision": self.agent_decision,
            "actions": self.agent_actions,
            "audit_hash": self.audit_hash,
        }


class DecisionReplayEngine:
    """Step-by-step deterministic decision replayer for audited sessions."""

    def __init__(self, session_id: str, student_id: str = "student_replay"):
        self.session_id = session_id
        self.student_id = student_id
        self._snapshots: List[ReplaySnapshot] = []
        self._cursor: int = 0

    def load_from_traces(self, traces: List[Dict[str, Any]]) -> int:
        """Load session chronology from an ordered list of IntegrityDecisionTraces."""
        self._snapshots.clear()
        sorted_traces = sorted(traces, key=lambda t: t.get("timestamp", 0))

        for idx, t in enumerate(sorted_traces):
            ts = t.get("timestamp", 0.0)
            iso = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts > 0 else "00:00:00"
            risk_info = t.get("risk", {})
            prev_r = risk_info.get("previous", 0.0)
            curr_r = risk_info.get("current", 0.0)

            corrob = t.get("corroboration") or {}
            uncert = t.get("uncertainty") or {}
            cfs = t.get("counterfactuals") or []
            top_cf = cfs[0].get("explanation", "Nominal") if cfs else "No major contributor"

            snap = ReplaySnapshot(
                step_index=idx,
                timestamp=ts,
                iso_time=iso,
                session_id=self.session_id,
                student_id=self.student_id,
                events_at_step=t.get("events", []),
                active_modalities=[c.get("source") for c in t.get("contributions", []) if c.get("source")],
                corroboration_label=corrob.get("corroboration_label", "UNCORROBORATED"),
                corroboration_score=corrob.get("corroboration_score", 0.0) or 0.0,
                previous_risk=prev_r,
                current_risk=curr_r,
                risk_delta=curr_r - prev_r,
                confidence=uncert.get("confidence", 0.5) or 0.5,
                uncertainty_reason=uncert.get("reason", ""),
                primary_causal_factor=top_cf,
                agent_decision=t.get("decision", "NORMAL"),
                agent_actions=t.get("actions", []),
                audit_hash=t.get("trace_id", None),
            )
            self._snapshots.append(snap)

        self._cursor = 0
        return len(self._snapshots)

    def total_steps(self) -> int:
        return len(self._snapshots)

    def current_step(self) -> Optional[ReplaySnapshot]:
        if 0 <= self._cursor < len(self._snapshots):
            return self._snapshots[self._cursor]
        return None

    def step_forward(self) -> Optional[ReplaySnapshot]:
        if self._cursor < len(self._snapshots) - 1:
            self._cursor += 1
            return self._snapshots[self._cursor]
        return self.current_step()

    def step_backward(self) -> Optional[ReplaySnapshot]:
        if self._cursor > 0:
            self._cursor -= 1
            return self._snapshots[self._cursor]
        return self.current_step()

    def seek(self, step_index: int) -> Optional[ReplaySnapshot]:
        if 0 <= step_index < len(self._snapshots):
            self._cursor = step_index
            return self._snapshots[self._cursor]
        return None

    def export_timeline(self) -> List[Dict[str, Any]]:
        """Export high-level timeline for scrubber UI."""
        return [
            {
                "step": s.step_index,
                "time": s.iso_time,
                "risk": round(s.current_risk, 3),
                "delta": round(s.risk_delta, 3),
                "decision": s.agent_decision,
                "events_count": len(s.events_at_step),
                "corroboration": s.corroboration_label,
            }
            for s in self._snapshots
        ]
