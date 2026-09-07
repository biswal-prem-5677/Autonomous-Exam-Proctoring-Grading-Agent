"""Integrity Decision Trace — structured explainability for every risk decision.

This is the centerpiece feature. For every high-risk incident, the trace
produces a structured explanation:

                    INTEGRITY DECISION TRACE

14:32:10
├── Vision
│     Face orientation anomaly
│     confidence = 0.74
├── Browser
│     Focus lost
│     confidence = 0.99
├── Audio
│     Speech activity
│     confidence = 0.81
▼
CROSS-MODAL CORROBORATION
3 independent signals within 4.8 seconds
▼
TEMPORAL MODEL
Previous risk: 31 → Updated risk: 82
▼
UNCERTAINTY: Confidence 0.89
▼
AGENT DECISION: HIGH_PRIORITY_REVIEW
▼
WHY? Multiple independent signals in close temporal proximity.
WHY NOT AUTOMATICALLY ACCUSE? Risk indicates suspicious evidence,
not proof of misconduct. Human review required.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import uuid


class DecisionOutcome(Enum):
    """Possible agent decisions."""
    NORMAL = "NORMAL"
    LOW_CONCERN = "LOW_CONCERN"
    MONITOR = "MONITOR"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    HIGH_PRIORITY_REVIEW = "HIGH_PRIORITY_REVIEW"
    TECHNICAL_EVENT = "TECHNICAL_EVENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class EvidenceContribution:
    """A single signal's contribution to the risk score."""
    source: str          # modality: vision, audio, keyboard, etc.
    event_type: str
    raw_severity: float  # 0.0–1.0
    context_weight: float  # adjusted for context
    baseline_deviation: float  # personal baseline deviation
    corroboration_factor: float  # 0.0–1.0 from corroboration engine
    contribution: float  # final weighted contribution to risk
    confidence: float
    details: Dict = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    """What-if analysis: how risk would change if a signal were absent."""
    if_absent: str        # which signal
    risk_without: float   # hypothetical risk
    delta: float          # risk reduction
    explanation: str


@dataclass
class IntegrityDecisionTrace:
    """Complete decision trace for one risk assessment.

    This is the explainability output for examiners.
    """
    trace_id: str
    timestamp: float
    session_id: str
    student_id: str
    question_id: Optional[str] = None

    # Risk values
    previous_risk: float = 0.0
    current_risk: float = 0.0
    evidence_score: float = 0.0
    ml_risk: Optional[float] = None
    anomaly_score: Optional[float] = None

    # Signal breakdown
    contributions: List[EvidenceContribution] = field(default_factory=list)
    corroboration_result: Optional[Dict] = None

    # Uncertainty
    confidence: float = 0.0
    signal_agreement: int = 0  # how many modalities agreed
    total_signals: int = 0
    uncertainty_reason: str = ""

    # Decision
    decision: str = DecisionOutcome.NORMAL.value
    decision_reason: str = ""
    counterfactuals: List[CounterfactualResult] = field(default_factory=list)

    # What did NOT contribute
    non_contributing: List[str] = field(default_factory=list)

    # Raw evidence
    events: List[Dict] = field(default_factory=list)

    @property
    def uncertainty(self) -> Dict[str, Any]:
        """Uncertainty info as a dict (for API compatibility)."""
        return {
            "confidence": round(self.confidence, 3),
            "signal_agreement": self.signal_agreement,
            "total_signals": self.total_signals,
            "reason": self.uncertainty_reason,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace for API responses and storage."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "timestamp_iso": _format_time(self.timestamp),
            "session_id": self.session_id,
            "student_id": self.student_id,
            "question_id": self.question_id,
            "risk": {
                "previous": round(self.previous_risk, 4),
                "current": round(self.current_risk, 4),
                "evidence_score": round(self.evidence_score, 4),
                "ml_risk": round(self.ml_risk, 4) if self.ml_risk is not None else None,
                "anomaly_score": (round(self.anomaly_score, 4)
                                  if self.anomaly_score is not None else None),
            },
            "contributions": [
                {
                    "source": c.source,
                    "event_type": c.event_type,
                    "raw_severity": round(c.raw_severity, 3),
                    "context_weight": round(c.context_weight, 3),
                    "baseline_deviation": round(c.baseline_deviation, 3),
                    "corroboration_factor": round(c.corroboration_factor, 3),
                    "contribution": round(c.contribution, 3),
                    "confidence": round(c.confidence, 3),
                    "details": c.details,
                }
                for c in self.contributions
            ],
            "corroboration": self.corroboration_result,
            "uncertainty": {
                "confidence": round(self.confidence, 3),
                "signal_agreement": self.signal_agreement,
                "total_signals": self.total_signals,
                "reason": self.uncertainty_reason,
            },
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "counterfactuals": [
                {
                    "if_absent": cf.if_absent,
                    "risk_without": round(cf.risk_without, 4),
                    "delta": round(cf.delta, 4),
                    "explanation": cf.explanation,
                }
                for cf in self.counterfactuals
            ],
            "non_contributing": self.non_contributing,
            "events": self.events,
        }


def build_trace(
    session_id: str,
    student_id: str,
    events: List[Dict],
    risk_score: float,
    previous_risk: float,
    contributions: List[EvidenceContribution],
    corroboration_result: Optional[Dict] = None,
    ml_risk: Optional[float] = None,
    anomaly_score: Optional[float] = None,
    confidence: float = 0.5,
    decision: str = DecisionOutcome.REVIEW_REQUIRED.value,
    decision_reason: str = "",
    non_contributing: Optional[List[str]] = None,
    question_id: Optional[str] = None,
) -> IntegrityDecisionTrace:
    """Build a complete decision trace.

    Args:
        session_id: Current exam session ID.
        student_id: Student being monitored.
        events: Raw events detected in this assessment cycle.
        risk_score: Current risk score (0.0–1.0).
        previous_risk: Risk score before this cycle.
        contributions: Per-signal contribution breakdown.
        corroboration_result: Cross-modal corroboration analysis.
        ml_risk: ML model risk prediction.
        anomaly_score: Anomaly detection score.
        confidence: Overall confidence in this assessment.
        decision: Agent decision outcome.
        decision_reason: Human-readable reason for decision.
        non_contributing: List of modalities that were normal.
        question_id: Current question context.

    Returns:
        IntegrityDecisionTrace ready for storage or API response.
    """
    import uuid
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    timestamp = time.time()

    # Compute uncertainty reason
    if confidence < 0.3:
        uncertainty_reason = _low_confidence_reason(contributions, corroboration_result)
    elif corroboration_result and corroboration_result.get("corroboration_label") in (
        "UNCORROBORATED", "NO_SIGNALS"
    ):
        uncertainty_reason = (
            "Single-modality evidence only. "
            "Cross-modal corroboration unavailable."
        )
    else:
        uncertainty_reason = _sufficient_evidence_reason(contributions, corroboration_result)

    # Compute counterfactuals
    counterfactuals = _compute_counterfactuals(
        risk_score, contributions, ml_risk
    )

    return IntegrityDecisionTrace(
        trace_id=trace_id,
        timestamp=timestamp,
        session_id=session_id,
        student_id=student_id,
        question_id=question_id,
        previous_risk=previous_risk,
        current_risk=risk_score,
        evidence_score=sum(c.contribution for c in contributions),
        ml_risk=ml_risk,
        anomaly_score=anomaly_score,
        contributions=contributions,
        corroboration_result=corroboration_result,
        confidence=confidence,
        signal_agreement=(corroboration_result.get("independent_modalities", 0)
                          if corroboration_result else 0),
        total_signals=corroboration_result.get("signal_count", len(events))
        if corroboration_result else len(events),
        uncertainty_reason=uncertainty_reason,
        decision=decision,
        decision_reason=decision_reason or _default_decision_reason(decision, risk_score),
        counterfactuals=counterfactuals,
        non_contributing=non_contributing or [],
        events=events,
    )


def _compute_counterfactuals(
    current_risk: float,
    contributions: List[EvidenceContribution],
    ml_risk: Optional[float],
) -> List[CounterfactualResult]:
    """Compute counterfactual risk: what would risk be without each signal?"""
    results = []
    total_contribution = sum(c.contribution for c in contributions)
    previous_risk = max(0.0, current_risk - total_contribution)

    for c in contributions:
        risk_without = previous_risk + (total_contribution - c.contribution)
        # Add ML component back if present
        if ml_risk is not None:
            risk_without = 0.55 * risk_without + 0.45 * ml_risk
        risk_without = min(max(risk_without, 0.0), 1.0)
        delta = current_risk - risk_without

        results.append(CounterfactualResult(
            if_absent=f"{c.source}: {c.event_type}",
            risk_without=round(risk_without, 4),
            delta=round(delta, 4),
            explanation=(f"If {c.source} {c.event_type} were absent, "
                         f"risk would be {risk_without:.0%} instead of "
                         f"{current_risk:.0%} (delta: -{delta:.0%})."),
        ))

    # Sort by delta descending — biggest contributors first
    results.sort(key=lambda r: r.delta, reverse=True)
    return results[:5]  # Top 5 contributors


def _low_confidence_reason(
    contributions: List[EvidenceContribution],
    corroboration: Optional[Dict],
) -> str:
    reasons = []
    if not contributions:
        reasons.append("No contributing signals detected.")
    if corroboration and corroboration.get("independent_modalities", 0) < 2:
        reasons.append("Single-modality evidence only.")
    if len(reasons):
        return " ".join(reasons)
    return "Insufficient signal quality for high-confidence assessment."


def _sufficient_evidence_reason(
    contributions: List[EvidenceContribution],
    corroboration: Optional[Dict],
) -> str:
    parts = []
    if contributions:
        parts.append(f"{len(contributions)} contributing signals.")
    if corroboration:
        label = corroboration.get("corroboration_label", "")
        mods = corroboration.get("independent_modalities", 0)
        if label == "STRONGLY_CORROBORATED":
            parts.append(f"Strong cross-modal corroboration ({mods} modalities).")
        elif label == "CORROBORATED":
            parts.append(f"Cross-modal corroboration present ({mods} modalities).")
    return " ".join(parts) if parts else "Adequate evidence quality."


def _default_decision_reason(decision: str, risk: float) -> str:
    reasons = {
        DecisionOutcome.NORMAL.value: "No suspicious activity detected.",
        DecisionOutcome.LOW_CONCERN.value: (
            "Minor signals detected but within expected range."
        ),
        DecisionOutcome.MONITOR.value: (
            "Elevated risk detected. Continuing observation."
        ),
        DecisionOutcome.REVIEW_REQUIRED.value: (
            f"Risk score {risk:.0%} exceeds review threshold. "
            "Human review required."
        ),
        DecisionOutcome.HIGH_PRIORITY_REVIEW.value: (
            f"Risk score {risk:.0%} exceeds escalation threshold with "
            "multiple corroborating signals. Immediate review recommended."
        ),
        DecisionOutcome.TECHNICAL_EVENT.value: (
            "Technical issue detected — not an integrity concern."
        ),
        DecisionOutcome.INSUFFICIENT_EVIDENCE.value: (
            "Unable to form confident assessment due to limited or "
            "unreliable signals."
        ),
    }
    return reasons.get(decision, "")


def _format_time(timestamp: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]


def build_explainability_trace(
    session_id: str,
    events: List[Dict],
    risk_result: Dict[str, Any],
    risk_engine: Any,
    evidence_memory: Any,
    agent_state: str,
    student_id: str = "unknown",
    question_id: Optional[str] = None,
) -> IntegrityDecisionTrace:
    """Build a full explainability trace from the agent's control loop output.

    This is the function called by ProctoringAgent._observe after each cycle.
    It converts raw events into structured EvidenceContributions, runs
    cross-modal corroboration, computes counterfactuals, and produces a
    DecisionTrace ready for the examiner UI.

    Args:
        session_id: Exam session identifier.
        events: List of detected event dicts from _perceive().
        risk_result: Dict from agent._observe() containing risk_score,
            ml_predictions, actions, etc.
        risk_engine: The RiskEngine instance (for weights and corroboration).
        evidence_memory: The EvidenceMemory instance (for signal retrieval).
        agent_state: Current agent state string.
        student_id: Student identifier.
        question_id: Current question context.

    Returns:
        IntegrityDecisionTrace with full explainability data.
    """
    trace_id = build_trace_id()
    now = time.time()

    # Extract risk values from result
    current_risk = risk_result.get("risk_score", 0.0)
    previous_risk = risk_result.get("previous_risk", 0.0)
    ml_predictions = risk_result.get("ml_predictions", {})
    ml_risk = ml_predictions.get("logistic_risk")
    anomaly_score = ml_predictions.get("anomaly_score")

    # Map event types to modalities
    MODALITY_MAP = {
        "face_absent": "vision", "multiple_faces": "vision",
        "head_turned": "vision", "gaze_away": "vision",
        "head_pose": "vision",
        "speech_detected": "audio", "loud_audio": "audio",
        "speech_activity": "audio",
        "rapid_typing": "keyboard", "idle_keyboard": "keyboard",
        "copy_paste": "keyboard", "paste_detected": "keyboard",
        "rapid_movement": "mouse", "idle_mouse": "mouse",
        "tab_switch": "browser", "window_blur": "browser",
        "fullscreen_exit": "browser", "visibility_change": "browser",
        "focus_lost": "browser",
    }

    # Determine the decision from agent state (string values)
    _state_decision_map = {
        "NORMAL": DecisionOutcome.NORMAL,
        "LOW_CONCERN": DecisionOutcome.LOW_CONCERN,
        "MONITOR": DecisionOutcome.MONITOR,
        "SUSPICIOUS": DecisionOutcome.REVIEW_REQUIRED,
        "HIGH_RISK": DecisionOutcome.HIGH_PRIORITY_REVIEW,
        "REVIEW_REQUIRED": DecisionOutcome.REVIEW_REQUIRED,
        "TECHNICAL_EVENT": DecisionOutcome.TECHNICAL_EVENT,
        "INSUFFICIENT_EVIDENCE": DecisionOutcome.INSUFFICIENT_EVIDENCE,
    }
    decision = _state_decision_map.get(agent_state, DecisionOutcome.REVIEW_REQUIRED)

    # Build corroboration signals from recent events
    from src.fusion import CorroborationSignal, compute_corroboration
    corrob_signals = []
    for evt in events:
        evt_type = evt.get("type", "")
        modality = MODALITY_MAP.get(evt_type, "vision")
        weight = risk_engine.EVENT_WEIGHTS.get(evt_type, 0.1) if hasattr(risk_engine, 'EVENT_WEIGHTS') else 0.1
        corrob_signals.append(CorroborationSignal(
            modality=modality,
            event_type=evt_type,
            severity=weight,
            confidence=evt.get("confidence", 0.5),
            timestamp=now,
        ))

    # Run corroboration
    corrob_result = None
    try:
        corrob_result = compute_corroboration(
            corrob_signals,
            temporal_window=getattr(risk_engine, 'temporal_window', 10.0)
        )
    except Exception:
        pass

    corrob_dict = corrob_result.to_dict() if corrob_result else None

    # Build evidence contributions for each event
    contributions: List[EvidenceContribution] = []
    total_raw = 0.0
    active_modalities = set()

    for evt in events:
        evt_type = evt.get("type", "")
        source = MODALITY_MAP.get(evt_type, "vision")
        confidence_val = evt.get("confidence", 0.5)
        raw_severity = risk_engine.EVENT_WEIGHTS.get(evt_type, 0.1) if hasattr(risk_engine, 'EVENT_WEIGHTS') else 0.1

        # Context weight (context-aware scoring)
        ctx_weight = 1.0
        try:
            from src.context import ExamContext as _EC, ExamPolicy as _EP
            exam_ctx = _EC(
                policy=_EP(),
                exam_elapsed_seconds=3600.0,
                exam_duration_seconds=7200.0,
            )
            ctx_weight = risk_engine._compute_context_weight(
                evt_type, exam_ctx, source
            ) if hasattr(risk_engine, '_compute_context_weight') else 1.0
        except Exception:
            pass

        # Baseline deviation
        base_dev = 0.0
        try:
            if hasattr(risk_engine, '_baseline_deviation') and hasattr(risk_engine, '_baseline'):
                bl = getattr(risk_engine, '_baseline', None)
                if bl and hasattr(bl, 'is_ready') and bl.is_ready():
                    base_dev = risk_engine._baseline_deviation(evt_type, bl)
        except Exception:
            pass

        # Corroboration factor: boost if corroborated
        corrob_factor = 1.0
        if corrob_result and corrob_result.corroboration_score > 0.5:
            corrob_factor = 1.0 + corrob_result.corroboration_score * 0.5
            active_modalities.add(source)

        contribution = raw_severity * ctx_weight * (1.0 + base_dev) * corrob_factor
        total_raw += contribution

        contributions.append(EvidenceContribution(
            source=source,
            event_type=evt_type,
            raw_severity=raw_severity,
            context_weight=ctx_weight,
            baseline_deviation=base_dev,
            corroboration_factor=corrob_factor,
            contribution=round(contribution, 4),
            confidence=confidence_val,
            details=evt.get("details", ""),
        ))

    # Determine non-contributing modalities
    all_modalities = {"vision", "audio", "keyboard", "mouse", "browser"}
    non_contributing = sorted(all_modalities - active_modalities)
    if not non_contributing:
        # If all modalities have some signal, check which ones had clean signals
        observed_mods = set()
        for evt in events:
            mod = MODALITY_MAP.get(evt.get("type", ""), "vision")
            observed_mods.add(mod)
        for mod in all_modalities - observed_mods:
            non_contributing.append(mod)

    # Compute confidence: higher with more corroboration
    confidence = min(
        (corrob_result.corroboration_score if corrob_result else 0.0) * 0.7 +
        len(contributions) * 0.1 +
        sum(c.confidence for c in contributions) / max(len(contributions), 1) * 0.3,
        1.0
    )
    confidence = max(confidence, 0.3)  # floor

    # Compute uncertainty reason
    signal_count = len(events)
    modality_count = len(active_modalities)
    if corrob_result:
        label = corrob_result.corroboration_label
        if label == "STRONGLY_CORROBORATED":
            uncertainty_reason = (
                f"High confidence: {modality_count} independent modalities "
                f"corroborated within {corrob_result.window_seconds:.1f}s window."
            )
        elif label == "CORROBORATED":
            uncertainty_reason = (
                f"Moderate confidence: {modality_count} modalities "
                f"provided corroborating signals."
            )
        elif label == "WEAK_CORROBORATION":
            uncertainty_reason = "Weak cross-modal corroboration."
        else:
            uncertainty_reason = (
                "Single-modality evidence only. "
                "Cross-modal corroboration unavailable."
            )
    elif signal_count == 0:
        uncertainty_reason = "No signals detected."
    else:
        uncertainty_reason = "Unable to compute corroboration."

    # Build counterfactuals using the internal helper
    try:
        counterfactuals = _compute_counterfactuals(
            current_risk, contributions, ml_risk
        )
    except Exception:
        counterfactuals = []

    # Decision reason
    decision_reason = _default_decision_reason(
        decision.value if hasattr(decision, 'value') else decision,
        current_risk
    )

    return IntegrityDecisionTrace(
        trace_id=trace_id,
        timestamp=now,
        session_id=session_id,
        student_id=student_id,
        question_id=question_id,
        previous_risk=round(previous_risk, 4),
        current_risk=round(current_risk, 4),
        evidence_score=round(total_raw, 4),
        ml_risk=round(ml_risk, 4) if ml_risk is not None else None,
        anomaly_score=round(anomaly_score, 4) if anomaly_score is not None else None,
        contributions=contributions,
        corroboration_result=corrob_dict,
        confidence=round(confidence, 3),
        signal_agreement=modality_count,
        total_signals=signal_count,
        uncertainty_reason=uncertainty_reason,
        decision=decision.value if hasattr(decision, 'value') else str(decision),
        decision_reason=decision_reason,
        counterfactuals=counterfactuals,
        non_contributing=non_contributing,
        events=events,
    )


def build_trace_id() -> str:
    """Generate a unique trace ID."""
    return f"trace_{uuid.uuid4().hex[:12]}"
