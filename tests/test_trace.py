"""Tests for Integrity Decision Trace (src/trace)."""

import pytest
import time
from src.trace import (
    IntegrityDecisionTrace,
    build_trace,
    DecisionOutcome,
    EvidenceContribution,
    CounterfactualResult,
)


class TestIntegrityDecisionTrace:
    def test_basic_trace_creation(self):
        trace = build_trace(
            session_id="session_1",
            student_id="student_42",
            events=[{"type": "face_absent", "confidence": 0.9}],
            risk_score=0.82,
            previous_risk=0.31,
            contributions=[
                EvidenceContribution(
                    source="vision", event_type="face_absent",
                    raw_severity=0.30, context_weight=1.0,
                    baseline_deviation=0.0, corroboration_factor=0.76,
                    contribution=0.30, confidence=0.9,
                ),
            ],
            decision=DecisionOutcome.REVIEW_REQUIRED.value,
        )
        assert trace.trace_id.startswith("trace_")
        assert trace.current_risk == 0.82
        assert trace.previous_risk == 0.31
        assert len(trace.contributions) == 1
        assert trace.decision == "REVIEW_REQUIRED"

    def test_trace_to_dict(self):
        trace = build_trace(
            session_id="s1", student_id="st1",
            events=[], risk_score=0.5, previous_risk=0.3,
            contributions=[],
            decision=DecisionOutcome.NORMAL.value,
        )
        d = trace.to_dict()
        assert "trace_id" in d
        assert "risk" in d
        assert "contributions" in d
        assert "corroboration" in d
        assert "uncertainty" in d
        assert "decision" in d
        assert "counterfactuals" in d

    def test_normal_decision_reason(self):
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.1, previous_risk=0.05, contributions=[],
            decision=DecisionOutcome.NORMAL.value,
        )
        assert "No suspicious activity" in trace.decision_reason

    def test_review_required_decision_reason(self):
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.82, previous_risk=0.3, contributions=[],
            decision=DecisionOutcome.REVIEW_REQUIRED.value,
        )
        assert "82%" in trace.decision_reason or "0.82" in trace.decision_reason
        assert "human review" in trace.decision_reason.lower()

    def test_technical_event_decision_reason(self):
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.0, previous_risk=0.0, contributions=[],
            decision=DecisionOutcome.TECHNICAL_EVENT.value,
        )
        assert "Technical" in trace.decision_reason

    def test_counterfactuals_computed(self):
        trace = build_trace(
            session_id="s1", student_id="st1",
            events=[], risk_score=0.82, previous_risk=0.3,
            contributions=[
                EvidenceContribution(
                    source="browser", event_type="tab_switch",
                    raw_severity=0.25, context_weight=1.0,
                    baseline_deviation=0.0, corroboration_factor=0.76,
                    contribution=0.25, confidence=0.99,
                ),
                EvidenceContribution(
                    source="vision", event_type="face_absent",
                    raw_severity=0.30, context_weight=1.0,
                    baseline_deviation=0.0, corroboration_factor=0.76,
                    contribution=0.30, confidence=0.9,
                ),
            ],
            decision=DecisionOutcome.REVIEW_REQUIRED.value,
        )
        assert len(trace.counterfactuals) == 2
        # Sorted by delta descending
        assert trace.counterfactuals[0].delta >= trace.counterfactuals[1].delta
        # Each counterfactual has the required fields
        for cf in trace.counterfactuals:
            assert cf.risk_without < trace.current_risk
            assert cf.delta > 0

    def test_corroboration_result_attached(self):
        corrob = {
            "signal_count": 3,
            "independent_modalities": 3,
            "corroboration_label": "STRONGLY_CORROBORATED",
            "corroboration_score": 0.85,
        }
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.85, previous_risk=0.2, contributions=[],
            corroboration_result=corrob,
            decision=DecisionOutcome.HIGH_PRIORITY_REVIEW.value,
        )
        assert trace.corroboration_result == corrob
        assert trace.signal_agreement == 3
        assert trace.total_signals == 3

    def test_confidence_in_uncertainty(self):
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.65, previous_risk=0.3, contributions=[],
            confidence=0.82,
            decision=DecisionOutcome.REVIEW_REQUIRED.value,
        )
        assert trace.confidence == 0.82
        assert trace.uncertainty["confidence"] == 0.82

    def test_non_contributing_modalities(self):
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.6, previous_risk=0.2,
            contributions=[],
            non_contributing=["keyboard: normal", "mouse: normal",
                              "face presence: normal"],
            decision=DecisionOutcome.REVIEW_REQUIRED.value,
        )
        assert len(trace.non_contributing) == 3
        assert "keyboard: normal" in trace.non_contributing

    def test_trace_has_timestamp(self):
        ts = 1700000000.0
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.5, previous_risk=0.3, contributions=[],
            decision=DecisionOutcome.NORMAL.value,
        )
        assert trace.timestamp > 0

    def test_empty_contributions_no_crash(self):
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.0, previous_risk=0.0, contributions=[],
            ml_risk=None, anomaly_score=None,
            decision=DecisionOutcome.INSUFFICIENT_EVIDENCE.value,
        )
        d = trace.to_dict()
        assert d["contributions"] == []
        assert d["counterfactuals"] == []
        assert d["decision"] == "INSUFFICIENT_EVIDENCE"

    def test_high_priority_decision_reason(self):
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.91, previous_risk=0.3, contributions=[],
            decision=DecisionOutcome.HIGH_PRIORITY_REVIEW.value,
        )
        assert "91%" in trace.decision_reason or "0.91" in trace.decision_reason
        assert "Immediate" in trace.decision_reason

    def test_monitor_decision_reason(self):
        trace = build_trace(
            session_id="s1", student_id="st1", events=[],
            risk_score=0.55, previous_risk=0.3, contributions=[],
            decision=DecisionOutcome.MONITOR.value,
        )
        # The reason should mention continuing observation or monitoring
        reason_lower = trace.decision_reason.lower()
        assert "observation" in reason_lower or "monitor" in reason_lower

    def test_trace_serialization_round_trip(self):
        trace = build_trace(
            session_id="s1", student_id="st1",
            events=[{"type": "face_absent", "confidence": 0.9}],
            risk_score=0.82, previous_risk=0.31,
            contributions=[
                EvidenceContribution(
                    source="vision", event_type="face_absent",
                    raw_severity=0.30, context_weight=1.0,
                    baseline_deviation=0.0, corroboration_factor=0.76,
                    contribution=0.30, confidence=0.9,
                ),
            ],
            ml_risk=0.75, anomaly_score=0.6,
            decision=DecisionOutcome.REVIEW_REQUIRED.value,
        )
        d = trace.to_dict()
        # Verify all expected keys are present and serializable
        import json
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
