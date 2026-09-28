"""Causal and Counterfactual Reasoning Engine.

Answers:
1. 'What evidence actually caused the risk increase?'
2. 'If evidence X had NOT occurred, what would the risk be?'
3. 'What is the Minimal Sufficient Cause (the smallest set of events that pushes risk over threshold)?'

Mathematical foundation:
    Marginal Causal Attribution:
        ΔR(e_i) = R(E) - R(E \\ {e_i})

    Joint Causal Attribution:
        ΔR(S) = R(E) - R(E \\ S)

    Minimal Sufficient Cause:
        S* = argmin_{S ⊆ E} |S|  s.t.  R(E \\ S) < threshold_review
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any
import copy


@dataclass
class CausalFactor:
    """Individual evidence factor with its marginal causal contribution."""
    factor_id: str
    modality: str
    event_type: str
    original_risk: float
    risk_without: float
    marginal_delta: float       # Original - Without
    percent_contribution: float # Share of total risk increase
    is_necessary_cause: bool    # Whether removing this factor alone drops risk below review threshold
    explanation: str


@dataclass
class CounterfactualAnalysis:
    """Comprehensive causal and counterfactual breakdown of a decision."""
    session_id: str
    original_risk: float
    decision_threshold: float
    factors: List[CausalFactor]
    minimal_sufficient_cause: List[str]  # Factor IDs
    robustness_margin: float             # How much excess risk above threshold exists
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "original_risk": round(self.original_risk, 4),
            "decision_threshold": round(self.decision_threshold, 4),
            "factors": [
                {
                    "factor_id": f.factor_id,
                    "modality": f.modality,
                    "event_type": f.event_type,
                    "risk_without": round(f.risk_without, 4),
                    "marginal_delta": round(f.marginal_delta, 4),
                    "percent_contribution": round(f.percent_contribution, 2),
                    "is_necessary_cause": f.is_necessary_cause,
                    "explanation": f.explanation,
                }
                for f in self.factors
            ],
            "minimal_sufficient_cause": self.minimal_sufficient_cause,
            "robustness_margin": round(self.robustness_margin, 4),
            "explanation": self.explanation,
        }


class CounterfactualEngine:
    """Computes causal attribution and counterfactual simulations."""

    def __init__(self, review_threshold: float = 0.50):
        self.review_threshold = review_threshold

    def analyze(
        self,
        session_id: str,
        current_risk: float,
        evidence_contributions: List[Dict[str, Any]],
        eval_risk_fn: Optional[Callable[[List[Dict[str, Any]]], float]] = None,
    ) -> CounterfactualAnalysis:
        """Run counterfactual reasoning over evidence contributions.

        Args:
            session_id: Session identifier
            current_risk: Present risk score (0.0 to 1.0)
            evidence_contributions: List of evidence items with keys:
                {'source': modality, 'event_type': str, 'contribution': float, ...}
            eval_risk_fn: Optional simulator function to re-evaluate risk given an evidence subset.
                          Defaults to linear delta approximation if None.
        """
        if not evidence_contributions:
            return CounterfactualAnalysis(
                session_id=session_id,
                original_risk=current_risk,
                decision_threshold=self.review_threshold,
                factors=[],
                minimal_sufficient_cause=[],
                robustness_margin=max(0.0, current_risk - self.review_threshold),
                explanation="No evidence factors present for counterfactual attribution.",
            )

        total_contribution = sum(c.get("contribution", 0.0) for c in evidence_contributions)
        factors: List[CausalFactor] = []

        for idx, item in enumerate(evidence_contributions):
            item_id = f"{item.get('source', 'sensor')}:{item.get('event_type', 'event')}_{idx}"
            modality = item.get("source", "unknown")
            event_type = item.get("event_type", "unknown")

            if eval_risk_fn:
                subset = [c for i, c in enumerate(evidence_contributions) if i != idx]
                risk_without = eval_risk_fn(subset)
            else:
                contrib = item.get("contribution", 0.0)
                risk_without = max(0.0, current_risk - contrib)

            delta = max(0.0, current_risk - risk_without)
            pct = (delta / total_contribution * 100.0) if total_contribution > 0 else 0.0
            is_necessary = (current_risk >= self.review_threshold) and (risk_without < self.review_threshold)

            expl = (
                f"Removing {modality} '{event_type}' reduces risk from "
                f"{current_risk:.1%} to {risk_without:.1%} (Δ = -{delta:.1%})."
            )
            if is_necessary:
                expl += " [NECESSARY CAUSE: removing this factor alone drops risk below review threshold]"

            factors.append(CausalFactor(
                factor_id=item_id,
                modality=modality,
                event_type=event_type,
                original_risk=current_risk,
                risk_without=risk_without,
                marginal_delta=delta,
                percent_contribution=pct,
                is_necessary_cause=is_necessary,
                explanation=expl,
            ))

        # Sort by marginal delta descending
        factors.sort(key=lambda f: f.marginal_delta, reverse=True)

        # Compute Minimal Sufficient Cause: greedy subset whose removal brings risk below threshold
        minimal_cause = []
        simulated_risk = current_risk
        if current_risk >= self.review_threshold:
            for f in factors:
                minimal_cause.append(f.factor_id)
                simulated_risk -= f.marginal_delta
                if simulated_risk < self.review_threshold:
                    break

        margin = max(0.0, current_risk - self.review_threshold)
        top_factor_str = factors[0].explanation if factors else "None"
        overall_expl = (
            f"Primary causal factor: {top_factor_str} "
            f"Minimal set of factors required to clear violation: {len(minimal_cause)} event(s)."
        )

        return CounterfactualAnalysis(
            session_id=session_id,
            original_risk=current_risk,
            decision_threshold=self.review_threshold,
            factors=factors,
            minimal_sufficient_cause=minimal_cause,
            robustness_margin=margin,
            explanation=overall_expl,
        )
