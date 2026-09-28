"""Uncertainty Engine — estimates epistemic and aleatoric uncertainty.

A world-class autonomous system must know when it does not know.
Instead of declaring: 'Cheating detected — 93% confidence' when audio is missing,
the uncertainty engine computes:

1. Epistemic Uncertainty (Missing Modalities & Sparse Observations):
   U_epistemic = Σ_{m ∈ Missing} w_m / Σ_{m ∈ All} w_m

2. Aleatoric Uncertainty (Signal Noise & Classification Boundary Ambiguity):
   U_aleatoric = 1.0 - 2.0 · |P(y) - 0.5|  (peaks at P=0.5, zero at P=0 or P=1)

3. Evidence Quality Score:
   Q_evidence = (1 - U_epistemic) · (1 - 0.5 · U_aleatoric) · CorroborationFactor

4. Confidence:
   Confidence = 1.0 - (0.6 · U_epistemic + 0.4 · U_aleatoric)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
import numpy as np


@dataclass
class UncertaintyEstimate:
    """Formal estimation of agent uncertainty for an inference cycle."""
    confidence: float                  # Overall confidence in [0.0, 1.0]
    epistemic_uncertainty: float       # Missing sensors / out-of-distribution
    aleatoric_uncertainty: float       # Inherent signal noise / boundary ambiguity
    evidence_quality: float            # Quality of observed sensory data
    active_modalities: List[str]
    missing_modalities: List[str]
    degraded_modalities: List[str]
    is_actionable: bool                # Whether confidence exceeds policy action threshold
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 4),
            "aleatoric_uncertainty": round(self.aleatoric_uncertainty, 4),
            "evidence_quality": round(self.evidence_quality, 4),
            "active_modalities": self.active_modalities,
            "missing_modalities": self.missing_modalities,
            "degraded_modalities": self.degraded_modalities,
            "is_actionable": self.is_actionable,
            "explanation": self.explanation,
        }


class UncertaintyEngine:
    """Calculates formal uncertainty over multimodal proctoring evidence."""

    MODALITY_WEIGHTS: Dict[str, float] = {
        "vision": 0.35,
        "audio": 0.25,
        "browser": 0.20,
        "keyboard": 0.12,
        "mouse": 0.08,
    }

    def __init__(self, action_confidence_threshold: float = 0.65):
        self.action_threshold = action_confidence_threshold

    def evaluate(
        self,
        observed_modalities: Set[str],
        signal_confidences: Dict[str, float],
        prediction_prob: Optional[float] = None,
        corroboration_score: float = 0.0,
        degraded_sensors: Optional[Set[str]] = None,
    ) -> UncertaintyEstimate:
        """Evaluate multimodal uncertainty for an assessment cycle.

        Args:
            observed_modalities: Set of sensory modalities that provided valid data
            signal_confidences: Dict of {modality: avg_sensor_confidence}
            prediction_prob: ML model probability (if available) for boundary uncertainty
            corroboration_score: Cross-modal corroboration score (0.0 to 1.0)
            degraded_sensors: Set of sensors operating with high noise/low FPS
        """
        all_mods = set(self.MODALITY_WEIGHTS.keys())
        missing_mods = all_mods - observed_modalities
        degraded = degraded_sensors or set()

        total_weight = sum(self.MODALITY_WEIGHTS.values())
        missing_weight = sum(self.MODALITY_WEIGHTS[m] for m in missing_mods if m in self.MODALITY_WEIGHTS)

        # 1. Epistemic uncertainty from missing and degraded channels
        degraded_penalty = sum(0.5 * self.MODALITY_WEIGHTS[m] for m in degraded if m in self.MODALITY_WEIGHTS)
        u_epistemic = min(1.0, (missing_weight + degraded_penalty) / total_weight)

        # 2. Aleatoric uncertainty from model prediction ambiguity
        if prediction_prob is not None:
            # Distance from certainty (0.0 or 1.0). Maximum at 0.5.
            u_aleatoric = 1.0 - 2.0 * abs(prediction_prob - 0.5)
        else:
            # If no model probability, estimate from signal confidences
            if signal_confidences:
                avg_conf = np.mean(list(signal_confidences.values()))
                u_aleatoric = 1.0 - float(avg_conf)
            else:
                u_aleatoric = 0.5

        u_aleatoric = float(np.clip(u_aleatoric, 0.0, 1.0))

        # 3. Overall confidence: higher with corroboration, lower with uncertainty
        base_confidence = 1.0 - (0.55 * u_epistemic + 0.45 * u_aleatoric)
        if corroboration_score > 0.5:
            base_confidence += 0.15 * (corroboration_score - 0.5)

        confidence = float(np.clip(base_confidence, 0.05, 1.0))

        # 4. Evidence quality score
        quality = (1.0 - u_epistemic) * (1.0 - 0.5 * u_aleatoric) * (0.8 + 0.2 * max(corroboration_score, 0.1))
        quality = float(np.clip(quality, 0.0, 1.0))

        is_actionable = confidence >= self.action_threshold

        # Human-readable explanation
        reasons = []
        if missing_mods:
            reasons.append(f"Modalities unavailable: {', '.join(sorted(missing_mods))}.")
        if degraded:
            reasons.append(f"Degraded sensor quality: {', '.join(sorted(degraded))}.")
        if u_aleatoric > 0.6:
            reasons.append("High signal boundary ambiguity.")
        if corroboration_score >= 0.7:
            reasons.append("Strong cross-modal evidence agreement boosts confidence.")
        elif len(observed_modalities) <= 1:
            reasons.append("Single-modality observation limits certainty.")

        explanation = " ".join(reasons) if reasons else "Sufficient multi-signal coverage with nominal certainty."

        return UncertaintyEstimate(
            confidence=confidence,
            epistemic_uncertainty=u_epistemic,
            aleatoric_uncertainty=u_aleatoric,
            evidence_quality=quality,
            active_modalities=sorted(list(observed_modalities)),
            missing_modalities=sorted(list(missing_mods)),
            degraded_modalities=sorted(list(degraded)),
            is_actionable=is_actionable,
            explanation=explanation,
        )
