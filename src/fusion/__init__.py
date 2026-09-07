"""Fusion engine — cross-modal corroboration and evidence fusion.

Cross-modal corroboration computes how independent signals across
different modalities temporally align to produce a confidence
multiplier:

  1 signal → confidence 0.45
  2 signals → 0.62
  3 signals → 0.78
  4+ signals → 0.86+

Uses Bayesian-inspired formula: P(E_1,E_2,...,E_n) ≈
P(E_1) × (1 + (n-1) × correlation_factor × avg_reliability)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ModalitySource(Enum):
    VISION = "vision"
    AUDIO = "audio"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    BROWSER = "browser"
    INTERACTION = "interaction"


@dataclass
class CorroborationSignal:
    modality: ModalitySource
    event_type: str
    severity: float
    confidence: float
    timestamp: float
    duration: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class CorroborationResult:
    signal_count: int
    independent_modalities: int
    temporal_window_seconds: float
    corroboration_score: float
    corroboration_label: str
    correlated_group: List[Dict]
    uncorrelated_signals: List[Dict]
    explanation: str


# Corroboration base scores — Bayesian-inspired
# P(E_1) = 0.45, P(E_1,E_2) ≈ 0.62, P(E_1,E_2,E_3) ≈ 0.78
_CORROBORATION_BASE = {1: 0.45, 2: 0.62, 3: 0.78, 4: 0.87, 5: 0.93, 6: 0.97}

_MODALITY_RELIABILITY: Dict[ModalitySource, float] = {
    ModalitySource.VISION: 0.90,
    ModalitySource.AUDIO: 0.70,
    ModalitySource.KEYBOARD: 0.75,
    ModalitySource.MOUSE: 0.70,
    ModalitySource.BROWSER: 0.95,
    ModalitySource.INTERACTION: 0.65,
}

DEFAULT_TEMPORAL_WINDOW = 5.0


def compute_corroboration(
    signals: List[CorroborationSignal],
    temporal_window: float = DEFAULT_TEMPORAL_WINDOW,
    min_signals: int = 2,
) -> CorroborationResult:
    """Analyze signals for cross-modal corroboration."""
    if not signals:
        return _empty_result(temporal_window)

    sorted_signals = sorted(signals, key=lambda s: s.timestamp)

    # Find temporally-correlated groups
    groups = _find_temporal_groups(sorted_signals, temporal_window)

    # Pick the largest correlated group as primary evidence cluster
    if groups:
        primary_group = max(groups, key=len)
    else:
        primary_group = [sorted_signals[0]]

    correlated = [_signal_to_dict(s) for s in primary_group]
    primary_ids = {id(s) for s in primary_group}
    uncorrelated = [
        _signal_to_dict(s) for s in sorted_signals
        if id(s) not in primary_ids
    ]

    unique_modalities = set(s.modality for s in primary_group)
    independent_count = len(unique_modalities)
    sig_count = len(primary_group)

    # Base score from corroboration table
    base_score = _CORROBORATION_BASE.get(min(sig_count, 6), 0.97)

    # Adjust by modality reliability — Bayesian-inspired multiplier
    if unique_modalities:
        avg_rel = np.mean([
            _MODALITY_RELIABILITY.get(m, 0.5) for m in unique_modalities
        ])
        # Blend base score with reliability factor
        # High reliability → score approaches 1.0
        base_score = base_score * (0.6 + 0.4 * avg_rel)

    # Temporal spread: tighter grouping = higher confidence
    if len(primary_group) > 1:
        time_span = primary_group[-1].timestamp - primary_group[0].timestamp
        if time_span < temporal_window:
            spread_bonus = (1.0 - time_span / temporal_window) * 0.10
            base_score = min(base_score + spread_bonus, 1.0)

    score = min(max(base_score, 0.0), 1.0)

    # Label based on independent modality count and score
    if independent_count < min_signals:
        label = "UNCORROBORATED"
    elif score >= 0.85:
        label = "STRONGLY_CORROBORATED"
    elif score >= 0.70:
        label = "CORROBORATED"
    elif score >= 0.50:
        label = "WEAKLY_CORROBORATED"
    else:
        label = "UNCORROBORATED"

    explanation = _explain(label, independent_count, sig_count,
                           unique_modalities, uncorrelated, temporal_window)

    return CorroborationResult(
        signal_count=len(sorted_signals),
        independent_modalities=independent_count,
        temporal_window_seconds=temporal_window,
        corroboration_score=round(score, 4),
        corroboration_label=label,
        correlated_group=correlated,
        uncorrelated_signals=uncorrelated,
        explanation=explanation,
    )


def _find_temporal_groups(signals, window):
    """Group signals within the temporal window of the group's first signal."""
    if len(signals) <= 1:
        return [signals] if signals else []

    groups = []
    current_group = [signals[0]]

    for sig in signals[1:]:
        time_since_start = sig.timestamp - current_group[0].timestamp
        if time_since_start <= window:
            current_group.append(sig)
        else:
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = [sig]

    if len(current_group) >= 2:
        groups.append(current_group)

    return groups


def _signal_to_dict(sig):
    return {
        "modality": sig.modality.value,
        "event_type": sig.event_type,
        "severity": round(sig.severity, 3),
        "confidence": round(sig.confidence, 3),
        "timestamp": sig.timestamp,
        "duration": sig.duration,
        "details": sig.details,
    }


def _explain(label, independent, total, modalities, uncorrelated, window):
    mod_names = ", ".join(sorted(m.value for m in modalities))
    uncorr_count = len(uncorrelated)

    if independent >= 4:
        return (f"{total} signals across {independent} independent modalities "
                f"({mod_names}) temporally aligned within {window:.1f}s. "
                f"Strong cross-modal corroboration. {label}.")
    elif independent >= 2:
        uncorr_note = (f" {uncorr_count} additional signal(s) without "
                        f"corroboration." if uncorr_count > 0 else "")
        return (f"{total} signals across {independent} independent modalities "
                f"({mod_names}) within {window:.1f}s. "
                f"Moderate cross-modal evidence.{uncorr_note} {label}.")
    elif independent == 1:
        return (f"{total} signal(s) from single modality ({mod_names}). "
                f"No cross-modal corroboration. {label}.")
    return f"No corroborating evidence. {label}."


def _empty_result(temporal_window):
    return CorroborationResult(
        signal_count=0, independent_modalities=0,
        temporal_window_seconds=temporal_window,
        corroboration_score=0.0, corroboration_label="NO_SIGNALS",
        correlated_group=[], uncorrelated_signals=[],
        explanation="No signals to corroborate.",
    )
