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
from collections import defaultdict
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
class ContradictionRecord:
    modality_a: str
    modality_b: str
    description: str
    confidence: float
    implication: str  # e.g., "external_audio_source", "automated_input_injection", "sensor_glitch"


@dataclass
class CorroborationResult:
    signal_count: int
    independent_modalities: int
    temporal_window_seconds: float
    corroboration_score: float
    corroboration_label: str
    correlated_group: List[Dict]
    uncorrelated_signals: List[Dict]
    contradictions: List[ContradictionRecord]
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


def _detect_contradictions(signals: List[CorroborationSignal]) -> List[ContradictionRecord]:
    """Detect physical/logical contradictions across sensory modalities.

    Examples:
    - Audio speech detected, but vision indicates closed mouth / no facial motion
      -> suggests background speaker / television / external noise, not student speaking.
    - Rapid keystroke typing, but motion detector shows zero body/hand motion
      -> suggests macro / script / automated input injection.
    - Gaze away event, but face is absent
      -> sensor mismatch / invalid gaze estimation without face.
    """
    contradictions = []
    events_by_mod = defaultdict(list)
    for s in signals:
        events_by_mod[s.modality].append(s)

    vision_events = {s.event_type: s for s in events_by_mod.get(ModalitySource.VISION, [])}
    audio_events = {s.event_type: s for s in events_by_mod.get(ModalitySource.AUDIO, [])}
    kb_events = {s.event_type: s for s in events_by_mod.get(ModalitySource.KEYBOARD, [])}

    # 1. Audio speech vs Vision mouth movement
    if "speech_detected" in audio_events:
        speech_sig = audio_events["speech_detected"]
        # Check if vision reports mouth still or no face motion in details
        mouth_moving = speech_sig.details.get("mouth_moving", False)
        if not mouth_moving and "face_absent" not in vision_events and vision_events:
            contradictions.append(ContradictionRecord(
                modality_a="audio",
                modality_b="vision",
                description="Speech detected in audio but vision observed no corresponding mouth movement.",
                confidence=0.82,
                implication="external_audio_source",
            ))

    # 2. Keystroke typing vs Motion
    if "rapid_typing" in kb_events and "movement" in vision_events:
        mov_sig = vision_events["movement"]
        if mov_sig.severity < 0.05:  # virtually zero movement
            contradictions.append(ContradictionRecord(
                modality_a="keyboard",
                modality_b="vision",
                description="Rapid typing detected while upper body / hand motion is static.",
                confidence=0.78,
                implication="automated_input_injection",
            ))

    # 3. Gaze deviation vs Face absent
    if "gaze_away" in vision_events and "face_absent" in vision_events:
        contradictions.append(ContradictionRecord(
            modality_a="vision_gaze",
            modality_b="vision_face",
            description="Gaze deviation registered while face is completely absent.",
            confidence=0.95,
            implication="sensor_contradiction",
        ))

    return contradictions


def compute_corroboration(
    signals: List[CorroborationSignal],
    temporal_window: float = DEFAULT_TEMPORAL_WINDOW,
    min_signals: int = 2,
) -> CorroborationResult:
    """Analyze signals for cross-modal corroboration, independence, and contradictions."""
    if not signals:
        return _empty_result(temporal_window)

    sorted_signals = sorted(signals, key=lambda s: s.timestamp)

    # Detect contradictions
    contradictions = _detect_contradictions(sorted_signals)

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
        base_score = base_score * (0.6 + 0.4 * avg_rel)

    # Temporal spread: tighter grouping = higher confidence
    if len(primary_group) > 1:
        time_span = primary_group[-1].timestamp - primary_group[0].timestamp
        if time_span < temporal_window:
            spread_bonus = (1.0 - time_span / temporal_window) * 0.10
            base_score = min(base_score + spread_bonus, 1.0)

    # Discount score if contradictions exist (e.g. external audio rather than student speaking)
    if contradictions:
        max_contra_conf = max(c.confidence for c in contradictions)
        base_score = max(0.2, base_score * (1.0 - 0.4 * max_contra_conf))

    score = min(max(base_score, 0.0), 1.0)

    # Label based on independent modality count, score, and contradictions
    if independent_count < min_signals:
        label = "UNCORROBORATED"
    elif contradictions:
        label = "CONTRADICTED"
    elif score >= 0.85:
        label = "STRONGLY_CORROBORATED"
    elif score >= 0.70:
        label = "CORROBORATED"
    elif score >= 0.50:
        label = "WEAKLY_CORROBORATED"
    else:
        label = "UNCORROBORATED"

    explanation = _explain(label, independent_count, sig_count,
                           unique_modalities, uncorrelated, temporal_window, contradictions)

    return CorroborationResult(
        signal_count=len(sorted_signals),
        independent_modalities=independent_count,
        temporal_window_seconds=temporal_window,
        corroboration_score=round(score, 4),
        corroboration_label=label,
        correlated_group=correlated,
        uncorrelated_signals=uncorrelated,
        contradictions=contradictions,
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


def _explain(label, independent, total, modalities, uncorrelated, window, contradictions=None):
    mod_names = ", ".join(sorted(m.value for m in modalities))
    uncorr_count = len(uncorrelated)
    contra_note = ""
    if contradictions:
        contra_note = f" CONTRADICTION DETECTED: {len(contradictions)} cross-modal conflict(s) identified."

    if label == "CONTRADICTED":
        return (f"Cross-modal contradiction observed between sensory channels.{contra_note} "
                f"Discounting integrity escalation pending human verification.")

    if independent >= 4:
        return (f"{total} signals across {independent} independent modalities "
                f"({mod_names}) temporally aligned within {window:.1f}s. "
                f"Strong cross-modal corroboration.{contra_note} {label}.")
    elif independent >= 2:
        uncorr_note = (f" {uncorr_count} additional signal(s) without "
                        f"corroboration." if uncorr_count > 0 else "")
        return (f"{total} signals across {independent} independent modalities "
                f"({mod_names}) within {window:.1f}s. "
                f"Moderate cross-modal evidence.{uncorr_note}{contra_note} {label}.")
    elif independent == 1:
        return (f"{total} signal(s) from single modality ({mod_names}). "
                f"No cross-modal corroboration.{contra_note} {label}.")
    return f"No corroborating evidence.{contra_note} {label}."


def _empty_result(temporal_window):
    return CorroborationResult(
        signal_count=0, independent_modalities=0,
        temporal_window_seconds=temporal_window,
        corroboration_score=0.0, corroboration_label="NO_SIGNALS",
        correlated_group=[], uncorrelated_signals=[],
        contradictions=[],
        explanation="No signals to corroborate.",
    )


class CrossModalCorroborationEngine:
    """Orchestrator for cross-modal corroboration, temporal alignment, and contradiction detection."""

    def __init__(self, temporal_window: float = DEFAULT_TEMPORAL_WINDOW, min_signals: int = 2):
        self.temporal_window = temporal_window
        self.min_signals = min_signals

    def evaluate(self, signals: List[CorroborationSignal]) -> CorroborationResult:
        """Evaluate a list of sensory signals for cross-modal corroboration."""
        return compute_corroboration(
            signals,
            temporal_window=self.temporal_window,
            min_signals=self.min_signals
        )


