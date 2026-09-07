"""Tests for cross-modal corroboration engine (src/fusion)."""

import pytest
import time
from src.fusion import (
    compute_corroboration,
    CorroborationSignal,
    ModalitySource,
    DEFAULT_TEMPORAL_WINDOW,
)


def _make_signal(modality, event_type, severity=0.5, confidence=0.8,
                 timestamp=None, duration=0.0):
    return CorroborationSignal(
        modality=modality,
        event_type=event_type,
        severity=severity,
        confidence=confidence,
        timestamp=timestamp if timestamp is not None else time.time(),
        duration=duration,
    )


class TestCorroborationEngine:
    """Cross-modal corroboration core behavior."""

    def test_no_signals_returns_empty(self):
        result = compute_corroboration([])
        assert result.signal_count == 0
        assert result.corroboration_label == "NO_SIGNALS"
        assert result.corroboration_score == 0.0
        assert result.independent_modalities == 0

    def test_single_signal_uncorroborated(self):
        sig = _make_signal(ModalitySource.VISION, "face_absent",
                           confidence=0.9, severity=0.3)
        result = compute_corroboration([sig])
        assert result.signal_count == 1
        assert result.independent_modalities == 1
        assert result.corroboration_label == "UNCORROBORATED"
        assert result.corroboration_score < 0.5

    def test_two_signals_corroborated(self):
        t = time.time()
        sig1 = _make_signal(ModalitySource.VISION, "face_absent",
                            timestamp=t, severity=0.3)
        sig2 = _make_signal(ModalitySource.BROWSER, "tab_switch",
                            timestamp=t + 0.5, severity=0.25)
        result = compute_corroboration([sig1, sig2])
        assert result.signal_count == 2
        assert result.independent_modalities == 2
        assert result.corroboration_label in ("CORROBORATED", "WEAKLY_CORROBORATED")
        assert result.corroboration_score >= 0.50

    def test_three_signals_strongly_corroborated(self):
        t = time.time()
        sigs = [
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.BROWSER, "tab_switch",
                         timestamp=t + 1.0),
            _make_signal(ModalitySource.AUDIO, "speech_detected",
                         timestamp=t + 2.0),
        ]
        result = compute_corroboration(sigs)
        assert result.independent_modalities == 3
        assert result.corroboration_score >= 0.70
        assert result.corroboration_label in ("CORROBORATED", "STRONGLY_CORROBORATED")

    def test_four_signals_very_high_corroboration(self):
        t = time.time()
        sigs = [
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.AUDIO, "speech_detected",
                         timestamp=t + 0.5),
            _make_signal(ModalitySource.BROWSER, "tab_switch",
                         timestamp=t + 1.0),
            _make_signal(ModalitySource.KEYBOARD, "copy_paste",
                         timestamp=t + 1.5),
        ]
        result = compute_corroboration(sigs)
        assert result.independent_modalities == 4
        assert result.corroboration_score >= 0.80
        assert result.corroboration_label == "STRONGLY_CORROBORATED"

    def test_signals_outside_window_not_corroborated(self):
        t = time.time()
        sig1 = _make_signal(ModalitySource.VISION, "face_absent",
                            timestamp=t, severity=0.3)
        sig2 = _make_signal(ModalitySource.BROWSER, "tab_switch",
                            timestamp=t + 10.0, severity=0.25)  # outside window
        result = compute_corroboration([sig1, sig2],
                                       temporal_window=5.0)
        # Should be in separate groups — primary group has only 1
        assert len(result.correlated_group) == 1
        assert result.corroboration_label == "UNCORROBORATED"
        assert len(result.uncorrelated_signals) == 1

    def test_same_modality_does_not_count_as_independent(self):
        t = time.time()
        sigs = [
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.VISION, "gaze_away",
                         timestamp=t + 0.5),
        ]
        result = compute_corroboration(sigs)
        assert result.independent_modalities == 1  # both are vision
        assert result.corroboration_label == "UNCORROBORATED"

    def test_mixed_modality_counts_correctly(self):
        t = time.time()
        sigs = [
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.VISION, "head_turned",
                         timestamp=t + 0.5),
            _make_signal(ModalitySource.AUDIO, "speech_detected",
                         timestamp=t + 1.0),
        ]
        result = compute_corroboration(sigs)
        # 2 vision + 1 audio = 2 independent modalities
        assert result.independent_modalities == 2

    def test_higher_confidence_boosts_score(self):
        t = time.time()
        low = _make_signal(ModalitySource.VISION, "face_absent",
                           timestamp=t, confidence=0.3, severity=0.3)
        high = _make_signal(ModalitySource.BROWSER, "tab_switch",
                            timestamp=t + 0.5, confidence=0.95, severity=0.25)
        result = compute_corroboration([low, high])
        # With high-reliability browser, the avg reliability should boost score
        assert result.corroboration_score > 0.5

    def test_browser_modality_high_reliability(self):
        """Browser events (tab_switch, focus) are nearly deterministic."""
        t = time.time()
        sig = _make_signal(ModalitySource.BROWSER, "tab_switch",
                           timestamp=t, confidence=0.99, severity=0.25)
        sig2 = _make_signal(ModalitySource.VISION, "face_absent",
                            timestamp=t + 0.5, severity=0.3)
        result = compute_corroboration([sig, sig2])
        # Browser is high reliability (0.95) so it should boost
        assert result.corroboration_score > 0.5

    def test_tight_temporal_grouping_boosts_score(self):
        """Signals clustered tightly in time score higher."""
        t = time.time()
        # Tight group
        tight = [
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.AUDIO, "speech",
                         timestamp=t + 0.3),
            _make_signal(ModalitySource.BROWSER, "tab_switch",
                         timestamp=t + 0.6),
        ]
        result_tight = compute_corroboration(tight, temporal_window=5.0)

        # Loose group
        loose = [
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.AUDIO, "speech",
                         timestamp=t + 3.0),
            _make_signal(ModalitySource.BROWSER, "tab_switch",
                         timestamp=t + 4.5),
        ]
        result_loose = compute_corroboration(loose, temporal_window=5.0)

        assert result_tight.corroboration_score > result_loose.corroboration_score

    def test_custom_temporal_window(self):
        t = time.time()
        sigs = [
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.AUDIO, "speech",
                         timestamp=t + 3.0),
        ]
        result_narrow = compute_corroboration(sigs, temporal_window=2.0)
        result_wide = compute_corroboration(sigs, temporal_window=5.0)

        assert result_narrow.corroboration_label == "UNCORROBORATED"
        assert result_wide.corroboration_label in (
            "WEAKLY_CORROBORATED", "CORROBORATED"
        )

    def test_result_has_explanation(self):
        t = time.time()
        sigs = [
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.AUDIO, "speech",
                         timestamp=t + 0.5),
            _make_signal(ModalitySource.BROWSER, "tab_switch",
                         timestamp=t + 1.0),
        ]
        result = compute_corroboration(sigs)
        assert len(result.explanation) > 0
        assert "3" in result.explanation  # mentions 3 signals

    def test_largest_group_selected_as_primary(self):
        """When there are multiple temporal groups, pick the largest."""
        t = time.time()
        sigs = [
            # Group A: 3 signals within 2s
            _make_signal(ModalitySource.VISION, "face_absent", timestamp=t),
            _make_signal(ModalitySource.AUDIO, "speech",
                         timestamp=t + 0.5),
            _make_signal(ModalitySource.BROWSER, "tab_switch",
                         timestamp=t + 1.0),
            # Group B: 1 signal 10s later
            _make_signal(ModalitySource.MOUSE, "rapid_movement",
                         timestamp=t + 10.0),
        ]
        result = compute_corroboration(sigs, temporal_window=5.0)
        assert result.signal_count == 4
        assert result.independent_modalities == 3
        assert len(result.correlated_group) == 3  # primary group is largest
