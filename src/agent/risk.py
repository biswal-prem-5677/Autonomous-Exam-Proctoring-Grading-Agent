"""Upgraded risk engine with context awareness, personal baseline,
and cross-modal corroboration.

Risk formula:
    R_t = α·R_{t-1} + (1-α)·[ Σ(w_i · e_i · c_i · w_ctx_i · w_base_i) + λ·C_t ]

where:
    e_i  = event severity
    w_i  = modality reliability weight
    c_i  = event confidence
    w_ctx_i  = context adjustment weight
    w_base_i = personal baseline deviation weight
    C_t  = corroboration score (0.0–1.0)
    λ    = corroboration strength multiplier
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from src.agent.memory import EvidenceMemory
from src.ml.logistic import LogisticRegressionScratch
from src.ml.anomaly import AnomalyDetector
from src.context import ExamContext, ExamPolicy, compute_context_weight
from src.baseline import StudentBaseline
from src.fusion import compute_corroboration, CorroborationSignal, ModalitySource


class RiskEngine:
    """Fuses multi-signal evidence into a unified risk score.

    Combines:
    - Event frequency from evidence memory (temporal decay)
    - Logistic regression classification
    - Anomaly detection scores
    - Cross-modal corroboration (new)
    - Context-aware weighting (new)
    - Personal baseline deviation (new)

    Risk formula: R_t = α·R_{t-1} + (1-α)·[Σw_i·e_i·c_i + λ·C_t + β·B_t]
    """

    # Base event severity weights
    EVENT_WEIGHTS: Dict[str, float] = {
        # Vision signals
        "face_absent": 0.30,
        "multiple_faces": 0.50,
        "head_turned": 0.20,
        "gaze_away": 0.15,
        # Audio signals
        "loud_audio": 0.10,
        "speech_detected": 0.35,
        # Keyboard signals
        "rapid_typing": 0.15,
        "idle_keyboard": 0.05,
        "copy_paste": 0.40,
        "paste_detected": 0.40,
        # Mouse signals
        "rapid_movement": 0.10,
        "idle_mouse": 0.05,
        "right_click_spam": 0.20,
        # Temporal / interface signals
        "tab_switch": 0.25,
        "window_blur": 0.15,
        "fullscreen_exit": 0.35,
        # Technical events (lower weight, separate tracking)
        "camera_disconnected": 0.05,
        "camera_restored": 0.0,
        "audio_unavailable": 0.02,
        "network_interruption": 0.03,
        "browser_crash": 0.03,
    }

    # Modality-to-source mapping for corroboration
    MODALITY_MAP: Dict[str, ModalitySource] = {
        "face_absent": ModalitySource.VISION,
        "multiple_faces": ModalitySource.VISION,
        "head_turned": ModalitySource.VISION,
        "gaze_away": ModalitySource.VISION,
        "speech_detected": ModalitySource.AUDIO,
        "loud_audio": ModalitySource.AUDIO,
        "copy_paste": ModalitySource.KEYBOARD,
        "paste_detected": ModalitySource.KEYBOARD,
        "rapid_typing": ModalitySource.KEYBOARD,
        "idle_keyboard": ModalitySource.KEYBOARD,
        "rapid_movement": ModalitySource.MOUSE,
        "idle_mouse": ModalitySource.MOUSE,
        "right_click_spam": ModalitySource.MOUSE,
        "tab_switch": ModalitySource.BROWSER,
        "window_blur": ModalitySource.BROWSER,
        "fullscreen_exit": ModalitySource.BROWSER,
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.alpha = self.config.get("alpha_decay", 0.8)
        self.escalation_threshold = self.config.get("escalation_threshold", 0.7)
        self.review_threshold = self.config.get("review_threshold", 0.5)
        self.multi_signal_required = self.config.get("multi_signal_required", 2)
        self.corroboration_lambda = self.config.get("corroboration_lambda", 0.25)
        self.baseline_beta = self.config.get("baseline_beta", 0.15)
        self.temporal_window = self.config.get("temporal_window_seconds", 5.0)

        self._logistic_model: Optional[LogisticRegressionScratch] = None
        self._anomaly_model: Optional[AnomalyDetector] = None

    def set_logistic_model(self, model: LogisticRegressionScratch) -> None:
        self._logistic_model = model

    def set_anomaly_model(self, model) -> None:
        self._anomaly_model = model

    def compute_risk(
        self,
        memory: EvidenceMemory,
        context: Optional[ExamContext] = None,
        baseline: Optional[StudentBaseline] = None,
    ) -> float:
        """Compute overall risk score from all evidence sources.

        Args:
            memory: Evidence memory with temporal decay.
            context: Optional exam context for context-aware weighting.
            baseline: Optional student baseline for personal anomaly detection.

        Returns:
            Risk score in [0.0, 1.0].
        """
        base_risk = memory.get_current_risk()
        event_counts = memory.get_event_counts()

        # Gather corroboration signals
        now = time.time() if 'time' in dir() else 0
        import time as _time
        now = _time.time()

        corrob_signals = []
        for event_type, count in event_counts.items():
            if count == 0:
                continue
            modality = self.MODALITY_MAP.get(event_type)
            if modality is None:
                continue
            recent = memory.get_recent_by_type(event_type)
            if recent:
                latest = recent[-1]
                corrob_signals.append(CorroborationSignal(
                    modality=modality,
                    event_type=event_type,
                    severity=self.EVENT_WEIGHTS.get(event_type, 0.1),
                    confidence=latest.confidence,
                    timestamp=latest.timestamp,
                ))

        # Cross-modal corroboration
        corrob_result = compute_corroboration(
            corrob_signals, temporal_window=self.temporal_window
        ) if corrob_signals else None
        corrob_score = corrob_result.corroboration_score if corrob_result else 0.0

        # Compute weighted evidence score with context + baseline
        weighted_sum = 0.0
        total_weight = 0.0

        for event_type, count in event_counts.items():
            weight = self.EVENT_WEIGHTS.get(event_type, 0.1)
            capped_count = min(count, 5)
            severity = weight * capped_count

            # Context weight
            modality_name = self._modality_name(event_type)
            ctx_weight = compute_context_weight(
                event_type, context, modality_name
            ) if context else 1.0

            # Baseline deviation weight
            base_dev = self._baseline_deviation(event_type, baseline) if baseline else 0.0

            # Combined weight: severity * context * (1 + baseline deviation)
            event_contribution = severity * ctx_weight * (1.0 + base_dev)
            weighted_sum += event_contribution
            total_weight += weight * capped_count if (weight * capped_count) > 0 else 1.0

        # Normalize
        evidence_score = min(weighted_sum / 2.0, 1.0) if total_weight > 0 else 0.0

        # Add corroboration boost
        evidence_score = evidence_score + self.corroboration_lambda * corrob_score

        # Add baseline anomaly boost
        if baseline and baseline.is_ready():
            max_dev = baseline.max_deviation({})
            evidence_score = evidence_score + self.baseline_beta * max_dev

        # Temporal decay
        decayed_risk = self.alpha * base_risk + (1 - self.alpha) * evidence_score

        # ML refinement
        if self._logistic_model is not None:
            features = self._build_feature_vector(event_counts,
                                                   memory.get_signal_summary())
            if features is not None and features.sum() > 0:
                try:
                    ml_proba = self._logistic_model.predict_proba(
                        features.reshape(1, -1)
                    )[0]
                    decayed_risk = 0.55 * decayed_risk + 0.45 * ml_proba
                except Exception:
                    pass

        # Anomaly boost
        if self._anomaly_model is not None:
            features = self._build_feature_vector(event_counts,
                                                   memory.get_signal_summary())
            if features is not None and features.sum() > 0:
                try:
                    score = self._anomaly_model.score(features.reshape(1, -1))[0]
                    if score > 0.5:
                        decayed_risk = min(decayed_risk + 0.1 * score, 1.0)
                except Exception:
                    pass

        return min(max(decayed_risk, 0.0), 1.0)

    def _modality_name(self, event_type: str) -> str:
        """Map event type to modality name for context weighting."""
        mapping = {
            "face_absent": "vision", "multiple_faces": "vision",
            "head_turned": "vision", "gaze_away": "vision",
            "speech_detected": "audio", "loud_audio": "audio",
            "rapid_typing": "keyboard", "idle_keyboard": "keyboard",
            "copy_paste": "keyboard", "paste_detected": "keyboard",
            "rapid_movement": "mouse", "idle_mouse": "mouse",
            "tab_switch": "browser", "window_blur": "browser",
            "fullscreen_exit": "browser",
        }
        return mapping.get(event_type, "vision")

    def _baseline_deviation(self, event_type: str,
                            baseline: StudentBaseline) -> float:
        """Compute baseline deviation contribution for an event type."""
        if baseline is None or not baseline.is_ready():
            return 0.0
        if event_type in ("face_absent", "multiple_faces"):
            dev, _ = baseline.deviation_face_presence(0.0)
            return dev
        if event_type in ("rapid_movement", "idle_mouse"):
            dev, _ = baseline.deviation_mouse(0.0)
            return dev
        return 0.0

    def _build_feature_vector(self, event_counts: dict,
                              signal_summary: dict) -> Optional[np.ndarray]:
        """Build feature vector for ML model."""
        features = []
        for signal in ['vision', 'audio', 'keyboard', 'mouse', 'temporal']:
            features.append(signal_summary.get(signal, 0))
        for event in ['face_absent', 'multiple_faces', 'copy_paste',
                      'speech_detected', 'fullscreen_exit']:
            features.append(event_counts.get(event, 0))
        return np.array(features) if features else None

    def compute_risk_decay(self, previous_risk: float, new_evidence: float) -> float:
        """Apply risk decay: R_t = alpha * R_{t-1} + (1-alpha) * E_t"""
        return self.alpha * previous_risk + (1 - self.alpha) * new_evidence

    def classify_risk_level(self, risk_score: float) -> str:
        """Classify risk score into level string."""
        if risk_score >= self.escalation_threshold:
            return "HIGH_RISK"
        elif risk_score >= self.review_threshold:
            return "SUSPICIOUS"
        return "NORMAL"

    def is_escalation_required(self, risk_score: float,
                                memory: EvidenceMemory) -> bool:
        """Check if escalation is warranted."""
        if risk_score >= self.escalation_threshold:
            return True
        if risk_score >= self.review_threshold:
            active = memory.count_active_signals()
            return active >= self.multi_signal_required
        return False

    def get_risk_color(self, risk_score: float) -> str:
        """Get display color class for risk meter."""
        if risk_score >= self.escalation_threshold:
            return "danger"
        elif risk_score >= self.review_threshold:
            return "warning"
        return "success"
