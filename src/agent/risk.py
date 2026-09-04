"""Risk engine — multi-signal evidence fusion for cheating risk scoring."""

from typing import Dict, List, Optional
import numpy as np

from src.agent.memory import EvidenceMemory
from src.ml.logistic import LogisticRegressionScratch


class RiskEngine:
    """Fuses multi-signal evidence into a unified risk score.

    Combines:
    - Event frequency from evidence memory
    - Logistic regression classification
    - Anomaly detection scores
    - Multi-signal convergence check

    Risk formula: R_t = alpha * R_{t-1} + (1 - alpha) * E_t
    where E_t is the current evidence intensity.
    """

    # Event severity weights (math-grounded, not arbitrary)
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
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.alpha = self.config.get("alpha_decay", 0.8)
        self.escalation_threshold = self.config.get("escalation_threshold", 0.7)
        self.review_threshold = self.config.get("review_threshold", 0.5)
        self.multi_signal_required = self.config.get("multi_signal_required", 2)

        self._logistic_model: Optional[LogisticRegressionScratch] = None
        self._anomaly_model = None  # Set externally if needed

    def set_logistic_model(self, model: LogisticRegressionScratch) -> None:
        """Attach a trained logistic regression model."""
        self._logistic_model = model

    def set_anomaly_model(self, model) -> None:
        """Attach a trained anomaly detector."""
        self._anomaly_model = model

    def compute_risk(self, memory: EvidenceMemory) -> float:
        """Compute overall risk score from evidence memory.

        Returns float in [0, 1].
        """
        # Get current decayed risk from memory
        base_risk = memory.get_current_risk()

        # Get event counts and signal summary
        signal_summary = memory.get_signal_summary()
        event_counts = memory.get_event_counts()

        # Compute severity-weighted event score
        severity_score = 0.0
        for event_type, count in event_counts.items():
            weight = self.EVENT_WEIGHTS.get(event_type, 0.1)
            severity_score += weight * min(count, 5)  # Cap at 5 occurrences

        severity_score = min(severity_score / 2.0, 1.0)  # Normalize to [0, 1]

        # Multi-signal convergence bonus
        active_signals = memory.count_active_signals(threshold=1)
        convergence_factor = min(active_signals / 5.0, 1.0)  # Up to 5 signal types

        # Combine: weighted average of base risk, severity, and convergence
        evidence_score = (
            0.40 * base_risk +
            0.35 * severity_score +
            0.25 * convergence_factor
        )

        # Apply logistic model if available for refinement
        if self._logistic_model is not None:
            features = self._build_feature_vector(event_counts, signal_summary)
            if features is not None and features.sum() > 0:
                try:
                    ml_proba = self._logistic_model.predict_proba(features.reshape(1, -1))[0]
                    evidence_score = 0.60 * evidence_score + 0.40 * ml_proba
                except Exception:
                    pass

        return min(max(evidence_score, 0.0), 1.0)

    def _build_feature_vector(self, event_counts: dict,
                              signal_summary: dict) -> Optional[np.ndarray]:
        """Build feature vector for ML model from event counts."""
        features = []

        # Signal category features (5)
        for signal in ['vision', 'audio', 'keyboard', 'mouse', 'temporal']:
            features.append(signal_summary.get(signal, 0))

        # Individual high-impact events
        for event in ['face_absent', 'multiple_faces', 'copy_paste',
                      'speech_detected', 'fullscreen_exit']:
            features.append(event_counts.get(event, 0))

        return np.array(features) if features else None

    def compute_risk_decay(self, previous_risk: float, new_evidence: float) -> float:
        """Apply risk decay formula: R_t = alpha * R_{t-1} + (1-alpha) * E_t"""
        return self.alpha * previous_risk + (1 - self.alpha) * new_evidence

    def classify_risk_level(self, risk_score: float) -> str:
        """Classify numeric risk score into a risk level string."""
        if risk_score >= self.escalation_threshold:
            return "HIGH_RISK"
        elif risk_score >= self.review_threshold:
            return "SUSPICIOUS"
        return "NORMAL"

    def is_escalation_required(self, risk_score: float,
                                memory: EvidenceMemory) -> bool:
        """Check if escalation is warranted based on risk and signal convergence."""
        if risk_score >= self.escalation_threshold:
            return True
        if risk_score >= self.review_threshold:
            active = memory.count_active_signals()
            return active >= self.multi_signal_required
        return False
