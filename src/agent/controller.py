"""Autonomous agent controller — the full OBSERVE→PERCEIVE→FEATUREIZE→PREDICT→REASON→DECIDE→ACT loop."""

import time
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum

from src.agent.risk import RiskEngine
from src.agent.memory import EvidenceMemory
from src.features.behavioral import BehavioralFeatureExtractor
from src.features.temporal import TemporalFeatureAggregator
from src.ml.logistic import LogisticRegressionScratch
from src.ml.anomaly import AnomalyDetector
from src.trace import IntegrityDecisionTrace, EvidenceContribution, CounterfactualResult, build_explainability_trace, build_trace_id
from src.baseline import StudentBaseline
from src.context import ExamContext


class AgentPhase(Enum):
    """Phases of the autonomous agent loop."""
    OBSERVE = "OBSERVE"
    PERCEIVE = "PERCEIVE"
    FEATUREIZE = "FEATUREIZE"
    PREDICT = "PREDICT"
    REASON = "REASON"
    DECIDE = "DECIDE"
    ACT = "ACT"


class AgentState(Enum):
    """Agent risk states."""
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    TECHNICAL_EVENT = "TECHNICAL_EVENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ProctoringAgent:
    """Autonomous agent that runs the full proctoring loop.

    Loop: OBSERVE → PERCEIVE → FEATUREIZE → PREDICT → REASON → DECIDE → ACT → repeat
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.session_id = str(uuid.uuid4())[:8]

        # Subsystems
        self.risk_engine = RiskEngine(self.config.get("risk", {}))
        self.evidence_memory = EvidenceMemory(
            window_seconds=self.config.get("risk", {}).get("evidence_window_seconds", 300),
            alpha=self.config.get("risk", {}).get("alpha_decay", 0.8),
        )
        self.feature_extractor = BehavioralFeatureExtractor()
        self.temporal_aggregator = TemporalFeatureAggregator(
            window_size=self.config.get("temporal_window", 30)
        )

        # ML models (set externally after training)
        self._logistic_model: Optional[LogisticRegressionScratch] = None
        self._anomaly_model: Optional[AnomalyDetector] = None
        self.risk_engine.set_logistic_model(self._logistic_model)

        # State
        self._state = AgentState.NORMAL
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_interval = self.config.get("loop_interval_seconds", 5)
        self._iteration = 0

        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_alert: Optional[Callable] = None

        # History for analysis
        self._history: List[Dict[str, Any]] = []

        # Last decision trace (for explainability)
        self._last_trace: Optional[Dict[str, Any]] = None

    def set_models(self, logistic: LogisticRegressionScratch = None,
                   anomaly: AnomalyDetector = None) -> None:
        """Attach trained ML models."""
        if logistic:
            self._logistic_model = logistic
            self.risk_engine.set_logistic_model(logistic)
        if anomaly:
            self._anomaly_model = anomaly
            self.risk_engine.set_anomaly_model(anomaly)

    def set_callbacks(self, on_state_change: Callable = None,
                      on_alert: Callable = None) -> None:
        """Set event callbacks."""
        self._on_state_change = on_state_change
        self._on_alert = on_alert

    # ── Public API ──────────────────────────────────────────────────────

    def start_loop(self) -> None:
        """Start the autonomous proctoring loop in a background thread."""
        self._running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def stop_loop(self) -> None:
        """Stop the proctoring loop."""
        self._running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=3)

    def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single batch of sensor signals through the full loop.

        Args:
            signals: Raw signals from all sensors

        Returns:
            Processing results including risk score and state
        """
        self._iteration += 1
        result = {
            "iteration": self._iteration,
            "timestamp": datetime.now().isoformat(),
            "phase": None,
            "risk_score": 0.0,
            "state": self._state.value,
            "events_detected": [],
            "features": None,
            "ml_predictions": {},
        }

        # PHASE 1: OBSERVE — signals arrive as input

        # PHASE 2: PERCEIVE — extract structured events from raw signals
        events = self._perceive(signals)
        result["events_detected"] = events

        # PHASE 3: FEATUREIZE — convert events to feature vector
        features = self._featureize(signals)
        result["features"] = features.tolist() if features is not None else None

        # PHASE 4: PREDICT — run ML models
        predictions = self._predict(features, signals)
        result["ml_predictions"] = predictions

        # PHASE 5: REASON — combine all evidence
        evidence_score = self._reason(events, features, predictions)
        result["risk_score"] = round(evidence_score, 4)

        # PHASE 6: DECIDE — determine state and escalation
        new_state = self._decide(evidence_score, events)
        result["state"] = new_state.value

        # PHASE 7: ACT — record evidence, trigger callbacks
        actions = self._act(events, evidence_score, new_state)
        result["actions"] = actions

        # Update temporal features
        self.temporal_aggregator.add(features) if features is not None else None

        # Store history
        self._history.append(result)

        # Build explainability trace
        try:
            self._last_trace = build_explainability_trace(
                session_id=self.session_id,
                events=events,
                risk_result=result,
                risk_engine=self.risk_engine,
                evidence_memory=self.evidence_memory,
                agent_state=self._state.value,
            ).to_dict()
        except Exception:
            self._last_trace = None

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "session_id": self.session_id,
            "state": self._state.value,
            "iteration": self._iteration,
            "running": self._running,
            "evidence_events": len(self.evidence_memory._entries),
            "current_risk": round(self.evidence_memory.get_current_risk(), 4),
            "history_length": len(self._history),
        }

    def get_last_trace(self) -> Optional[Dict[str, Any]]:
        """Get the last explainability decision trace."""
        return self._last_trace

    def get_history(self) -> List[Dict[str, Any]]:
        """Get processing history."""
        return list(self._history)

    def reset(self) -> None:
        """Reset agent state for a new session."""
        self.evidence_memory.clear()
        self.temporal_aggregator._window.clear()
        self._state = AgentState.NORMAL
        self._iteration = 0
        self._history.clear()

    # ── Private: Loop Implementation ────────────────────────────────────

    def _run_loop(self) -> None:
        """Background loop for continuous proctoring."""
        while self._running:
            # In a real system, sensors would be polled here
            # For now, the loop just maintains the timer
            time.sleep(self._loop_interval)

    # ── Private: Phase Implementations ──────────────────────────────────

    def _perceive(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """PERCEIVE: Extract structured events from raw signals."""
        events = []

        # Vision events
        if not signals.get("face_present", True):
            events.append({
                "type": "face_absent",
                "confidence": signals.get("face_confidence", 0.0),
                "source": "vision",
            })
        face_count = signals.get("face_count", 1)
        if face_count > 1:
            events.append({
                "type": "multiple_faces",
                "confidence": 0.8,
                "source": "vision",
            })

        head = signals.get("head_pose", {}) or {}
        if abs(head.get("yaw", 0)) > 45:
            events.append({
                "type": "head_turned",
                "confidence": 0.7,
                "source": "vision",
            })

        # Audio events
        audio = signals.get("audio_features", {}) or {}
        if audio.get("volume_db", -100) > -30:
            events.append({
                "type": "loud_audio",
                "confidence": 0.5,
                "source": "audio",
            })

        # Keyboard events
        kb_idle = signals.get("keyboard_idle_seconds", 0)
        if kb_idle > 120:
            events.append({
                "type": "idle_keyboard",
                "confidence": 0.3,
                "source": "keyboard",
            })
        if signals.get("paste_detected", False):
            events.append({
                "type": "paste_detected",
                "confidence": 0.9,
                "source": "keyboard",
            })

        # Mouse events
        mouse_idle = signals.get("mouse_idle_seconds", 0)
        if mouse_idle > 120:
            events.append({
                "type": "idle_mouse",
                "confidence": 0.3,
                "source": "mouse",
            })

        # Temporal events
        if signals.get("tab_switch", False):
            events.append({
                "type": "tab_switch",
                "confidence": 0.6,
                "source": "temporal",
            })

        return events

    def _featureize(self, signals: Dict[str, Any]) -> Optional[Any]:
        """FEATUREIZE: Convert signals to ML feature vector."""
        try:
            return self.feature_extractor.extract(signals)
        except Exception:
            return None

    def _predict(self, features: Optional[Any],
                 signals: Dict[str, Any]) -> Dict[str, Any]:
        """PREDICT: Run ML models on features."""
        predictions = {
            "logistic_risk": None,
            "anomaly_score": None,
            "anomaly_detected": False,
        }

        # Logistic regression prediction
        if self._logistic_model is not None and features is not None:
            try:
                proba = self._logistic_model.predict_proba(features.reshape(1, -1))[0]
                predictions["logistic_risk"] = float(proba)
            except Exception:
                pass

        # Anomaly detection
        if self._anomaly_model is not None and features is not None:
            try:
                score = self._anomaly_model.score(features.reshape(1, -1))
                predictions["anomaly_score"] = float(score[0])
                predictions["anomaly_detected"] = bool(score[0] > 0.5)
            except Exception:
                pass

        return predictions

    def _reason(self, events: List[Dict], features: Optional[Any],
                predictions: Dict) -> float:
        """REASON: Combine all evidence into unified risk score."""
        # Add events to memory
        for evt in events:
            self.evidence_memory.add_event(evt["type"], evt["confidence"])

        # Compute risk from risk engine
        risk = self.risk_engine.compute_risk(self.evidence_memory)

        # Fuse with ML predictions
        logistic = predictions.get("logistic_risk")
        if logistic is not None:
            risk = 0.55 * risk + 0.45 * logistic

        anomaly = predictions.get("anomaly_score")
        if anomaly is not None and anomaly > 0.5:
            risk = min(risk + 0.1 * anomaly, 1.0)

        return min(max(risk, 0.0), 1.0)

    def _decide(self, risk_score: float,
                events: List[Dict]) -> AgentState:
        """DECIDE: Determine agent state based on risk score.

        State progression:
        NORMAL → MONITOR → SUSPICIOUS → HIGH_RISK → REVIEW_REQUIRED
        TECHNICAL_EVENT for sensor failures (separate from integrity)
        INSUFFICIENT_EVIDENCE when signals are unreliable
        """
        thresholds = self.config.get("risk", {})
        escalation = thresholds.get("escalation_threshold", 0.7)
        review = thresholds.get("review_threshold", 0.5)
        multi_signal = thresholds.get("multi_signal_required", 2)

        # Check for technical events first
        technical_events = [e for e in events
                           if e.get("type") in (
                               "camera_disconnected", "audio_unavailable",
                               "network_interruption", "browser_crash",
                           )]
        if technical_events and not events:
            return AgentState.TECHNICAL_EVENT

        # Check signal quality
        active_signals = self.evidence_memory.count_active_signals()

        if active_signals == 0 and events:
            return AgentState.INSUFFICIENT_EVIDENCE

        new_state = AgentState.NORMAL
        if risk_score >= escalation:
            new_state = AgentState.HIGH_RISK
        elif risk_score >= review:
            if active_signals >= multi_signal:
                new_state = AgentState.SUSPICIOUS
            else:
                new_state = AgentState.MONITOR

        # State transition
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            if self._on_state_change:
                self._on_state_change(old_state.value, new_state.value)

        return new_state

    def _act(self, events: List[Dict], risk_score: float,
             state: AgentState) -> List[str]:
        """ACT: Execute decisions — record evidence, trigger alerts."""
        actions = []

        # Log critical events to evidence
        for evt in events:
            self.evidence_memory.add_event(evt["type"], evt["confidence"])
            actions.append(f"logged:{evt['type']}")

        # Alert on state change
        if state in (AgentState.HIGH_RISK, AgentState.REVIEW_REQUIRED):
            if self._on_alert:
                self._on_alert(state.value, risk_score, events)
            actions.append(f"alert:{state.value}")

        if state == AgentState.HIGH_RISK:
            actions.append("escalated")
        elif state == AgentState.SUSPICIOUS:
            actions.append("monitoring")

        return actions
