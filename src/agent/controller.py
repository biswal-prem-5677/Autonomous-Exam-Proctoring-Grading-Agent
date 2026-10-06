"""Autonomous agent controller — the full OBSERVE→PERCEIVE→FEATUREIZE→PREDICT→REASON→DECIDE→ACT loop."""

import time
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum

from src.agent.risk import RiskEngine
from src.agent.memory import EvidenceMemory
from src.features.behavioral import BehavioralFeatureExtractor
from src.features.temporal import TemporalFeatureAggregator
from src.ml.logistic import LogisticRegressionScratch
from src.ml.anomaly import AnomalyDetector
from src.trace import build_explainability_trace
from src.trace.audit_chain import AuditChain
from src.agent.technical import TechnicalEventEngine
from src.agent.uncertainty import UncertaintyEngine
from src.agent.counterfactual import CounterfactualEngine
from src.baseline import StudentBaseline, BehavioralSample
from src.context import ExamContext, QuestionContext, get_default_policy, compute_context_weight
from src.fusion import CrossModalCorroborationEngine


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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        self.technical_engine = TechnicalEventEngine(
            grace_period_seconds=self.config.get("technical_grace_period", 60.0)
        )
        self.uncertainty_engine = UncertaintyEngine(
            action_confidence_threshold=self.config.get("action_confidence_threshold", 0.65)
        )
        self.counterfactual_engine = CounterfactualEngine(
            review_threshold=self.config.get("risk", {}).get("review_threshold", 0.50)
        )
        self.corroboration_engine = CrossModalCorroborationEngine()
        self.baseline = self.config.get("baseline") or StudentBaseline(
            student_id=self.config.get("student_id", "student_1")
        )
        self.context = self.config.get("context") or ExamContext(
            policy=self.config.get("policy", get_default_policy("standard"))
        )
        self.audit_chain = AuditChain(
            db_path=self.config.get("audit_db", "data/audit_chain.db")
        )


        # ML models (set externally after training)
        self._logistic_model: Optional[LogisticRegressionScratch] = None
        self._anomaly_model: Optional[AnomalyDetector] = None

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

    def set_models(self, logistic: Optional[LogisticRegressionScratch] = None,
                   anomaly: Optional[AnomalyDetector] = None) -> None:
        """Attach trained ML models."""
        if logistic:
            self._logistic_model = logistic
            self.risk_engine.set_logistic_model(logistic)
        if anomaly:
            self._anomaly_model = anomaly
            self.risk_engine.set_anomaly_model(anomaly)

    def set_callbacks(self, on_state_change: Optional[Callable] = None,
                      on_alert: Optional[Callable] = None) -> None:
        """Set event callbacks."""
        self._on_state_change = on_state_change
        self._on_alert = on_alert

    def set_baseline(self, baseline: StudentBaseline) -> None:
        """Attach personal student behavioral baseline."""
        self.baseline = baseline

    def set_context(self, context: ExamContext) -> None:
        """Attach active exam context & policy."""
        self.context = context

    def set_question_context(self, question: QuestionContext) -> None:
        """Update current question context for context-sensitive risk weighting."""
        self.context.current_question = question

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
            Processing results including risk score, state, uncertainty, counterfactuals, and baseline
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
            "uncertainty": None,
            "counterfactuals": [],
            "technical_incidents": [],
            "baseline": {},
        }

        # PHASE 1: OBSERVE & BASELINE UPDATE
        # Continually update personal baseline from passive continuous behavioral signals
        if any(k in signals for k in ("typing_wpm", "mouse_velocity", "face_present_rate", "gaze_off_screen")):
            sample = BehavioralSample(
                timestamp=signals.get("detected_at", time.time()),
                typing_wpm=float(signals.get("typing_wpm", 0.0)),
                mouse_velocity=float(signals.get("mouse_velocity", 0.0)),
                mouse_clicks=int(signals.get("mouse_clicks", 0)),
                face_present=float(signals.get("face_present_rate", 1.0 if signals.get("face_present", True) else 0.0)),
                face_count=int(signals.get("face_count", 1)),
                gaze_off_screen_seconds=float(signals.get("gaze_off_screen", 0.0)),
                pause_duration_seconds=float(signals.get("pause_duration_seconds", 0.0)),
            )
            self.baseline.add_sample(sample)

        # PHASE 2: PERCEIVE — extract structured events from raw signals
        raw_events = self._perceive(signals)

        # Filter and isolate technical faults from student integrity violations
        system_diagnostics = {
            "camera_connected": signals.get("camera_connected", True),
            "camera_fps": signals.get("camera_fps", 30.0),
            "network_online": signals.get("network_online", True),
            "ping_ms": signals.get("ping_ms", 45.0),
            "mic_active": signals.get("mic_active", True),
            "cpu_percent": signals.get("cpu_percent", 15.0),
        }
        events, technical_incidents = self.technical_engine.analyze_system_state(system_diagnostics, raw_events)
        result["events_detected"] = events
        result["technical_incidents"] = [t.to_dict() for t in technical_incidents]

        # PHASE 3: FEATUREIZE — convert events to feature vector
        features = self._featureize(signals)
        result["features"] = features.tolist() if features is not None else None

        # PHASE 4: PREDICT — run ML models
        predictions = self._predict(features, signals)
        result["ml_predictions"] = predictions

        # PHASE 5: REASON — combine evidence, baseline deviations, uncertainty & counterfactuals
        evidence_score, uncertainty_estimate, counterfactual_analysis = self._reason(
            events, features, predictions, signals
        )
        result["risk_score"] = round(evidence_score, 4)
        result["uncertainty"] = uncertainty_estimate.to_dict() if uncertainty_estimate else None
        result["counterfactuals"] = [
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
            for f in (counterfactual_analysis.factors if counterfactual_analysis else [])
        ]
        result["baseline"] = {
            "maturity": self.baseline.maturity_score(),
            "is_ready": self.baseline.is_ready(),
            "max_deviation": round(self.baseline.max_deviation(signals), 4),
        }

        # PHASE 6: DECIDE — determine state, accounting for technical incidents and uncertainty
        new_state = self._decide(evidence_score, events, uncertainty_estimate, technical_incidents)
        result["state"] = new_state.value

        # PHASE 7: ACT — record evidence, trigger callbacks, commit to cryptographic audit chain
        actions = self._act(events, evidence_score, new_state, result)
        result["actions"] = actions

        # Update temporal features
        if features is not None:
            self.temporal_aggregator.add(features)

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
                real_context=self.context,
                real_baseline=self.baseline,
            ).to_dict()
        except Exception:
            self._last_trace = None

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status including the last explainability trace."""
        return {
            "session_id": self.session_id,
            "state": self._state.value,
            "iteration": self._iteration,
            "running": self._running,
            "evidence_events": len(self.evidence_memory._entries),
            "current_risk": round(self.evidence_memory.get_current_risk(), 4),
            "history_length": len(self._history),
            "last_trace": self._last_trace,
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
        """PERCEIVE: Extract structured events from raw signals.

        Each event carries a real timestamp from the source signal or
        the time of detection. No fabricated timestamps.
        """
        events = []
        detected_at = signals.get("detected_at", time.time())
        session_elapsed = signals.get("session_elapsed", 0)

        # Vision events
        if not signals.get("face_present", True):
            events.append({
                "type": "face_absent",
                "confidence": signals.get("face_confidence", 0.0),
                "source": "vision",
                "timestamp": detected_at,
                "session_elapsed": session_elapsed,
            })
        face_count = signals.get("face_count", 1)
        if face_count > 1:
            events.append({
                "type": "multiple_faces",
                "confidence": 0.8,
                "source": "vision",
                "timestamp": detected_at,
                "session_elapsed": session_elapsed,
            })

        head = signals.get("head_pose", {}) or {}
        if abs(head.get("yaw", 0)) > 45:
            events.append({
                "type": "head_turned",
                "confidence": 0.7,
                "source": "vision",
                "timestamp": detected_at,
                "session_elapsed": session_elapsed,
            })

        # Audio events
        audio = signals.get("audio_features", {}) or {}
        if audio.get("volume_db", -100) > -30:
            events.append({
                "type": "loud_audio",
                "confidence": 0.5,
                "source": "audio",
                "timestamp": detected_at,
                "session_elapsed": session_elapsed,
            })

        # Keyboard events
        kb_idle = signals.get("keyboard_idle_seconds", 0)
        if kb_idle > 120:
            events.append({
                "type": "idle_keyboard",
                "confidence": 0.3,
                "source": "keyboard",
                "timestamp": detected_at,
                "session_elapsed": session_elapsed,
            })
        if signals.get("paste_detected", False):
            events.append({
                "type": "paste_detected",
                "confidence": 0.9,
                "source": "keyboard",
                "timestamp": detected_at,
                "session_elapsed": session_elapsed,
            })

        # Mouse events
        mouse_idle = signals.get("mouse_idle_seconds", 0)
        if mouse_idle > 120:
            events.append({
                "type": "idle_mouse",
                "confidence": 0.3,
                "source": "mouse",
                "timestamp": detected_at,
                "session_elapsed": session_elapsed,
            })

        # Temporal events
        if signals.get("tab_switch", False):
            events.append({
                "type": "tab_switch",
                "confidence": 0.6,
                "source": "browser",
                "timestamp": detected_at,
                "session_elapsed": session_elapsed,
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
        predictions: Dict[str, Any] = {
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
                predictions: Dict, signals: Optional[Dict[str, Any]] = None) -> Tuple[float, Any, Any]:
        """REASON: Combine evidence, context, baseline deviations, uncertainty, and counterfactuals."""
        signals = signals or {}

        # Add events to memory with context-sensitive weighting
        for evt in events:
            source = evt.get("source", "vision")
            w = compute_context_weight(evt["type"], self.context, source)
            self.evidence_memory.add_event(evt["type"], evt["confidence"] * w)

        # Base risk from evidence memory
        risk = self.risk_engine.compute_risk(self.evidence_memory)

        # Personal baseline deviation
        baseline_deviations = self.baseline.compute_all_deviations(signals)
        max_dev = max((d[0] for d in baseline_deviations.values()), default=0.0)
        maturity = self.baseline.maturity_score()
        if maturity >= 0.4 and max_dev > 0.5:
            risk = min(risk + 0.15 * max_dev * maturity, 1.0)

        # Fuse with ML predictions
        logistic = predictions.get("logistic_risk")
        if logistic is not None:
            risk = 0.55 * risk + 0.45 * logistic

        anomaly = predictions.get("anomaly_score")
        if anomaly is not None and anomaly > 0.5:
            risk = min(risk + 0.1 * anomaly, 1.0)

        evidence_score = min(max(risk, 0.0), 1.0)

        # Uncertainty quantification
        observed_modalities = set()
        for evt in events:
            src = evt.get("source")
            if src:
                observed_modalities.add(src)
        if signals.get("face_present", True):
            observed_modalities.add("vision")
        if signals.get("mic_active", True):
            observed_modalities.add("audio")
        if signals.get("typing_wpm", 0) > 0:
            observed_modalities.add("keyboard")
        if signals.get("mouse_velocity", 0) > 0:
            observed_modalities.add("mouse")
        if signals.get("browser_active", True):
            observed_modalities.add("browser")

        signal_confidences = {
            evt.get("source", "general"): evt.get("confidence", 0.5)
            for evt in events
        }
        uncertainty_estimate = self.uncertainty_engine.evaluate(
            observed_modalities=observed_modalities,
            signal_confidences=signal_confidences,
            prediction_prob=predictions.get("logistic_risk"),
            corroboration_score=0.7 if len(observed_modalities) >= 2 else 0.3,
        )

        # Counterfactual causal attribution
        evidence_items = [
            {
                "factor_id": f"f_{i}",
                "source": evt.get("source", "general"),
                "event_type": evt.get("type", "unknown"),
                "contribution": evt.get("confidence", 0.5) * self.risk_engine.EVENT_WEIGHTS.get(evt.get("type", ""), 0.1),
            }
            for i, evt in enumerate(events)
        ]
        counterfactual_analysis = self.counterfactual_engine.analyze(
            session_id=self.session_id,
            current_risk=evidence_score,
            evidence_contributions=evidence_items,
        )

        return evidence_score, uncertainty_estimate, counterfactual_analysis

    def _decide(self, risk_score: float,
                events: List[Dict],
                uncertainty_estimate: Optional[Any] = None,
                technical_incidents: Optional[List[Any]] = None) -> AgentState:
        """DECIDE: Determine agent state based on risk score, uncertainty, and technical events.

        State progression:
        NORMAL → MONITOR → SUSPICIOUS → HIGH_RISK → REVIEW_REQUIRED
        TECHNICAL_EVENT for hardware/network faults (separate from integrity concerns)
        INSUFFICIENT_EVIDENCE when sensor reliability is low
        """
        thresholds = self.config.get("risk", {})
        escalation = thresholds.get("escalation_threshold", 0.7)
        review = thresholds.get("review_threshold", 0.5)
        multi_signal = thresholds.get("multi_signal_required", 2)

        # 1. Check for technical incidents first — isolated from integrity violations
        if technical_incidents and not events:
            return self._transition_state(AgentState.TECHNICAL_EVENT)

        # 2. Check signal quality and uncertainty
        active_signals = self.evidence_memory.count_active_signals()
        if uncertainty_estimate and not uncertainty_estimate.is_actionable:
            # Epistemic/aleatoric uncertainty is high — do not prematurely escalate
            if risk_score >= review:
                return self._transition_state(AgentState.MONITOR)
            return self._transition_state(AgentState.INSUFFICIENT_EVIDENCE if events else AgentState.NORMAL)

        if active_signals == 0 and events:
            return self._transition_state(AgentState.INSUFFICIENT_EVIDENCE)

        # 3. Risk-based escalation
        if risk_score >= escalation:
            new_state = AgentState.HIGH_RISK
        elif risk_score >= review:
            if active_signals >= multi_signal:
                new_state = AgentState.SUSPICIOUS
            else:
                new_state = AgentState.MONITOR
        else:
            new_state = AgentState.NORMAL

        return self._transition_state(new_state)

    def _transition_state(self, new_state: AgentState) -> AgentState:
        """Execute state transition and trigger callback if state changed."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            if self._on_state_change:
                self._on_state_change(old_state.value, new_state.value)
        return new_state

    def _act(self, events: List[Dict], risk_score: float,
             state: AgentState, result_payload: Optional[Dict[str, Any]] = None) -> List[str]:
        """ACT: Execute decisions — record evidence, trigger alerts, commit to audit chain."""
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
        elif state == AgentState.TECHNICAL_EVENT:
            actions.append("technical_isolation_active")

        # Cryptographically commit event and decision to immutable audit chain
        try:
            audit_data = {
                "risk_score": round(risk_score, 4),
                "state": state.value,
                "events_count": len(events),
                "actions": actions,
            }
            if result_payload:
                if result_payload.get("uncertainty"):
                    audit_data["uncertainty"] = result_payload["uncertainty"]
                if result_payload.get("baseline"):
                    audit_data["baseline"] = result_payload["baseline"]
                if result_payload.get("technical_incidents"):
                    audit_data["technical_incidents"] = result_payload["technical_incidents"]

            self.audit_chain.append(
                event_type=f"agent_decision_{state.value.lower()}",
                actor_id="autonomous_agent",
                session_id=self.session_id,
                data=audit_data,
            )
        except Exception:
            pass

        return actions

