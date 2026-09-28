"""Adversarial Red-Team Simulator and Calibration Engine.

Evaluates platform resilience against active attempts to evade proctoring:
    - Photo / video replay attacks
    - Virtual camera driver emulation
    - Synthetic / automated keystroke injection (macro/script)
    - Clipboard pasting / remote desktop injection
    - Second person / background whispers
    - High-frequency browser switching floods

Calculates research-grade evaluation metrics:
    - True Positive Rate (TPR / Sensitivity)
    - False Positive Rate (FPR / False Alarm Rate)
    - Expected Calibration Error (ECE)
    - Brier Score (mean squared probability error)
    - False Escalation Rate (FER) & False Intervention Rate (FIR)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum


class AttackType(Enum):
    NORMAL_BENIGN = "normal_benign"
    REPLAY_ATTACK = "replay_attack"
    VIRTUAL_CAMERA = "virtual_camera"
    SYNTHETIC_KEYSTROKES = "synthetic_keystrokes"
    CLIPBOARD_INJECTION = "clipboard_injection"
    SECOND_PERSON = "second_person"
    REMOTE_DESKTOP = "remote_desktop"
    AUDIO_WHISPER = "audio_whisper"


@dataclass
class RedTeamScenario:
    scenario_id: str
    attack_type: AttackType
    is_malicious: bool
    description: str
    synthetic_signals: List[Dict[str, Any]]


@dataclass
class CalibrationMetrics:
    brier_score: float
    expected_calibration_error: float   # ECE
    max_calibration_error: float        # MCE
    confidence_bins: List[Dict[str, float]]


@dataclass
class RedTeamBenchmarkReport:
    total_sessions: int
    malicious_sessions: int
    benign_sessions: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    tpr: float                          # Recall
    fpr: float                          # False Alarm Rate
    precision: float
    f1_score: float
    false_escalation_rate: float
    calibration: CalibrationMetrics
    attack_breakdown: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total": self.total_sessions,
                "tpr_recall": round(self.tpr, 4),
                "fpr_fallout": round(self.fpr, 4),
                "precision": round(self.precision, 4),
                "f1_score": round(self.f1_score, 4),
                "false_escalation_rate": round(self.false_escalation_rate, 4),
            },
            "calibration": {
                "brier_score": round(self.calibration.brier_score, 4),
                "ece": round(self.calibration.expected_calibration_error, 4),
                "mce": round(self.calibration.max_calibration_error, 4),
            },
            "matrix": {
                "tp": self.true_positives,
                "fp": self.false_positives,
                "tn": self.true_negatives,
                "fn": self.false_negatives,
            },
            "per_attack_accuracy": self.attack_breakdown,
        }


class AdversarialSimulator:
    """Generates synthetic adversarial exam sessions to stress-test the agent."""

    @staticmethod
    def generate_scenario(attack_type: AttackType, duration_ticks: int = 10) -> RedTeamScenario:
        signals = []
        is_malicious = attack_type != AttackType.NORMAL_BENIGN

        for t in range(duration_ticks):
            base_signal = {
                "session_elapsed": t * 5.0,
                "face_present": True,
                "face_confidence": 0.95,
                "face_count": 1,
                "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "audio_features": {"volume_db": -50.0, "speech_detected": False},
                "keyboard_idle_seconds": 2.0,
                "mouse_idle_seconds": 2.0,
                "tab_switch": False,
                "paste_detected": False,
            }

            if attack_type == AttackType.REPLAY_ATTACK:
                # Perfectly fixed landmarks, 0 natural tremor/micro-movement
                base_signal["landmark_variance"] = 0.0001
                base_signal["optical_flow_magnitude"] = 0.0
                if t >= 4:
                    base_signal["paste_detected"] = True

            elif attack_type == AttackType.VIRTUAL_CAMERA:
                base_signal["virtual_cam_flag"] = True
                base_signal["camera_fps"] = 60.0  # suspiciously constant
                if t >= 3:
                    base_signal["tab_switch"] = True

            elif attack_type == AttackType.SYNTHETIC_KEYSTROKES:
                # Mechanical constant typing rhythm: zero variance in inter-keystroke timing
                base_signal["keystroke_jitter_ms"] = 0.0
                base_signal["typing_wpm"] = 140.0
                base_signal["rapid_typing"] = True

            elif attack_type == AttackType.CLIPBOARD_INJECTION:
                if t == 5:
                    base_signal["paste_detected"] = True
                    base_signal["pasted_character_count"] = 750

            elif attack_type == AttackType.SECOND_PERSON:
                if t >= 3:
                    base_signal["face_count"] = 2
                    base_signal["audio_features"] = {"volume_db": -25.0, "speech_detected": True}

            elif attack_type == AttackType.REMOTE_DESKTOP:
                if t >= 4:
                    base_signal["mouse_teleport_detected"] = True
                    base_signal["window_blur"] = True

            elif attack_type == AttackType.AUDIO_WHISPER:
                if t >= 2:
                    base_signal["audio_features"] = {"volume_db": -38.0, "speech_detected": True}

            signals.append(base_signal)

        return RedTeamScenario(
            scenario_id=f"redteam_{attack_type.value}",
            attack_type=attack_type,
            is_malicious=is_malicious,
            description=f"Automated test scenario for {attack_type.value}",
            synthetic_signals=signals,
        )


class RedTeamHarness:
    """Runs adversarial benchmarks and measures calibration & detection metrics."""

    def __init__(self, escalation_threshold: float = 0.65):
        self.escalation_threshold = escalation_threshold

    def evaluate_agent(self, agent_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> RedTeamBenchmarkReport:
        """Run all attack scenarios and benign sessions through agent.

        Args:
            agent_fn: Function that accepts a signal dict and returns dict with 'risk_score' and 'state'
        """
        scenarios = [
            AdversarialSimulator.generate_scenario(AttackType.NORMAL_BENIGN),
            AdversarialSimulator.generate_scenario(AttackType.NORMAL_BENIGN),
            AdversarialSimulator.generate_scenario(AttackType.REPLAY_ATTACK),
            AdversarialSimulator.generate_scenario(AttackType.VIRTUAL_CAMERA),
            AdversarialSimulator.generate_scenario(AttackType.SYNTHETIC_KEYSTROKES),
            AdversarialSimulator.generate_scenario(AttackType.CLIPBOARD_INJECTION),
            AdversarialSimulator.generate_scenario(AttackType.SECOND_PERSON),
            AdversarialSimulator.generate_scenario(AttackType.REMOTE_DESKTOP),
            AdversarialSimulator.generate_scenario(AttackType.AUDIO_WHISPER),
        ]

        y_true = []
        y_prob = []
        y_pred = []
        attack_stats: Dict[str, List[bool]] = {}

        tp = fp = tn = fn = 0
        false_escalations = 0

        for scen in scenarios:
            attack_name = scen.attack_type.value
            if attack_name not in attack_stats:
                attack_stats[attack_name] = []

            max_risk = 0.0
            escalated = False

            for sig in scen.synthetic_signals:
                res = agent_fn(sig)
                risk = res.get("risk_score", 0.0)
                state = res.get("state", "NORMAL")
                max_risk = max(max_risk, risk)
                if state in ("HIGH_RISK", "REVIEW_REQUIRED") or risk >= self.escalation_threshold:
                    escalated = True

            predicted_malicious = escalated or (max_risk >= self.escalation_threshold)
            y_true.append(1 if scen.is_malicious else 0)
            y_prob.append(max_risk)
            y_pred.append(1 if predicted_malicious else 0)

            if scen.is_malicious:
                if predicted_malicious:
                    tp += 1
                    attack_stats[attack_name].append(True)
                else:
                    fn += 1
                    attack_stats[attack_name].append(False)
            else:
                if predicted_malicious:
                    fp += 1
                    false_escalations += 1
                    attack_stats[attack_name].append(False)
                else:
                    tn += 1
                    attack_stats[attack_name].append(True)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
        fer = false_escalations / max(1, (tn + fp))

        # Compute Calibration: Brier Score & ECE
        brier = float(np.mean([(p - y)**2 for p, y in zip(y_prob, y_true)]))
        ece, mce, bins = self._compute_calibration_curve(y_true, y_prob, n_bins=5)

        attack_breakdown = {}
        for att, outcomes in attack_stats.items():
            attack_breakdown[att] = {
                "accuracy": round(sum(outcomes) / len(outcomes), 2) if outcomes else 0.0,
                "trials": len(outcomes),
            }

        return RedTeamBenchmarkReport(
            total_sessions=len(scenarios),
            malicious_sessions=sum(y_true),
            benign_sessions=len(y_true) - sum(y_true),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            tpr=tpr,
            fpr=fpr,
            precision=prec,
            f1_score=f1,
            false_escalation_rate=fer,
            calibration=CalibrationMetrics(
                brier_score=brier,
                expected_calibration_error=ece,
                max_calibration_error=mce,
                confidence_bins=bins,
            ),
            attack_breakdown=attack_breakdown,
        )

    @staticmethod
    def _compute_calibration_curve(y_true: List[int], y_prob: List[float], n_bins: int = 5):
        """Compute Expected Calibration Error (ECE) across probability bins."""
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bins_data = []

        total = len(y_true)
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i+1]
            indices = [idx for idx, p in enumerate(y_prob) if low <= p < high or (i == n_bins - 1 and p == 1.0)]
            if not indices:
                continue

            bin_acc = np.mean([y_true[idx] for idx in indices])
            bin_conf = np.mean([y_prob[idx] for idx in indices])
            bin_weight = len(indices) / total
            diff = abs(bin_acc - bin_conf)

            ece += bin_weight * diff
            mce = max(mce, diff)

            bins_data.append({
                "bin_range": f"{low:.1f}-{high:.1f}",
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "count": len(indices),
            })

        return float(ece), float(mce), bins_data
