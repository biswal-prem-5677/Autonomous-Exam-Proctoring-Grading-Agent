"""
Keystroke Dynamics Module.

Implements behavioral biometrics based on typing patterns for:
    1. Keystroke timing feature extraction (hold time, inter-key latency)
    2. Typing rhythm profiling (mean, std, burst rate, idle detection)
    3. Keystroke anomaly detection (deviation from baseline profile)

Mathematical Foundation:
────────────────────────────────────────────────────────────────────────────
Feature Vector (per time window):
    K = [H_μ, H_σ, L_μ, L_σ, R_burst, R_idle, T_rate, R_consistency]

Where:
    H_i     = t_release,i − t_press,i      (hold time for key i)
    L_i     = t_press,i+1 − t_release,i    (inter-key latency)
    R_burst = N_keys_in_500ms / 500ms      (burst rate)
    R_idle  = I(Δt > 120s)                 (idle indicator)
    T_rate  = N_keys / window_duration      (typing rate)
    R_consistency = 1 − (σ_H / μ_H)        (hold-time consistency)

Anomaly Detection:
    z_i = (x_i − μ_i) / σ_i
    A = √(Σ z_i²)
    Flag: A > χ²(df=dim, p=0.95)
────────────────────────────────────────────────────────────────────────────
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import math


@dataclass
class KeystrokeEvent:
    """A single keystroke timing event."""
    key: str
    press_time: float
    release_time: Optional[float] = None
    hold_time: Optional[float] = None

    def finalize(self) -> None:
        if self.release_time is not None and self.hold_time is None:
            self.hold_time = self.release_time - self.press_time


@dataclass
class KeystrokeProfile:
    """Baseline typing profile for a user/session."""
    hold_time_mean: float = 0.12
    hold_time_std: float = 0.05
    latency_mean: float = 0.10
    latency_std: float = 0.04
    burst_rate_mean: float = 5.0
    burst_rate_std: float = 2.0
    typing_rate_mean: float = 3.5
    typing_rate_std: float = 1.2
    consistency_score: float = 0.75
    n_samples: int = 0


class KeystrokeDynamics:
    """Keystroke behavioral biometrics engine."""

    BURST_WINDOW_MS = 500
    IDLE_THRESHOLD_S = 120.0

    def __init__(self, window_size: int = 60):
        self._events: deque = deque(maxlen=2000)
        self._profile: Optional[KeystrokeProfile] = None
        self._window_size = window_size
        self._lock = threading.Lock()
        self._last_key_time: float = 0.0
        self._keys_in_window: deque = deque(maxlen=window_size)
        self._is_idle: bool = False

    # ─── Event Recording ──────────────────────────────────────────────────

    def record_key_press(self, key: str) -> None:
        """Record a key press event."""
        now = time.time()
        event = KeystrokeEvent(key=key, press_time=now)
        with self._lock:
            self._events.append(event)
        self._last_key_time = now
        self._is_idle = False
        self._keys_in_window.append(now)

    def record_key_release(self, key: str) -> None:
        """Record a key release event."""
        now = time.time()
        with self._lock:
            for event in reversed(self._events):
                if event.key == key and event.release_time is None:
                    event.release_time = now
                    event.hold_time = now - event.press_time
                    break

    # ─── Feature Extraction ───────────────────────────────────────────────

    def extract_features(self) -> dict:
        """Extract keystroke feature vector from recent events.

        Returns:
            dict with keys:
                hold_time_mean, hold_time_std,
                latency_mean, latency_std,
                burst_rate, idle_flag, typing_rate, consistency
        """
        with self._lock:
            events = list(self._events)

        if len(events) < 2:
            return self._default_features()

        # Hold times
        hold_times = [e.hold_time for e in events if e.hold_time is not None]
        # Inter-key latencies
        press_times = [e.press_time for e in events if e.press_time is not None]
        latencies = [
            press_times[i + 1] - press_times[i]
            for i in range(len(press_times) - 1)
            if press_times[i + 1] - press_times[i] > 0
        ]

        # Recent window stats
        now = time.time()
        recent_events = [e for e in events if now - e.press_time < 10.0]
        recent_presses = [e.press_time for e in recent_events if e.press_time]
        burst_rate = self._compute_burst_rate(recent_presses, now)
        typing_rate = self._compute_typing_rate(recent_presses, now)

        # Idle detection
        idle_flag = 1.0 if (now - self._last_key_time > self.IDLE_THRESHOLD_S) else 0.0
        if not recent_presses:
            idle_flag = 1.0

        # Consistency
        consistency = self._compute_consistency(hold_times)

        features = {
            "hold_time_mean": _safe_mean(hold_times, default=0.12),
            "hold_time_std": _safe_std(hold_times, default=0.05),
            "latency_mean": _safe_mean(latencies, default=0.10),
            "latency_std": _safe_std(latencies, default=0.04),
            "burst_rate": burst_rate,
            "idle_flag": idle_flag,
            "typing_rate": typing_rate,
            "consistency": consistency,
            "n_events": len(events),
            "n_hold_times": len(hold_times),
            "n_latencies": len(latencies),
        }
        return features

    def extract_feature_vector(self) -> List[float]:
        """Return features as a flat list for ML input."""
        f = self.extract_features()
        return [
            f["hold_time_mean"],
            f["hold_time_std"],
            f["latency_mean"],
            f["latency_std"],
            f["burst_rate"],
            f["idle_flag"],
            f["typing_rate"],
            f["consistency"],
        ]

    # ─── Baseline Profile ─────────────────────────────────────────────────

    def build_profile(self, events: List[KeystrokeEvent]) -> KeystrokeProfile:
        """Build a typing profile from a training session of normal behavior."""
        hold_times = [e.hold_time for e in events if e.hold_time is not None]
        presses = sorted([e.press_time for e in events if e.press_time is not None])
        latencies = [
            presses[i + 1] - presses[i]
            for i in range(len(presses) - 1)
            if presses[i + 1] - presses[i] > 0
        ]
        now = time.time()
        recent = [p for p in presses if now - p < 10.0]
        burst = self._compute_burst_rate(recent, now)
        rate = self._compute_typing_rate(recent, now)
        consistency = self._compute_consistency(hold_times)

        self._profile = KeystrokeProfile(
            hold_time_mean=_safe_mean(hold_times, 0.12),
            hold_time_std=_safe_std(hold_times, 0.05),
            latency_mean=_safe_mean(latencies, 0.10),
            latency_std=_safe_std(latencies, 0.04),
            burst_rate_mean=burst if burst > 0 else 5.0,
            burst_rate_std=max(burst * 0.3, 0.5),
            typing_rate_mean=rate if rate > 0 else 3.5,
            typing_rate_std=max(rate * 0.3, 0.5),
            consistency_score=consistency,
            n_samples=len(events),
        )
        return self._profile

    # ─── Anomaly Detection ────────────────────────────────────────────────

    def compute_anomaly_score(self, features: dict) -> float:
        """Compute anomaly score vs. baseline profile.

        Returns a scalar anomaly score:
            z_agg = sqrt(Σ z_i²) / sqrt(dimensions)

        Higher values → more anomalous.
        """
        if self._profile is None:
            return 0.0

        p = self._profile
        z_scores = []

        comparisons = [
            (features.get("hold_time_mean", 0.12), p.hold_time_mean, max(p.hold_time_std, 0.01)),
            (features.get("hold_time_std", 0.05), p.hold_time_std, max(p.hold_time_std * 0.5, 0.005)),
            (features.get("latency_mean", 0.10), p.latency_mean, max(p.latency_std, 0.01)),
            (features.get("latency_std", 0.04), p.latency_std, max(p.latency_std * 0.5, 0.005)),
            (features.get("burst_rate", 0.0), p.burst_rate_mean, max(p.burst_rate_std, 0.1)),
            (features.get("typing_rate", 0.0), p.typing_rate_mean, max(p.typing_rate_std, 0.1)),
        ]

        for value, mean, std in comparisons:
            if std > 0:
                z_scores.append((value - mean) / std)
            else:
                z_scores.append(0.0)

        # Aggregate
        z_agg = math.sqrt(sum(z * z for z in z_scores)) / math.sqrt(len(z_scores))
        return min(z_agg, 5.0)  # Cap at 5 standard deviations

    def is_anomalous(self, threshold: float = 2.5) -> Tuple[bool, float]:
        """Check if current typing is anomalous.

        Returns:
            (is_anomalous, anomaly_score)
        """
        features = self.extract_features()
        score = self.compute_anomaly_score(features)
        return score > threshold, score

    # ─── Private Helpers ──────────────────────────────────────────────────

    def _compute_burst_rate(self, press_times: List[float], now: float) -> float:
        """Keys per second in a 500ms sliding window."""
        recent = [t for t in press_times if now - t < 0.5]
        return len(recent) / 0.5 if recent else 0.0

    def _compute_typing_rate(self, press_times: List[float], now: float) -> float:
        """Keys per second over the last 10 seconds."""
        recent = [t for t in press_times if now - t < 10.0]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return len(recent) / span if span > 0 else 0.0

    @staticmethod
    def _compute_consistency(hold_times: List[float]) -> float:
        """Coefficient of variation inverse: 1 − (σ/μ).

        1.0 = perfectly consistent, 0.0 = highly variable.
        """
        if len(hold_times) < 2:
            return 0.75
        mu = sum(hold_times) / len(hold_times)
        if mu == 0:
            return 0.0
        variance = sum((h - mu) ** 2 for h in hold_times) / len(hold_times)
        sigma = math.sqrt(variance)
        cv = sigma / mu
        return max(0.0, min(1.0, 1.0 - cv))

    def _default_features(self) -> dict:
        return {
            "hold_time_mean": 0.12,
            "hold_time_std": 0.05,
            "latency_mean": 0.10,
            "latency_std": 0.04,
            "burst_rate": 0.0,
            "idle_flag": 1.0,
            "typing_rate": 0.0,
            "consistency": 0.75,
            "n_events": 0,
            "n_hold_times": 0,
            "n_latencies": 0,
        }

    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._events.clear()
            self._profile = None
            self._last_key_time = 0.0
            self._keys_in_window.clear()
            self._is_idle = False

    @property
    def profile(self) -> Optional[KeystrokeProfile]:
        return self._profile


# ─── Stats Helpers ──────────────────────────────────────────────────────────

def _safe_mean(values: List[float], default: float) -> float:
    return sum(values) / len(values) if values else default

def _safe_std(values: List[float], default: float) -> float:
    if len(values) < 2:
        return default
    mu = sum(values) / len(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
