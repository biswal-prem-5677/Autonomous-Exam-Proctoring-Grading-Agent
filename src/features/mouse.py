"""
Mouse Dynamics Module.

Implements behavioral biometrics based on mouse movement patterns:
    1. Movement feature extraction (velocity, acceleration, distance, curvature)
    2. Click pattern analysis
    3. Trajectory classification
    4. Mouse anomaly detection

Mathematical Foundation:
────────────────────────────────────────────────────────────────────────────
Feature Vector (per time window):
    M = [v_mean, v_std, v_max, a_mean, a_std, a_max, d_total, d_horizontal,
         d_vertical, curvature_mean, click_rate, right_click_rate,
         scroll_rate, idle_ratio, movement_type]

Where:
    v(t)    = sqrt((dx/dt)² + (dy/dt)²)    (instantaneous velocity)
    a(t)    = dv/dt                         (acceleration)
    d_total  = Σ sqrt(dx² + dy²)            (total distance traveled)
    curvature = |dθ/ds|                      (path curvature)
    click_rate = N_clicks / window_duration
    idle_ratio = idle_time / window_duration

Movement Classification:
    - Reading:   Low velocity, low curvature, intermittent
    - Searching: Medium velocity, high direction changes
    - Selecting: Bursts of movement with pauses
    - Navigating: Linear, goal-directed movement

Anomaly Detection:
    Same keystroke-based z-score approach:
        A = √(Σ z_i²)
────────────────────────────────────────────────────────────────────────────
"""

import time
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class MovementType(Enum):
    """Classified mouse movement patterns."""
    READING = "reading"
    SEARCHING = "searching"
    SELECTING = "selecting"
    NAVIGATING = "navigating"
    IDLE = "idle"
    ANOMALOUS = "anomalous"


@dataclass
class MouseEvent:
    """A single mouse event with position and timestamp."""
    x: float
    y: float
    timestamp: float
    event_type: str  # move, click, scroll
    button: Optional[str] = None  # left, right, middle


@dataclass
class MovementMetrics:
    """Computed movement metrics from a sequence of mouse events."""
    velocity_mean: float = 0.0
    velocity_std: float = 0.0
    velocity_max: float = 0.0
    acceleration_mean: float = 0.0
    acceleration_std: float = 0.0
    acceleration_max: float = 0.0
    total_distance: float = 0.0
    horizontal_distance: float = 0.0
    vertical_distance: float = 0.0
    curvature_mean: float = 0.0
    click_count: int = 0
    right_click_count: int = 0
    scroll_count: int = 0
    idle_ratio: float = 1.0
    movement_type: MovementType = MovementType.IDLE
    anomaly_score: float = 0.0


class MouseDynamics:
    """Mouse behavioral biometrics engine."""

    IDLE_THRESHOLD_S = 30.0
    VELOCITY_WINDOW_MS = 100
    MIN_EVENTS_FOR_ANALYSIS = 2

    def __init__(self, max_events: int = 2000):
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._last_position: Optional[Tuple[float, float]] = None
        self._last_timestamp: Optional[float] = None
        self._idle_start: Optional[float] = None

    # ─── Event Recording ──────────────────────────────────────────────────

    def record_move(self, x: float, y: float) -> None:
        """Record mouse movement."""
        now = time.time()
        event = MouseEvent(x=x, y=y, timestamp=now, event_type="move")
        with self._lock:
            self._events.append(event)
        self._update_position(x, y, now)

    def record_click(self, x: float, y: float, button: str = "left") -> None:
        """Record mouse click."""
        now = time.time()
        event = MouseEvent(x=x, y=y, timestamp=now, event_type="click", button=button)
        with self._lock:
            self._events.append(event)
        self._update_position(x, y, now)

    def record_scroll(self, x: float, y: float) -> None:
        """Record mouse scroll."""
        now = time.time()
        event = MouseEvent(x=x, y=y, timestamp=now, event_type="scroll")
        with self._lock:
            self._events.append(event)

    # ─── Feature Extraction ─────────────────────────────────────────────────

    def extract_metrics(self) -> MovementMetrics:
        """Extract comprehensive movement metrics from recent events."""
        with self._lock:
            events = list(self._events)

        if len(events) < self.MIN_EVENTS_FOR_ANALYSIS:
            return self._default_metrics()

        # Separate by type
        moves = [e for e in events if e.event_type == "move"]
        clicks = [e for e in events if e.event_type == "click"]
        scrolls = [e for e in events if e.event_type == "scroll"]

        # Compute velocities and accelerations
        velocities = self._compute_velocities(moves)
        accelerations = self._compute_accelerations(velocities)

        # Distance metrics
        total_dist, h_dist, v_dist = self._compute_distances(moves)

        # Curvature
        curvature_mean = self._compute_curvature(moves)

        # Idle detection
        now = time.time()
        recent = [e for e in events if now - e.timestamp < 10.0]
        idle_time = sum(10.0 - (now - e.timestamp) for e in recent) / len(recent) if recent else 10.0
        idle_ratio = idle_time / 10.0

        # Movement type classification
        movement_type = self._classify_movement(
            velocities, accelerations, curvature_mean, idle_ratio
        )

        # Build metrics
        metrics = MovementMetrics(
            velocity_mean=_safe_mean(velocities) if velocities else 0.0,
            velocity_std=_safe_std(velocities) if velocities else 0.0,
            velocity_max=max(velocities) if velocities else 0.0,
            acceleration_mean=_safe_mean(accelerations) if accelerations else 0.0,
            acceleration_std=_safe_std(accelerations) if accelerations else 0.0,
            acceleration_max=max(accelerations) if accelerations else 0.0,
            total_distance=total_dist,
            horizontal_distance=h_dist,
            vertical_distance=v_dist,
            curvature_mean=curvature_mean,
            click_count=len(clicks),
            right_click_count=sum(1 for c in clicks if c.button == "right"),
            scroll_count=len(scrolls),
            idle_ratio=idle_ratio,
            movement_type=movement_type,
        )
        return metrics

    def extract_feature_vector(self) -> List[float]:
        """Return features as flat list for ML input."""
        m = self.extract_metrics()
        return [
            m.velocity_mean,
            m.velocity_std,
            m.velocity_max,
            m.acceleration_mean,
            m.acceleration_std,
            m.acceleration_max,
            m.total_distance,
            m.horizontal_distance,
            m.vertical_distance,
            m.curvature_mean,
            m.click_count / 100.0,  # normalize
            m.right_click_count / 100.0,
            m.scroll_count / 100.0,
            m.idle_ratio,
            # Movement type as one-hot
            1.0 if m.movement_type == MovementType.READING else 0.0,
            1.0 if m.movement_type == MovementType.SEARCHING else 0.0,
            1.0 if m.movement_type == MovementType.SELECTING else 0.0,
            1.0 if m.movement_type == MovementType.NAVIGATING else 0.0,
            1.0 if m.movement_type == MovementType.IDLE else 0.0,
            1.0 if m.movement_type == MovementType.ANOMALOUS else 0.0,
        ]

    # ─── Private Helpers ──────────────────────────────────────────────────

    def _update_position(self, x: float, y: float, timestamp: float) -> None:
        """Update position tracking."""
        if self._idle_start is not None:
            # Was idle, now active
            self._idle_start = None
        self._last_position = (x, y)
        self._last_timestamp = timestamp

    def _compute_velocities(self, moves: List[MouseEvent]) -> List[float]:
        """Compute instantaneous velocities between consecutive moves."""
        if len(moves) < 2:
            return []

        velocities = []
        for i in range(1, len(moves)):
            dt = moves[i].timestamp - moves[i - 1].timestamp
            if dt <= 0:
                continue
            dx = moves[i].x - moves[i - 1].x
            dy = moves[i].y - moves[i - 1].y
            dist = math.sqrt(dx * dx + dy * dy)
            velocities.append(dist / dt)  # pixels per second
        return velocities

    def _compute_accelerations(self, velocities: List[float]) -> List[float]:
        """Compute accelerations from velocities."""
        if len(velocities) < 2:
            return []
        accelerations = []
        for i in range(1, len(velocities)):
            dv = velocities[i] - velocities[i - 1]
            accelerations.append(abs(dv))
        return accelerations

    def _compute_distances(self, moves: List[MouseEvent]) -> Tuple[float, float, float]:
        """Compute total, horizontal, and vertical distances."""
        if len(moves) < 2:
            return 0.0, 0.0, 0.0

        total = 0.0
        h_dist = 0.0
        v_dist = 0.0
        for i in range(1, len(moves)):
            dx = abs(moves[i].x - moves[i - 1].x)
            dy = abs(moves[i].y - moves[i - 1].y)
            total += math.sqrt(dx * dx + dy * dy)
            h_dist += dx
            v_dist += dy
        return total, h_dist, v_dist

    def _compute_curvature(self, moves: List[MouseEvent]) -> float:
        """Compute mean path curvature.

        Curvature κ = |dθ/ds| where θ is direction angle.
        """
        if len(moves) < 3:
            return 0.0

        curvatures = []
        for i in range(1, len(moves) - 1):
            # Direction vectors
            dx1 = moves[i].x - moves[i - 1].x
            dy1 = moves[i].y - moves[i - 1].y
            dx2 = moves[i + 1].x - moves[i].x
            dy2 = moves[i + 1].y - moves[i].y

            # Angle change
            theta1 = math.atan2(dy1, dx1) if dx1 != 0 or dy1 != 0 else 0
            theta2 = math.atan2(dy2, dx2) if dx2 != 0 or dy2 != 0 else 0
            d_theta = abs(theta2 - theta1)
            if d_theta > math.pi:
                d_theta = 2 * math.pi - d_theta

            # Arc length
            ds = math.sqrt(dx1 * dx1 + dy1 * dy1)
            if ds > 0:
                curvatures.append(d_theta / ds)
        return _safe_mean(curvatures) if curvatures else 0.0

    def _classify_movement(
        self,
        velocities: List[float],
        accelerations: List[float],
        curvature: float,
        idle_ratio: float,
    ) -> MovementType:
        """Classify movement pattern."""
        # Idle detection
        if idle_ratio > 0.7 or len(velocities) < 3:
            return MovementType.IDLE

        v_mean = _safe_mean(velocities) if velocities else 0
        v_std = _safe_std(velocities) if velocities else 0

        # Anomalous: extremely fast or erratic
        if v_mean > 5000 or (v_std > v_mean * 0.8 and v_mean > 500):
            return MovementType.ANOMALOUS

        # Reading: low velocity, low acceleration
        if v_mean < 200 and _safe_mean(accelerations) < 500:
            return MovementType.READING

        # Searching: medium velocity, high direction changes
        if v_mean < 800 and curvature > 0.01:
            return MovementType.SEARCHING

        # Selecting: bursty with pauses
        if v_mean < 600 and _safe_mean(accelerations) > 1000:
            return MovementType.SELECTING

        # Navigating: linear, goal-directed
        return MovementType.NAVIGATING

    def _default_metrics(self) -> MovementMetrics:
        return MovementMetrics(
            velocity_mean=0.0,
            velocity_std=0.0,
            velocity_max=0.0,
            acceleration_mean=0.0,
            acceleration_std=0.0,
            acceleration_max=0.0,
            total_distance=0.0,
            horizontal_distance=0.0,
            vertical_distance=0.0,
            curvature_mean=0.0,
            click_count=0,
            right_click_count=0,
            scroll_count=0,
            idle_ratio=1.0,
            movement_type=MovementType.IDLE,
        )

    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._events.clear()
            self._last_position = None
            self._last_timestamp = None
            self._idle_start = None


# ─── Stats Helpers ──────────────────────────────────────────────────────────

def _safe_mean(values: List[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default

def _safe_std(values: List[float], default: float = 0.0) -> float:
    if len(values) < 2:
        return default
    mu = sum(values) / len(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
