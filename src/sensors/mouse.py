"""Mouse sensor — monitors mouse movements, clicks, and idle time."""

import time
import threading
from typing import Optional, Callable, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class MouseSample:
    """A single mouse activity sample."""
    timestamp: float
    x: float
    y: float
    dx: float       # movement delta x
    dy: float       # movement delta y
    speed: float    # pixels per second
    event: str      # move, click, scroll


class MouseSensor:
    """Monitors mouse activity patterns for behavioral analysis."""

    def __init__(self, idle_threshold_seconds: int = 120):
        self.idle_threshold = idle_threshold_seconds
        self._last_activity: float = time.time()
        self._last_position: Optional[Tuple[float, float]] = None
        self._running = False
        self._samples: List[MouseSample] = []
        self._max_samples = 1000
        self._on_activity: Optional[Callable] = None

    def start(self, on_activity: Optional[Callable] = None) -> bool:
        """Start monitoring mouse activity."""
        self._on_activity = on_activity
        self._running = True
        self._last_activity = time.time()

        try:
            from pynput import mouse

            def on_move(x, y):
                self._record_movement(x, y, "move")

            def on_click(x, y, button, pressed):
                self._record_movement(x, y, "click")

            def on_scroll(x, y, dx, dy):
                self._record_movement(x, y, "scroll")

            self._listener = mouse.Listener(
                on_move=on_move, on_click=on_click, on_scroll=on_scroll
            )
            self._listener.start()
            return True
        except ImportError:
            self._monitor_thread = threading.Thread(
                target=self._poll_loop, daemon=True
            )
            self._monitor_thread.start()
            return True

    def _record_movement(self, x: float, y: float, event: str) -> None:
        """Record a mouse event."""
        now = time.time()
        dx, dy = 0.0, 0.0
        speed = 0.0

        if self._last_position is not None:
            dx = x - self._last_position[0]
            dy = y - self._last_position[1]
            dt = now - self._last_activity
            if dt > 0:
                speed = (dx**2 + dy**2) ** 0.5 / dt

        sample = MouseSample(
            timestamp=now, x=x, y=y,
            dx=dx, dy=dy, speed=speed, event=event,
        )

        self._last_activity = now
        self._last_position = (x, y)
        self._samples.append(sample)

        # Trim old samples
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples:]

        if self._on_activity:
            self._on_activity(sample)

    def _poll_loop(self) -> None:
        """Fallback polling loop."""
        while self._running:
            time.sleep(0.1)

    def get_idle_seconds(self) -> float:
        """Get current idle time in seconds."""
        return time.time() - self._last_activity

    def is_idle(self) -> bool:
        """Check if mouse has been idle beyond threshold."""
        return self.get_idle_seconds() > self.idle_threshold

    def get_recent_samples(self, window_seconds: float = 60) -> List[MouseSample]:
        """Get samples from the last window_seconds."""
        cutoff = time.time() - window_seconds
        return [s for s in self._samples if s.timestamp >= cutoff]

    def get_movement_stats(self, window_seconds: float = 60) -> dict:
        """Compute movement statistics for a time window."""
        samples = self.get_recent_samples(window_seconds)
        if not samples:
            return {"avg_speed": 0, "total_distance": 0, "event_count": 0}

        speeds = [s.speed for s in samples if s.speed > 0]
        total_dist = sum(
            (s.dx**2 + s.dy**2) ** 0.5 for s in samples
        )

        return {
            "avg_speed": np.mean(speeds) if speeds else 0,
            "max_speed": np.max(speeds) if speeds else 0,
            "total_distance": total_dist,
            "event_count": len(samples),
            "click_count": sum(1 for s in samples if s.event == "click"),
        }

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if hasattr(self, '_listener') and self._listener:
            self._listener.stop()
