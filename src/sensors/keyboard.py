"""Keyboard sensor — monitors typing events and idle time."""

import time
import threading
from typing import Optional, Callable
from datetime import datetime


class KeyboardSensor:
    """Monitors keyboard activity and detects idle periods."""

    def __init__(self, idle_threshold_seconds: int = 120):
        self.idle_threshold = idle_threshold_seconds
        self._last_activity: float = time.time()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._on_activity: Optional[Callable] = None
        self._on_idle: Optional[Callable] = None
        self._event_count = 0
        self._events_per_minute: list = []

    def start(self, on_activity: Optional[Callable] = None,
              on_idle: Optional[Callable] = None) -> bool:
        """Start monitoring keyboard events."""
        self._on_activity = on_activity
        self._on_idle = on_idle
        self._running = True
        self._last_activity = time.time()

        # Use pynput for cross-platform keyboard monitoring
        try:
            from pynput import keyboard

            def on_press(key):
                self._handle_activity()

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            return True
        except ImportError:
            # Fallback: manual polling
            self._monitor_thread = threading.Thread(
                target=self._poll_loop, daemon=True
            )
            self._monitor_thread.start()
            return True

    def _handle_activity(self) -> None:
        """Record keyboard activity."""
        now = time.time()
        self._event_count += 1
        self._last_activity = now

        # Track events per minute for feature extraction
        self._events_per_minute.append(now)
        # Keep only last minute of events
        self._events_per_minute = [
            t for t in self._events_per_minute if now - t <= 60
        ]

        if self._on_activity:
            self._on_activity()

    def _poll_loop(self) -> None:
        """Fallback polling loop."""
        import ctypes

        last_key_state = False
        while self._running:
            # Windows-specific: check if any key is being pressed
            # This is a simplified fallback
            current = ctypes.windll.user32.GetKeyState(0x20) & 0x8000 != 0
            if current and not last_key_state:
                self._handle_activity()
            last_key_state = current
            time.sleep(0.05)

    def get_idle_seconds(self) -> float:
        """Get current idle time in seconds."""
        return time.time() - self._last_activity

    def get_events_per_minute(self) -> float:
        """Get typing rate (events per minute)."""
        now = time.time()
        self._events_per_minute = [
            t for t in self._events_per_minute if now - t <= 60
        ]
        return len(self._events_per_minute)

    def get_total_events(self) -> int:
        """Get total key press count."""
        return self._event_count

    def is_idle(self) -> bool:
        """Check if keyboard has been idle beyond threshold."""
        return self.get_idle_seconds() > self.idle_threshold

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if hasattr(self, '_listener') and self._listener:
            self._listener.stop()
