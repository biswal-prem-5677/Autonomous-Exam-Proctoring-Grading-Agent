"""
Browser Environment Monitoring Module.

Monitors browser-level events that may indicate suspicious examination behavior:
    - Tab/window focus changes
    - Fullscreen exit events
    - Copy/paste/cut operations
    - Keyboard shortcuts (Ctrl+T, Ctrl+N, etc.)
    - Context menu suppression
    - Developer tools access
    - Browser navigation events

Mathematical Model:
────────────────────────────────────────────────────────────────────────────
Event logging: Each event E is recorded with:
    E = (timestamp, event_type, severity, source_element)

Severity weights:
    TAB_SWITCH = 0.30
    FULLSCREEN_EXIT = 0.35
    COPY_PASTE = 0.40
    SHORTCUT_DETECTED = 0.25
    CONTEXT_MENU = 0.15
    DEVTOOLS = 0.50

Aggregated browser risk:
    R_browser = (Σ event_weights × event_durations) / window_duration
    → Normalized to [0, 1]

Architecture:
    Browser Events (JavaScript)
         │
         ▼
    EventBuffer (circular buffer, max 500 events)
         │
         ▼
    BrowserRiskEngine (aggregation + scoring)
         │
         ▼
    Risk Score (float [0, 1])
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Callable
import json


class BrowserEventType(Enum):
    """Types of browser-level events."""
    TAB_FOCUS_LOST = "tab_focus_lost"
    FULLSCREEN_EXIT = "fullscreen_exit"
    COPY = "copy"
    PASTE = "paste"
    CUT = "cut"
    SHORTCUT_DETECTED = "shortcut_detected"
    CONTEXT_MENU = "context_menu"
    DEVTOOLS_OPEN = "devtools_open"
    NAVIGATION = "navigation"
    WINDOW_BLUR = "window_blur"
    VISIBILITY_HIDDEN = "visibility_hidden"


# Severity weights — calibrated based on literature review of proctoring systems
EVENT_SEVERITY: Dict[BrowserEventType, float] = {
    BrowserEventType.TAB_FOCUS_LOST: 0.30,
    BrowserEventType.FULLSCREEN_EXIT: 0.35,
    BrowserEventType.COPY: 0.40,
    BrowserEventType.PASTE: 0.40,
    BrowserEventType.CUT: 0.40,
    BrowserEventType.SHORTCUT_DETECTED: 0.25,
    BrowserEventType.CONTEXT_MENU: 0.15,
    BrowserEventType.DEVTOOLS_OPEN: 0.50,
    BrowserEventType.NAVIGATION: 0.20,
    BrowserEventType.WINDOW_BLUR: 0.30,
    BrowserEventType.VISIBILITY_HIDDEN: 0.25,
}

# Forbidden shortcut combinations
FORBIDDEN_SHORTCUTS = [
    "Ctrl+T", "Ctrl+N", "Ctrl+W", "Ctrl+Shift+N",  # Tab/window management
    "Ctrl+Shift+I", "F12", "Ctrl+Shift+J", "Ctrl+Shift+C",  # Dev tools
    "Alt+Tab", "Ctrl+Alt+Delete",  # OS-level
    "Ctrl+P",  # Print
]


@dataclass
class BrowserEvent:
    """A single browser monitoring event."""
    timestamp: float
    event_type: BrowserEventType
    severity: float
    duration: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "severity": self.severity,
            "duration": self.duration,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        elapsed = self.timestamp
        return (
            f"BrowserEvent({self.event_type.value}, "
            f"t={elapsed:.1f}s, sev={self.severity:.2f})"
        )


class EventBuffer:
    """Thread-safe circular buffer for browser events.

    Capacity: 500 events (configurable).
    Decay: Older events gradually lose weight.
    """

    def __init__(self, max_events: int = 500):
        self._buffer: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def add(self, event: BrowserEvent) -> None:
        with self._lock:
            self._buffer.append(event)

    def get_events_in_window(self, window_seconds: float) -> List[BrowserEvent]:
        """Return events within the last window_seconds."""
        cutoff = time.time() - window_seconds
        with self._lock:
            return [e for e in self._buffer if e.timestamp >= cutoff]

    def get_all(self) -> List[BrowserEvent]:
        with self._lock:
            return list(self._buffer)

    def get_recent_n(self, n: int) -> List[BrowserEvent]:
        with self._lock:
            return list(self._buffer)[-n:]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def event_type_counts(self) -> Dict[str, int]:
        """Count events by type in current buffer."""
        counts: Dict[str, int] = {}
        with self._lock:
            for e in self._buffer:
                counts[e.event_type.value] = counts.get(e.event_type.value, 0) + 1
        return counts


class BrowserRiskEngine:
    """Computes browser-related risk score from event buffer.

    Risk Formula:
    ────────────────────────────────────────────────────────────────────────
    R_browser(t) = max over windows of:

        Σ [severity_i × exp(-λ × age_i)] / window_duration

    where age_i is the age of event i in seconds and λ is the decay rate.

    With temporal decay:
        R = Σ (severity_i × decay_factor_i) / N_windows

    Default λ = 0.1 (half-life ≈ 7 seconds at 10Hz sampling).
    ────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        decay_lambda: float = 0.1,
        severity_threshold: float = 0.5,
        event_threshold: int = 3,
    ):
        self.event_buffer = EventBuffer()
        self.window_seconds = window_seconds
        self.decay_lambda = decay_lambda
        self.severity_threshold = severity_threshold
        self.event_threshold = event_threshold
        self._current_risk: float = 0.0
        self._lock = threading.Lock()

    def record_event(
        self,
        event_type: BrowserEventType,
        metadata: Optional[Dict] = None,
        duration: float = 0.0,
    ) -> BrowserEvent:
        """Record a new browser event."""
        event = BrowserEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=EVENT_SEVERITY.get(event_type, 0.1),
            duration=duration,
            metadata=metadata or {},
        )
        self.event_buffer.add(event)
        return event

    def compute_risk(self) -> float:
        """Compute current browser risk score [0, 1].

        Algorithm:
        1. Retrieve events in current window
        2. Apply exponential decay to each event based on age
        3. Sum weighted severity contributions
        4. Normalize by window duration
        """
        now = time.time()
        events = self.event_buffer.get_events_in_window(self.window_seconds)

        if not events:
            with self._lock:
                self._current_risk *= 0.95  # Gentle decay when no events
                return self._current_risk

        # Weighted sum with exponential age decay
        weighted_sum = 0.0
        for event in events:
            age = now - event.timestamp
            decay_factor = np.exp(-self.decay_lambda * age) if _HAS_NUMPY else _math_exp(-self.decay_lambda * age)
            weighted_sum += event.severity * decay_factor

        # Normalize: max possible score in a window
        max_events = 20  # reasonable upper bound
        max_severity = 0.50  # DEVTOOLS
        max_possible = max_events * max_severity

        risk = min(weighted_sum / max_possible, 1.0)

        with self._lock:
            # Smooth with previous risk
            self._current_risk = 0.7 * self._current_risk + 0.3 * risk

        return self._current_risk

    def get_event_summary(self) -> Dict:
        """Get summary of events in current window."""
        events = self.event_buffer.get_events_in_window(self.window_seconds)
        summary = {
            "total_events": len(events),
            "event_counts": self.event_buffer.event_type_counts,
            "max_severity": max((e.severity for e in events), default=0.0),
            "window_seconds": self.window_seconds,
        }
        return summary

    def should_escalate(self) -> bool:
        """Determine if browser evidence alone warrants escalation."""
        events = self.event_buffer.get_events_in_window(self.window_seconds)
        if not events:
            return False

        # Critical event types trigger immediate escalation
        critical_events = {
            BrowserEventType.DEVTOOLS_OPEN,
            BrowserEventType.FULLSCREEN_EXIT,
        }
        for event in events:
            if event.event_type in critical_events:
                return True

        # High-severity event count threshold
        high_severity_count = sum(
            1 for e in events if e.severity >= self.severity_threshold
        )
        return high_severity_count >= self.event_threshold

    def reset(self) -> None:
        """Reset risk state (e.g., between exam sessions)."""
        self.event_buffer.clear()
        with self._lock:
            self._current_risk = 0.0


# ─── JavaScript Injection Code ──────────────────────────────────────────────
# This string is injected into the exam page to capture browser events.

BROWSER_MONITOR_JS = """
(function() {
    'use strict';

    const EVENT_TYPES = {
        TAB_FOCUS_LOST: 'tab_focus_lost',
        FULLSCREEN_EXIT: 'fullscreen_exit',
        COPY: 'copy',
        PASTE: 'paste',
        CUT: 'cut',
        SHORTCUT: 'shortcut_detected',
        CONTEXT_MENU: 'context_menu',
        DEVTOOLS: 'devtools_open',
        NAVIGATION: 'navigation',
        WINDOW_BLUR: 'window_blur',
        VISIBILITY_HIDDEN: 'visibility_hidden'
    };

    const FORBIDDEN = [
        'Control+t', 'Control+n', 'Control+w', 'Control+Shift+n',
        'Control+Shift+i', 'F12', 'Control+Shift+j', 'Control+Shift+c',
        'Control+p'
    ];

    function sendEvent(type, metadata) {
        window.parent.postMessage({ source: 'exam-monitor', event: type, data: metadata }, '*');
    }

    // Visibility change
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            sendEvent(EVENT_TYPES.VISIBILITY_HIDDEN, { time: Date.now() });
        }
    });

    // Fullscreen change
    document.addEventListener('fullscreenchange', function() {
        if (!document.fullscreenElement) {
            sendEvent(EVENT_TYPES.FULLSCREEN_EXIT, { time: Date.now() });
        }
    });

    // Window blur
    window.addEventListener('blur', function() {
        sendEvent(EVENT_TYPES.WINDOW_BLUR, { time: Date.now() });
    });

    // Copy/Paste/Cut
    document.addEventListener('copy', function(e) { sendEvent(EVENT_TYPES.COPY); });
    document.addEventListener('paste', function(e) { sendEvent(EVENT_TYPES.PASTE); });
    document.addEventListener('cut', function(e) { sendEvent(EVENT_TYPES.CUT); });

    // Context menu
    document.addEventListener('contextmenu', function(e) {
        sendEvent(EVENT_TYPES.CONTEXT_MENU, { target: e.target.tagName });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        const combo = [];
        if (e.ctrlKey || e.metaKey) combo.push('Control');
        if (e.shiftKey) combo.push('Shift');
        if (e.altKey) combo.push('Alt');
        if (e.key !== 'Control' && e.key !== 'Shift' && e.key !== 'Alt' && e.key !== 'Meta') {
            combo.push(e.key.length === 1 ? e.key.toUpperCase() : e.key);
        }
        const shortcut = combo.join('+');
        if (FORBIDDEN.some(f => shortcut.includes(f.replace('Control', 'Control').replace('+t', '').replace('+n', '')))) {
            sendEvent(EVENT_TYPES.SHORTCUT, { combo: shortcut });
        }
    });

    // DevTools detection (heuristic)
    const threshold = 160;
    setInterval(function() {
        if (window.outerHeight - window.innerHeight > threshold ||
            window.outerWidth - window.innerWidth > threshold) {
            sendEvent(EVENT_TYPES.DEVTOOLS, {
                widthDiff: window.outerWidth - window.innerWidth,
                heightDiff: window.outerHeight - window.innerHeight
            });
        }
    }, 1000);

    // Navigation
    window.addEventListener('beforeunload', function() {
        sendEvent(EVENT_TYPES.NAVIGATION, { url: window.location.href });
    });
})();
"""


# ─── Import compatibility ───────────────────────────────────────────────────

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    import math as _math
    def _math_exp(x): return _math.exp(x)
