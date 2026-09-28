"""Technical-Event Intelligence — separates technical failures from integrity violations.

A critical design requirement:
STUDENT ANOMALY ≠ SYSTEM HARDWARE / NETWORK OUTAGE.

Examples:
- Camera USB/driver disconnects -> Face absent detected
  * NAIVE SYSTEM: "Face absent for 30s -> Cheating detected!"
  * INTELLIGENT SYSTEM: "Camera hardware dropped (0 FPS, device error). Mark TECHNICAL_INCIDENT. Suppress integrity score penalty."

- WiFi disconnects for 15 seconds -> Browser connection lost
  * NAIVE SYSTEM: "Fullscreen exit / connection severed -> Violation!"
  * INTELLIGENT SYSTEM: "Socket timeout & network adapter reset. Grant network grace period. Pause exam clock."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time


class TechnicalIncidentType(Enum):
    CAMERA_DISCONNECT = "CAMERA_DISCONNECT"
    CAMERA_FRAME_DROP = "CAMERA_FRAME_DROP"
    MICROPHONE_ERROR = "MICROPHONE_ERROR"
    NETWORK_DISCONNECT = "NETWORK_DISCONNECT"
    NETWORK_HIGH_LATENCY = "NETWORK_HIGH_LATENCY"
    BROWSER_CRASH = "BROWSER_CRASH"
    SYSTEM_HIGH_LOAD = "SYSTEM_HIGH_LOAD"
    OS_NOTIFICATION = "OS_NOTIFICATION"


@dataclass
class TechnicalIncident:
    """A verified hardware or infrastructure incident."""
    incident_id: str
    incident_type: TechnicalIncidentType
    timestamp: float
    duration_seconds: float
    diagnostics: Dict[str, Any]
    suppressed_event_types: List[str]
    mitigation_action: str
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "incident_type": self.incident_type.value,
            "timestamp": self.timestamp,
            "duration_seconds": round(self.duration_seconds, 2),
            "diagnostics": self.diagnostics,
            "suppressed_event_types": self.suppressed_event_types,
            "mitigation_action": self.mitigation_action,
            "is_active": self.is_active,
        }


class TechnicalEventEngine:
    """Detects, tracks, and isolates technical incidents during an examination."""

    # Map technical causes to sensory events that should NOT penalize integrity
    SUPPRESSION_MAP = {
        TechnicalIncidentType.CAMERA_DISCONNECT: [
            "face_absent", "head_turned", "gaze_away", "camera_disconnected"
        ],
        TechnicalIncidentType.CAMERA_FRAME_DROP: [
            "face_absent", "gaze_away"
        ],
        TechnicalIncidentType.MICROPHONE_ERROR: [
            "audio_unavailable", "loud_audio"
        ],
        TechnicalIncidentType.NETWORK_DISCONNECT: [
            "tab_switch", "window_blur", "fullscreen_exit", "network_interruption"
        ],
        TechnicalIncidentType.SYSTEM_HIGH_LOAD: [
            "idle_keyboard", "idle_mouse", "frame_delay"
        ],
    }

    def __init__(self, grace_period_seconds: float = 60.0):
        self.grace_period_seconds = grace_period_seconds
        self._active_incidents: Dict[str, TechnicalIncident] = {}
        self._incident_history: List[TechnicalIncident] = []

    def analyze_system_state(
        self,
        system_diagnostics: Dict[str, Any],
        raw_events: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[TechnicalIncident]]:
        """Filter incoming raw events, isolating technical events from legitimate behavioral events.

        Args:
            system_diagnostics: Hardware state e.g.
                {'camera_connected': bool, 'camera_fps': float, 'network_online': bool,
                 'ping_ms': float, 'mic_active': bool, 'cpu_percent': float}
            raw_events: Incoming events from perception engine

        Returns:
            Tuple of (purified_behavioral_events, active_technical_incidents)
        """
        now = time.time()
        newly_detected_incidents: List[TechnicalIncident] = []

        # 1. Check camera status
        if not system_diagnostics.get("camera_connected", True):
            inc = self._register_or_update(
                TechnicalIncidentType.CAMERA_DISCONNECT,
                diagnostics={"fps": system_diagnostics.get("camera_fps", 0)},
                mitigation="Pause camera-based violation penalties. Request camera re-authorization.",
            )
            newly_detected_incidents.append(inc)
        elif system_diagnostics.get("camera_fps", 30) < 5.0:
            inc = self._register_or_update(
                TechnicalIncidentType.CAMERA_FRAME_DROP,
                diagnostics={"fps": system_diagnostics.get("camera_fps", 0)},
                mitigation="Low camera framerate. Suppress temporal vision penalties.",
            )
            newly_detected_incidents.append(inc)
        else:
            self._resolve_incident(TechnicalIncidentType.CAMERA_DISCONNECT)
            self._resolve_incident(TechnicalIncidentType.CAMERA_FRAME_DROP)

        # 2. Check network status
        if not system_diagnostics.get("network_online", True):
            inc = self._register_or_update(
                TechnicalIncidentType.NETWORK_DISCONNECT,
                diagnostics={"ping": system_diagnostics.get("ping_ms", None)},
                mitigation="Network disconnected. Start 60s reconnection grace period.",
            )
            newly_detected_incidents.append(inc)
        else:
            self._resolve_incident(TechnicalIncidentType.NETWORK_DISCONNECT)

        # 3. Check audio hardware
        if not system_diagnostics.get("mic_active", True):
            inc = self._register_or_update(
                TechnicalIncidentType.MICROPHONE_ERROR,
                diagnostics={"mic_state": "inactive_or_denied"},
                mitigation="Audio channel muted or unavailable. Mark audio modality missing.",
            )
            newly_detected_incidents.append(inc)
        else:
            self._resolve_incident(TechnicalIncidentType.MICROPHONE_ERROR)

        # 4. Filter events: remove events that are artifacts of technical faults
        suppressed_types = set()
        for inc in self._active_incidents.values():
            if inc.is_active:
                for ev_type in inc.suppressed_event_types:
                    suppressed_types.add(ev_type)

        purified_events = []
        for ev in raw_events:
            ev_type = ev.get("type", "")
            if ev_type in suppressed_types:
                # Mark as suppressed rather than silent drop so it remains auditable
                ev_copy = dict(ev)
                ev_copy["suppressed_by_technical_incident"] = True
                # Exclude from risk calculation
                continue
            purified_events.append(ev)

        return purified_events, list(self._active_incidents.values())

    def _register_or_update(
        self,
        incident_type: TechnicalIncidentType,
        diagnostics: Dict[str, Any],
        mitigation: str,
    ) -> TechnicalIncident:
        key = incident_type.value
        now = time.time()
        if key in self._active_incidents and self._active_incidents[key].is_active:
            inc = self._active_incidents[key]
            inc.duration_seconds = now - inc.timestamp
            inc.diagnostics = diagnostics
            return inc

        import uuid
        inc = TechnicalIncident(
            incident_id=f"tech_{uuid.uuid4().hex[:8]}",
            incident_type=incident_type,
            timestamp=now,
            duration_seconds=0.0,
            diagnostics=diagnostics,
            suppressed_event_types=self.SUPPRESSION_MAP.get(incident_type, []),
            mitigation_action=mitigation,
            is_active=True,
        )
        self._active_incidents[key] = inc
        self._incident_history.append(inc)
        return inc

    def _resolve_incident(self, incident_type: TechnicalIncidentType) -> None:
        key = incident_type.value
        if key in self._active_incidents and self._active_incidents[key].is_active:
            inc = self._active_incidents[key]
            inc.is_active = False
            inc.duration_seconds = time.time() - inc.timestamp

    def has_active_fault(self) -> bool:
        return any(inc.is_active for inc in self._active_incidents.values())

    def get_active_incidents(self) -> List[TechnicalIncident]:
        return [inc for inc in self._active_incidents.values() if inc.is_active]
