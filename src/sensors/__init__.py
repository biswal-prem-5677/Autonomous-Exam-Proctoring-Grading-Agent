"""Sensor module — webcam, keyboard, mouse, and browser data acquisition.

Submodules:
    webcam   — MediaPipe/OpenCV face detection
    keyboard — Keystroke timing capture
    mouse    — Mouse movement tracking
    browser  — Browser event monitoring (tab switches, shortcuts, etc.)
"""

from .webcam import WebcamSensor
from .keyboard import KeyboardSensor
from .mouse import MouseSensor
from .browser import BrowserRiskEngine, BrowserEventType

__all__ = [
    "WebcamSensor",
    "KeyboardSensor",
    "MouseSensor",
    "BrowserRiskEngine",
    "BrowserEventType",
]
