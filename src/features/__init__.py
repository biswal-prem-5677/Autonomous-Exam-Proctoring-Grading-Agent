"""Feature engineering module — behavioral, temporal, and normalization utilities.

Submodules:
    behavioral  — 22-dim behavioral feature vector extraction
    temporal    — rolling window aggregation with trend detection
    normalization — Z-score normalization pipeline
    keystroke   — keystroke dynamics biometrics
    mouse       — mouse dynamics biometrics
"""

from .behavioral import BehavioralFeatureExtractor
from .temporal import TemporalFeatureAggregator
from .normalization import FeatureNormalizer
from .keystroke import KeystrokeDynamics, KeystrokeProfile, KeystrokeEvent
from .mouse import MouseDynamics, MovementMetrics, MovementType, MouseEvent

__all__ = [
    "BehavioralFeatureExtractor",
    "TemporalFeatureAggregator",
    "FeatureNormalizer",
    "KeystrokeDynamics",
    "KeystrokeProfile",
    "KeystrokeEvent",
    "MouseDynamics",
    "MovementMetrics",
    "MovementType",
    "MouseEvent",
]
