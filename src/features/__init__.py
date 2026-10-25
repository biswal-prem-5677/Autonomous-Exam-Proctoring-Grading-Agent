"""Feature engineering module — behavioral, temporal, and normalization utilities.

Submodules:
    behavioral  — 22-dim behavioral feature vector extraction
    temporal    — rolling window aggregation with trend detection
    normalization — Z-score normalization pipeline
    keystroke   — keystroke dynamics biometrics
    mouse       — mouse dynamics biometrics
"""

from .canonical import (
    CANONICAL_FEATURE_NAMES,
    CANONICAL_INDICES,
    N_CANONICAL_FEATURES,
    N_FEATURES,
    extract_canonical_features,
    dict_to_canonical_vector,
    canonical_vector_to_dict,
    validate_canonical_vector,
)
from .behavioral import BehavioralFeatureExtractor
from .temporal import TemporalFeatureAggregator
from .normalization import FeatureNormalizer
from .keystroke import KeystrokeDynamics, KeystrokeProfile, KeystrokeEvent
from .mouse import MouseDynamics, MovementMetrics, MovementType, MouseEvent

__all__ = [
    "CANONICAL_FEATURE_NAMES",
    "CANONICAL_INDICES",
    "N_CANONICAL_FEATURES",
    "N_FEATURES",
    "extract_canonical_features",
    "dict_to_canonical_vector",
    "canonical_vector_to_dict",
    "validate_canonical_vector",
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
