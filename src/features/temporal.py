"""Temporal feature aggregator — rolling window statistics and trends."""

import numpy as np
from typing import Dict, List, Optional


class TemporalFeatureAggregator:
    """Aggregates features over time windows for temporal analysis."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._window: List[np.ndarray] = []

    def add(self, feature_vector: np.ndarray) -> None:
        """Add a feature vector to the rolling window."""
        self._window.append(feature_vector)
        if len(self._window) > self.window_size:
            self._window.pop(0)

    def get_window(self) -> np.ndarray:
        """Get the current window as a 2D array."""
        if not self._window:
            return np.zeros((0, 0))
        return np.stack(self._window)

    def aggregate(self) -> Dict:
        """Compute aggregated statistics over the window."""
        window = self.get_window()
        if window.shape[0] == 0:
            return {"mean": np.zeros(0), "std": np.zeros(0), "trend": np.zeros(0)}

        return {
            "mean": np.mean(window, axis=0),
            "std": np.std(window, axis=0),
            "min": np.min(window, axis=0),
            "max": np.max(window, axis=0),
            "trend": self._compute_trend(window),
        }

    def _compute_trend(self, window: np.ndarray) -> np.ndarray:
        """Compute linear trend (slope) for each feature."""
        if window.shape[0] < 3:
            return np.zeros(window.shape[1])
        t = np.arange(window.shape[0])
        slopes = []
        for i in range(window.shape[1]):
            y = window[:, i]
            slope = np.polyfit(t, y, 1)[0] if len(set(y)) > 1 else 0.0
            slopes.append(slope)
        return np.array(slopes)
