"""Feature normalization and preprocessing utilities."""

import numpy as np
from typing import Optional, Tuple


class FeatureNormalizer:
    """Z-score normalization with running mean/std estimation."""

    def __init__(self):
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._count = 0
        self._initialized = False

    def fit(self, X: np.ndarray) -> None:
        """Fit normalizer on training data."""
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1.0  # Avoid division by zero
        self._initialized = True

    def update(self, X: np.ndarray) -> np.ndarray:
        """Update running statistics and return normalized features."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._initialized:
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0)
            self._std[self._std == 0] = 1.0
            self._count = X.shape[0]
            self._initialized = True
            return (X - self._mean) / self._std

        # Running update
        total = self._count + X.shape[0]
        delta_mean = np.mean(X, axis=0) - self._mean
        self._mean += delta_mean * (X.shape[0] / total)

        old_m = self._mean - delta_mean
        self._std = np.sqrt(
            (self._count * self._std ** 2 + X.shape[0] * np.std(X, axis=0) ** 2
             + self._count * X.shape[0] * delta_mean ** 2 / total) / total
        )
        self._std[self._std == 0] = 1.0
        self._count = total

        return (X - self._mean) / self._std

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using fitted statistics."""
        if not self._initialized:
            return X
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X - self._mean) / self._std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Denormalize features."""
        if not self._initialized:
            return X
        return X * self._std + self._mean
