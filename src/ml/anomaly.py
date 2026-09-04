"""Anomaly detection — Z-score aggregation and Mahalanobis distance."""

import numpy as np
from typing import Optional, Dict, Tuple


class AnomalyDetector:
    """Detects anomalous behavior patterns using statistical methods.

    Combines Z-score analysis with Mahalanobis distance for
    multivariate anomaly detection.
    """

    def __init__(self, zscore_threshold: float = 2.5,
                 mahalanobis_confidence: float = 0.95):
        self.zscore_threshold = zscore_threshold
        self.mahalanobis_confidence = mahalanobis_confidence
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._fitted = False
        self._chi2_threshold: float = 0.0

    def fit(self, X: np.ndarray) -> 'AnomalyDetector':
        """Fit the anomaly detector on normal behavior data.

        Args:
            X: Normal behavior feature matrix (n_samples, n_features)

        Returns:
            self
        """
        self._mean = np.mean(X, axis=0)
        self._cov = np.cov(X.T)
        # Regularize covariance matrix
        self._cov += np.eye(self._cov.shape[0]) * 1e-6
        self._cov_inv = np.linalg.inv(self._cov)

        # Chi-squared threshold
        from scipy.stats import chi2
        self._chi2_threshold = chi2.ppf(self.mahalanobis_confidence, df=X.shape[1])
        self._fitted = True
        return self

    def compute_zscore(self, x: np.ndarray) -> np.ndarray:
        """Compute Z-scores for each feature dimension."""
        if self._mean is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        return (x - self._mean) / (np.std(self._mean) + 1e-10)

    def compute_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each sample.

        The Mahalanobis distance measures how far a point is from
        the distribution of normal behavior, accounting for covariance.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        diff = X - self._mean
        md_squared = np.sum(diff @ self._cov_inv * diff, axis=1)
        return np.sqrt(md_squared)

    def is_anomaly(self, X: np.ndarray, method: str = "combined") -> Tuple[bool, Dict]:
        """Determine if samples are anomalous.

        Args:
            X: Feature vector(s) to check
            method: 'zscore', 'mahalanobis', or 'combined'

        Returns:
            (is_anomaly, details_dict)
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        details = {}

        if method in ("zscore", "combined"):
            zscores = self.compute_zscore(X)
            max_zscore = np.max(np.abs(zscores))
            zscore_anomaly = max_zscore > self.zscore_threshold
            details["zscore"] = float(max_zscore)
            details["zscore_anomaly"] = bool(zscore_anomaly)

        if method in ("mahalanobis", "combined"):
            md = self.compute_mahalanobis(X)
            md_anomaly = md > self._chi2_threshold
            details["mahalanobis_distance"] = float(md[0])
            details["mahalanobis_anomaly"] = bool(md_anomaly[0])

        if method == "combined":
            is_anomaly = details.get("zscore_anomaly", False) or details.get("mahalanobis_anomaly", False)
            details["combined_anomaly"] = bool(is_anomaly)
        elif method == "zscore":
            is_anomaly = details["zscore_anomaly"]
        else:
            is_anomaly = details["mahalanobis_anomaly"]

        return bool(is_anomaly), details

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (0 = normal, 1 = highly anomalous)."""
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        md = self.compute_mahalanobis(X)
        # Normalize to 0-1 range based on chi2 threshold
        scores = np.clip(md / self._chi2_threshold, 0, 1)
        return scores
