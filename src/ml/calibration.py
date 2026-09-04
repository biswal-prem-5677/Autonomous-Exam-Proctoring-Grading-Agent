"""Model calibration and evaluation utilities."""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score,
    confusion_matrix, classification_report
)


class ModelCalibrator:
    """Calibrates model predictions for reliable risk scores."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._calibrated = False

    def calibrate(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply calibration to probability predictions.

        Uses sigmoid calibration (Platt scaling) approach.
        """
        from sklearn.linear_model import LogisticRegression

        # Platt scaling
        calibrator = LogisticRegression()
        calibrator.fit(y_proba.reshape(-1, 1), y_true)
        calibrated_proba = calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        self._calibrated = True
        return y_true, calibrated_proba

    def evaluate_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                           threshold: float) -> Dict:
        """Evaluate model at a specific decision threshold."""
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "threshold": threshold,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        }

    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                                metric: str = "f1") -> Tuple[float, Dict]:
        """Find the optimal decision threshold for a given metric."""
        from sklearn.metrics import f1_score, precision_score, recall_score

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0.0
        best_metrics = {}

        metric_funcs = {
            "f1": f1_score,
            "precision": precision_score,
            "recall": recall_score,
        }
        score_func = metric_funcs.get(metric, f1_score)

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            score = score_func(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = self.evaluate_threshold(y_true, y_proba, threshold)

        return float(best_threshold), best_metrics


class ModelEvaluator:
    """Comprehensive model evaluation for all three ML modules."""

    @staticmethod
    def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray,
                            y_proba: np.ndarray) -> Dict:
        """Evaluate a binary classifier."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else 0.0,
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        }

    @staticmethod
    def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate a regression model."""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
