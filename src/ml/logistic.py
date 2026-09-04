"""Logistic Regression classifier for behavioral risk scoring.

Implements gradient descent from scratch (no sklearn for the core algorithm).
Uses sklearn for utilities (train_test_split, metrics) only.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class LogisticRegressionScratch:
    """Logistic regression implemented with gradient descent."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 regularization: float = 0.1, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.tolerance = tolerance
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss with L2 regularization."""
        m = X.shape[0]
        z = X @ self.weights + self.bias
        predictions = self._sigmoid(z)
        # Clip to avoid log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)

        loss = -np.mean(
            y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
        )
        # L2 regularization
        loss += (self.regularization / (2 * m)) * np.sum(self.weights ** 2)
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        """Train the logistic regression model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)

        Returns:
            self
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        prev_loss = float('inf')

        for i in range(self.max_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            predictions = self._sigmoid(z)

            # Gradients with L2 regularization
            dw = (1 / m) * (X.T @ (predictions - y)) + (self.regularization / m) * self.weights
            db = (1 / m) * np.sum(predictions - y)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            # Convergence check
            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        if self.weights is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Compute evaluation metrics."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)) if len(set(y)) > 1 else 0.0,
        }
