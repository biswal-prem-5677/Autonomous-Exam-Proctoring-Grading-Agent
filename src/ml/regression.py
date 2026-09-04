"""Linear regression and random forest regression for performance prediction."""

import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LinearRegressionScratch:
    """Linear regression with gradient descent."""

    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionScratch':
        """Train using gradient descent (MSE loss)."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.max_iterations):
            predictions = X @ self.weights + self.bias
            errors = predictions - y
            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)

            dw = (2 / m) * (X.T @ errors)
            db = (2 / m) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        if self.weights is None:
            raise RuntimeError("Model not trained.")
        return X @ self.weights + self.bias


class PerformancePredictor:
    """Wrapper for performance prediction using ensemble methods."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        self.linear_model = SkLinearRegression()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PerformancePredictor':
        """Train the performance prediction model."""
        self.model.fit(X, y)
        self.linear_model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict performance scores."""
        if not self._fitted:
            raise RuntimeError("Model not trained.")
        return self.model.predict(X)

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction combining RF and linear model."""
        rf_pred = self.model.predict(X)
        linear_pred = self.linear_model.predict(X)
        return 0.7 * rf_pred + 0.3 * linear_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        return {
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
        }
