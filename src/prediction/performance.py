"""Performance prediction — estimates future exam scores from historical data."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PerformancePredictor:
    """Predicts student future performance using ensemble regression.

    Combines Random Forest and Linear Regression for robust predictions.
    Features: past scores, attendance, study hours, difficulty, etc.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        self.linear_model = SkLinearRegression()
        self._fitted = False
        self._feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'PerformancePredictor':
        """Train the performance prediction model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target scores (n_samples,)
            feature_names: Optional names for features
        """
        if X.shape[0] < 3:
            raise ValueError(f"Need at least 3 samples to train, got {X.shape[0]}")

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.rf_model.fit(X, y)
        self.linear_model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, method: str = "ensemble") -> np.ndarray:
        """Predict performance scores.

        Args:
            X: Feature matrix (n_samples, n_features)
            method: 'rf', 'linear', or 'ensemble'
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call fit() first.")

        if method == "rf":
            return self.rf_model.predict(X)
        elif method == "linear":
            return self.linear_model.predict(X)
        else:
            rf_pred = self.rf_model.predict(X)
            linear_pred = self.linear_model.predict(X)
            return 0.7 * rf_pred + 0.3 * linear_pred

    def predict_with_confidence(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict with confidence interval using RF variance."""
        preds = np.array([tree.predict(X)[0] for tree in self.rf_model.estimators_])
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)

        return {
            "predicted_score": float(mean_pred),
            "std_dev": float(std_pred),
            "lower_bound": float(max(0, mean_pred - 1.96 * std_pred)),
            "upper_bound": float(min(100, mean_pred + 1.96 * std_pred)),
            "confidence": "high" if std_pred < 5 else ("medium" if std_pred < 15 else "low"),
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances from the Random Forest model."""
        if not self._fitted:
            return {}
        importances = self.rf_model.feature_importances_
        return dict(zip(self._feature_names,
                        [float(v) for v in importances]))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        if not self._fitted:
            raise RuntimeError("Model not trained.")
        y_pred = self.predict(X)
        return {
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "r2": float(r2_score(y, y_pred)),
        }


class DifficultyEstimator:
    """Estimates question difficulty from response data."""

    def __init__(self):
        self._difficulty_map: Dict[str, float] = {}

    def estimate_from_responses(self, question_id: str,
                                 correct_count: int, total_count: int) -> float:
        """Estimate difficulty as proportion of students who answered correctly.

        Returns 0.0 (easy) to 1.0 (hard).
        """
        if total_count == 0:
            return 0.5
        difficulty = 1.0 - (correct_count / total_count)
        self._difficulty_map[question_id] = round(difficulty, 4)
        return self._difficulty_map[question_id]

    def get_difficulty(self, question_id: str) -> Optional[float]:
        """Get estimated difficulty for a question."""
        return self._difficulty_map.get(question_id)


class KnowledgeTracer:
    """Tracks student knowledge state across topics using Bayesian approach."""

    def __init__(self, initial_mastery: float = 0.5,
                 learning_rate: float = 0.1):
        self.initial_mastery = initial_mastery
        self.learning_rate = learning_rate
        self._knowledge: Dict[str, float] = {}

    def initialize_topic(self, topic: str, mastery: float = None) -> None:
        """Initialize knowledge state for a topic."""
        self._knowledge[topic] = mastery if mastery is not None else self.initial_mastery

    def update(self, topic: str, correct: bool) -> float:
        """Update knowledge state after answering a question.

        Args:
            topic: Topic identifier
            correct: Whether the answer was correct

        Returns:
            Updated mastery probability [0, 1]
        """
        if topic not in self._knowledge:
            self._knowledge[topic] = self.initial_mastery

        current = self._knowledge[topic]
        if correct:
            # Increase mastery (Bayesian-like update toward 1)
            new_mastery = current + self.learning_rate * (1 - current)
        else:
            # Decrease mastery toward 0
            new_mastery = current - self.learning_rate * current

        self._knowledge[topic] = max(0.0, min(1.0, new_mastery))
        return self._knowledge[topic]

    def get_mastery(self, topic: str) -> float:
        """Get current mastery level for a topic."""
        return self._knowledge.get(topic, self.initial_mastery)

    def get_weak_topics(self, threshold: float = 0.4) -> List[str]:
        """Return topics below mastery threshold."""
        return [t for t, m in self._knowledge.items() if m < threshold]

    def get_strong_topics(self, threshold: float = 0.7) -> List[str]:
        """Return topics above mastery threshold."""
        return [t for t, m in self._knowledge.items() if m >= threshold]
