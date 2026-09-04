"""
ML Training Pipeline.

Orchestrates the complete ML workflow:
    1. Training data generation
    2. Model training (logistic regression + anomaly)
    3. Model evaluation
    4. Threshold calibration
    5. Model persistence

This pipeline produces the trained models needed by the agent for:
    - Behavioral risk prediction (logistic regression)
    - Anomaly detection (Z-score + Mahalanobis)

Usage:
    pipeline = TrainingPipeline()
    pipeline.run(n_normal=500, n_suspicious=500)
    pipeline.save_models("models/")
    # Or load:
    pipeline.load_models("models/")
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict
import json

from src.ml.training_data import TrainingDataGenerator, generate_training_data
from src.ml.logistic import LogisticRegressionScratch
from src.ml.anomaly import AnomalyDetector
from src.ml.calibration import ModelCalibrator, ModelEvaluator


@dataclass
class TrainingConfig:
    """Configuration for ML training pipeline."""
    n_normal: int = 500
    n_suspicious: int = 500
    test_ratio: float = 0.2
    random_seed: int = 42
    logistic_learning_rate: float = 0.01
    logistic_max_iterations: int = 1000
    logistic_regularization: float = 0.1
    anomaly_zscore_threshold: float = 2.5
    anomaly_confidence: float = 0.95


@dataclass
class TrainingResults:
    """Results from model training."""
    dataset_stats: Dict
    logistic_metrics: Dict
    anomaly_metrics: Dict
    optimal_threshold: float
    threshold_metrics: Dict
    feature_importance: Dict


class TrainingPipeline:
    """End-to-end ML training pipeline."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.generator = TrainingDataGenerator(seed=self.config.random_seed)

        # Models
        self.logistic_model: Optional[LogisticRegressionScratch] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.calibrator = ModelCalibrator()

        # Data
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Results
        self.results: Optional[TrainingResults] = None

    def run(self) -> TrainingResults:
        """Execute the full training pipeline."""
        # Step 1: Generate training data
        print("Generating training data...")
        X, y = self.generator.generate_dataset(
            n_normal=self.config.n_normal,
            n_suspicious=self.config.n_suspicious,
            seed=self.config.random_seed,
        )

        # Dataset statistics
        dataset_stats = self.generator.compute_statistics(X, y)
        print(f"Dataset: {len(y)} samples, {np.sum(y==1)} suspicious")

        # Step 2: Train/test split
        X_train, X_test, y_train, y_test = self.generator.train_test_split(
            X, y, test_ratio=self.config.test_ratio, seed=self.config.random_seed
        )
        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train, X_test, y_train, y_test
        )
        print(f"Train: {len(y_train)}, Test: {len(y_test)}")

        # Step 3: Train logistic regression
        print("Training logistic regression...")
        self.logistic_model = LogisticRegressionScratch(
            learning_rate=self.config.logistic_learning_rate,
            max_iterations=self.config.logistic_max_iterations,
            regularization=self.config.logistic_regularization,
        )
        self.logistic_model.fit(X_train, y_train)

        # Evaluate logistic regression
        y_pred_logistic = self.logistic_model.predict(X_test)
        y_proba_logistic = self.logistic_model.predict_proba(X_test)
        logistic_metrics = ModelEvaluator.evaluate_classifier(
            y_test, y_pred_logistic, y_proba_logistic
        )
        print(f"Logistic - Accuracy: {logistic_metrics['accuracy']:.3f}, "
              f"F1: {logistic_metrics['f1']:.3f}, "
              f"ROC-AUC: {logistic_metrics['roc_auc']:.3f}")

        # Step 4: Train anomaly detector on normal data
        print("Training anomaly detector...")
        X_normal_train = X_train[y_train == 0]
        self.anomaly_detector = AnomalyDetector(
            zscore_threshold=self.config.anomaly_zscore_threshold,
            mahalanobis_confidence=self.config.anomaly_confidence,
        )
        self.anomaly_detector.fit(X_normal_train)

        # Evaluate anomaly detection
        anomaly_scores = self.anomaly_detector.score(X_test)
        y_pred_anomaly = (anomaly_scores > 0.5).astype(int)
        anomaly_metrics = ModelEvaluator.evaluate_classifier(
            y_test, y_pred_anomaly, anomaly_scores
        )
        print(f"Anomaly - Accuracy: {anomaly_metrics['accuracy']:.3f}, "
              f"F1: {anomaly_metrics['f1']:.3f}")

        # Step 5: Find optimal threshold
        print("Finding optimal threshold...")
        optimal_threshold, threshold_metrics = self.calibrator.find_optimal_threshold(
            y_test, y_proba_logistic, metric="f1"
        )
        print(f"Optimal threshold: {optimal_threshold:.3f}")

        # Step 6: Compute feature importance (coefficients)
        feature_importance = self._compute_feature_importance()

        # Assemble results
        self.results = TrainingResults(
            dataset_stats=dataset_stats,
            logistic_metrics=logistic_metrics,
            anomaly_metrics=anomaly_metrics,
            optimal_threshold=optimal_threshold,
            threshold_metrics=threshold_metrics,
            feature_importance=feature_importance,
        )

        return self.results

    def _compute_feature_importance(self) -> Dict:
        """Compute feature importance from logistic coefficients."""
        if self.logistic_model is None:
            return {}

        coefs = self.logistic_model.weights
        abs_coefs = np.abs(coefs)
        normalized = abs_coefs / abs_coefs.sum()

        feature_names = [
            "face_present", "face_confidence", "face_count", "absence_ratio",
            "head_yaw", "head_pitch", "head_roll",
            "audio_rms", "audio_db", "audio_zcr", "audio_flag", "spectral",
            "kb_idle", "kb_rate", "key_hold", "key_lat",
            "mouse_speed", "mouse_dist", "click_rate", "mouse_idle",
            "multi_face", "progress",
        ]

        importance = {
            name: float(coef)
            for name, coef in zip(feature_names, coefs)
        }
        ranked = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
        return ranked

    def save_models(self, model_dir: str) -> None:
        """Save trained models to disk."""
        import pickle

        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save logistic model
        if self.logistic_model:
            with open(path / "logistic_model.pkl", "wb") as f:
                pickle.dump(self.logistic_model, f)

        # Save anomaly detector
        if self.anomaly_detector:
            with open(path / "anomaly_model.pkl", "wb") as f:
                pickle.dump(self.anomaly_detector, f)

        # Save results
        if self.results:
            results_dict = {
                "optimal_threshold": self.results.optimal_threshold,
                "logistic_metrics": self.results.logistic_metrics,
                "anomaly_metrics": self.results.anomaly_metrics,
                "feature_importance": self.results.feature_importance,
            }
            with open(path / "training_results.json", "w") as f:
                json.dump(results_dict, f, indent=2)

        print(f"Models saved to {model_dir}")

    def load_models(self, model_dir: str) -> None:
        """Load trained models from disk."""
        import pickle

        path = Path(model_dir)

        # Load logistic model
        logistic_path = path / "logistic_model.pkl"
        if logistic_path.exists():
            with open(logistic_path, "rb") as f:
                self.logistic_model = pickle.load(f)
            print(f"Loaded logistic model from {logistic_path}")

        # Load anomaly detector
        anomaly_path = path / "anomaly_model.pkl"
        if anomaly_path.exists():
            with open(anomaly_path, "rb") as f:
                self.anomaly_detector = pickle.load(f)
            print(f"Loaded anomaly model from {anomaly_path}")

    def predict_risk(self, features: np.ndarray) -> Dict:
        """Predict risk using both models."""
        if self.logistic_model is None or self.anomaly_detector is None:
            raise RuntimeError("Models not trained. Run pipeline.run() first.")

        # Logistic prediction
        logistic_proba = self.logistic_model.predict_proba(features.reshape(1, -1))[0]

        # Anomaly score
        anomaly_score = self.anomaly_detector.score(features.reshape(1, -1))[0]

        # Combined risk
        combined_risk = 0.6 * logistic_proba + 0.4 * anomaly_score

        return {
            "logistic_risk": float(logistic_proba),
            "anomaly_score": float(anomaly_score),
            "combined_risk": float(combined_risk),
            "is_anomalous": bool(anomaly_score > 0.5),
        }


def run_full_pipeline(
    n_normal: int = 500,
    n_suspicious: int = 500,
    model_dir: str = "models",
) -> TrainingResults:
    """Convenience function to run the full pipeline."""
    config = TrainingConfig(
        n_normal=n_normal,
        n_suspicious=n_suspicious,
    )
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    pipeline.save_models(model_dir)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("ML TRAINING PIPELINE")
    print("=" * 60)
    results = run_full_pipeline(n_normal=500, n_suspicious=500)
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Optimal threshold: {results.optimal_threshold:.3f}")
    print(f"Logistic F1: {results.logistic_metrics['f1']:.3f}")
    print(f"Anomaly F1: {results.anomaly_metrics['f1']:.3f}")
    print("=" * 60)
