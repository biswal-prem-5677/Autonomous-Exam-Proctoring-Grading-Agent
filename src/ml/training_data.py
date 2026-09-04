"""
Training Data Generator for the Autonomous Exam Proctoring Agent.

Generates labeled behavioral feature vectors for supervised ML training.
Produces both normal and suspicious behavioral patterns for model training.

Mathematical Framework:
────────────────────────────────────────────────────────────────────────────
Each sample is a 22-dimensional feature vector x ∈ R²² with label y ∈ {0, 1}.

Classes:
    0 — NORMAL behavior
    1 — SUSPICIOUS behavior (controlled behavioral anomalies)

Feature Vector (22 dimensions):
    [0]  face_present         — binary (0 or 1)
    [1]  face_confidence      — float [0, 1]
    [2]  face_count           — int, capped at 4 → normalized
    [3]  absence_ratio        — float [0, 1]
    [4]  head_yaw             — float, degrees
    [5]  head_pitch           — float, degrees
    [6]  head_roll            — float, degrees
    [7]  audio_rms            — float [0, 1]
    [8]  audio_db             — float, normalized
    [9]  audio_zcr            — float [0, 1]
    [10] audio_threshold_flag  — binary
    [11] spectral_centroid     — float, normalized
    [12] keyboard_idle_flag    — binary
    [13] keystroke_rate        — float, normalized
    [14] key_hold_mean         — float, normalized
    [15] key_latency_mean      — float, normalized
    [16] mouse_speed           — float, normalized
    [17] mouse_distance         — float, normalized
    [18] click_rate            — float, normalized
    [19] mouse_idle_flag       — binary
    [20] multi_face_events     — int, normalized
    [21] session_progress      — float [0, 1]

Label Generation:
    Normal samples:   y = 0
    Suspicious:       y = 1 (with configurable label noise)

Usage:
    generator = TrainingDataGenerator(seed=42)
    X, y = generator.generate_dataset(n_normal=500, n_suspicious=500)
    X_train, X_test, y_train, y_test = generator.train_test_split(X, y)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List
import json


class BehaviorClass(Enum):
    """Controlled behavioral scenario labels."""
    NORMAL = 0
    LOOKING_AWAY = 1
    ABSENCE = 2
    MULTIPLE_PERSON = 3
    AUDIO_ACTIVITY = 4
    TAB_SWITCH = 5
    UNUSUAL_TYPING = 6
    UNUSUAL_MOUSE = 7
    MULTI_SIGNAL = 8


# ─── Feature Index Constants ────────────────────────────────────────────────

IX_FACE_PRESENT = 0
IX_FACE_CONF = 1
IX_FACE_COUNT = 2
IX_ABSENCE_RATIO = 3
IX_HEAD_YAW = 4
IX_HEAD_PITCH = 5
IX_HEAD_ROLL = 6
IX_AUDIO_RMS = 7
IX_AUDIO_DB = 8
IX_AUDIO_ZCR = 9
IX_AUDIO_FLAG = 10
IX_SPECTRAL = 11
IX_KB_IDLE = 12
IX_KB_RATE = 13
IX_KEY_HOLD = 14
IX_KEY_LAT = 15
IX_MOUSE_SPD = 16
IX_MOUSE_DIST = 17
IX_CLICK_RATE = 18
IX_MOUSE_IDLE = 19
IX_MULTI_FACE = 20
IX_PROGRESS = 21

N_FEATURES = 22


# ─── Normal Behavior Profiles ──────────────────────────────────────────────

@dataclass
class BehaviorProfile:
    """Statistical profile for generating behavioral samples."""
    # Face
    face_present_prob: float = 0.95
    face_confidence_mu: float = 0.88
    face_confidence_sigma: float = 0.08
    face_count_mu: float = 1.0
    face_count_sigma: float = 0.15
    absence_ratio_mu: float = 0.02
    absence_ratio_sigma: float = 0.04

    # Head pose
    head_yaw_mu: float = 0.0
    head_yaw_sigma: float = 0.12
    head_pitch_mu: float = 0.0
    head_pitch_sigma: float = 0.08
    head_roll_mu: float = 0.0
    head_roll_sigma: float = 0.05

    # Audio (quiet exam room)
    audio_rms_mu: float = 0.05
    audio_rms_sigma: float = 0.04
    audio_db_mu: float = 0.08
    audio_db_sigma: float = 0.06
    audio_zcr_mu: float = 0.10
    audio_zcr_sigma: float = 0.06
    audio_flag_prob: float = 0.08
    spectral_mu: float = 0.15
    spectral_sigma: float = 0.10

    # Keyboard (steady typing)
    kb_idle_prob: float = 0.15
    kb_rate_mu: float = 0.50
    kb_rate_sigma: float = 0.18
    key_hold_mu: float = 0.12
    key_hold_sigma: float = 0.05
    key_lat_mu: float = 0.10
    key_lat_sigma: float = 0.04

    # Mouse (normal navigation)
    mouse_speed_mu: float = 0.30
    mouse_speed_sigma: float = 0.18
    mouse_dist_mu: float = 0.25
    mouse_dist_sigma: float = 0.15
    click_rate_mu: float = 0.08
    click_rate_sigma: float = 0.06
    mouse_idle_prob: float = 0.20

    # Temporal
    multi_face_mu: float = 0.0
    multi_face_sigma: float = 0.01
    progress_mu: float = 0.50
    progress_sigma: float = 0.25


class TrainingDataGenerator:
    """Generates labeled behavioral feature vectors for ML training."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.normal_profile = BehaviorProfile()

    # ─── Public API ─────────────────────────────────────────────────────────

    def generate_dataset(
        self,
        n_normal: int = 500,
        n_suspicious: int = 500,
        noise_std: float = 0.05,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete labeled dataset.

        Args:
            n_normal: Number of normal behavior samples.
            n_suspicious: Number of suspicious behavior samples.
            noise_std: Standard deviation of Gaussian noise added to all features.
            seed: Optional random seed override.

        Returns:
            X: (n_normal + n_suspicious, 22) feature matrix.
            y: (n_normal + n_suspicious,) binary label vector.
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Normal samples
        X_normal = self._generate_normal_batch(n_normal, noise_std)
        y_normal = np.zeros(n_normal, dtype=np.int64)

        # Suspicious samples — mixed scenarios
        n_per_class = n_suspicious // len(BehaviorScenario)
        X_suspicious_list = []
        y_suspicious_list = []

        for scenario in BehaviorScenario:
            n_samples = n_per_class + (0 if scenario != list(BehaviorScenario)[-1]
                                       else n_suspicious % len(BehaviorScenario))
            X_scenario = self._generate_suspicious_batch(scenario, n_samples, noise_std)
            X_suspicious_list.append(X_scenario)
            y_suspicious_list.append(np.ones(n_samples, dtype=np.int64))

        X_suspicious = np.vstack(X_suspicious_list)
        y_suspicious = np.hstack(y_suspicious_list)

        # Combine and shuffle
        X = np.vstack([X_normal, X_suspicious])
        y = np.hstack([y_normal, y_suspicious])

        perm = self.rng.permutation(len(y))
        return X[perm], y[perm]

    def generate_scenario_dataset(
        self,
        scenario: BehaviorScenario,
        n_samples: int = 100,
        noise_std: float = 0.05,
    ) -> np.ndarray:
        """Generate samples for a single suspicious scenario.

        Useful for ablation studies: "How does the model perform
        on looking-away-only data vs. multi-signal data?"
        """
        return self._generate_suspicious_batch(scenario, n_samples, noise_std)

    def generate_normal_dataset(
        self,
        n_samples: int = 100,
        noise_std: float = 0.05,
    ) -> np.ndarray:
        """Generate normal behavior samples."""
        return self._generate_normal_batch(n_samples, noise_std)

    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = 0.2,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train/test split preserving class balance."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng

        n_test = int(len(y) * test_ratio)
        perm = rng.permutation(len(y))

        # Stratify by class
        mask_0 = y == 0
        mask_1 = y == 1

        test_idx_0 = perm[mask_0[perm]][:n_test // 2]
        train_idx_0 = perm[mask_0[perm]][n_test // 2:]
        test_idx_1 = perm[mask_1[perm]][:n_test // 2]
        train_idx_1 = perm[mask_1[perm]][n_test // 2:]

        train_idx = np.hstack([train_idx_0, train_idx_1])
        test_idx = np.hstack([test_idx_0, test_idx_1])
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def cross_validation_folds(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        seed: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Generate stratified k-fold cross-validation splits."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = self.rng

        folds = []
        # Stratify
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        rng.shuffle(idx_0)
        rng.shuffle(idx_1)

        fold_size_0 = len(idx_0) // n_folds
        fold_size_1 = len(idx_1) // n_folds

        for i in range(n_folds):
            # Test fold
            test_start_0 = i * fold_size_0
            test_end_0 = test_start_0 + fold_size_0 if i < n_folds - 1 else len(idx_0)
            test_start_1 = i * fold_size_1
            test_end_1 = test_start_1 + fold_size_1 if i < n_folds - 1 else len(idx_1)

            test_idx = np.hstack([
                idx_0[test_start_0:test_end_0],
                idx_1[test_start_1:test_end_1],
            ])
            train_idx = np.hstack([
                np.concatenate([idx_0[:test_start_0], idx_0[test_end_0:]]),
                np.concatenate([idx_1[:test_start_1], idx_1[test_end_1:]]),
            ])
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)
            folds.append((X[train_idx], X[test_idx], y[train_idx], y[test_idx]))

        return folds

    def save_dataset(self, X: np.ndarray, y: np.ndarray, filepath: str) -> None:
        """Save dataset to JSON for inspection and replay."""
        data = {
            "features": X.tolist(),
            "labels": y.tolist(),
            "n_samples": len(y),
            "n_features": N_FEATURES,
            "class_distribution": {
                "normal": int(np.sum(y == 0)),
                "suspicious": int(np.sum(y == 1)),
            },
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return np.array(data["features"]), np.array(data["labels"])

    # ─── Private Generation Methods ──────────────────────────────��──────────

    def _generate_normal_batch(
        self, n: int, noise: float
    ) -> np.ndarray:
        """Generate normal behavior feature vectors."""
        p = self.normal_profile
        X = np.zeros((n, N_FEATURES))

        for i in range(n):
            # Face
            X[i, IX_FACE_PRESENT] = 1 if self.rng.random() < p.face_present_prob else 0
            X[i, IX_FACE_CONF] = np.clip(
                self.rng.normal(p.face_confidence_mu, p.face_confidence_sigma), 0, 1
            )
            if X[i, IX_FACE_PRESENT] == 0:
                X[i, IX_FACE_CONF] = 0.0
            X[i, IX_FACE_COUNT] = np.clip(
                self.rng.normal(p.face_count_mu, p.face_count_sigma), 0, 4
            )
            X[i, IX_ABSENCE_RATIO] = max(0,
                self.rng.normal(p.absence_ratio_mu, p.absence_ratio_sigma)
            )

            # Head pose — small random deviations when face is present
            if X[i, IX_FACE_PRESENT]:
                X[i, IX_HEAD_YAW] = np.clip(
                    self.rng.normal(p.head_yaw_mu, p.head_yaw_sigma), -1, 1
                )
                X[i, IX_HEAD_PITCH] = np.clip(
                    self.rng.normal(p.head_pitch_mu, p.head_pitch_sigma), -1, 1
                )
            else:
                X[i, IX_HEAD_YAW] = 0.0
                X[i, IX_HEAD_PITCH] = 0.0
            X[i, IX_HEAD_ROLL] = np.clip(
                self.rng.normal(p.head_roll_mu, p.head_roll_sigma), -1, 1
            )

            # Audio
            X[i, IX_AUDIO_RMS] = max(0,
                self.rng.normal(p.audio_rms_mu, p.audio_rms_sigma)
            )
            X[i, IX_AUDIO_DB] = max(0,
                self.rng.normal(p.audio_db_mu, p.audio_db_sigma)
            )
            X[i, IX_AUDIO_ZCR] = max(0,
                self.rng.normal(p.audio_zcr_mu, p.audio_zcr_sigma)
            )
            X[i, IX_AUDIO_FLAG] = 1 if self.rng.random() < p.audio_flag_prob else 0
            X[i, IX_SPECTRAL] = max(0,
                self.rng.normal(p.spectral_mu, p.spectral_sigma)
            )

            # Keyboard
            X[i, IX_KB_IDLE] = 1 if self.rng.random() < p.kb_idle_prob else 0
            if not X[i, IX_KB_IDLE]:
                X[i, IX_KB_RATE] = np.clip(
                    self.rng.normal(p.kb_rate_mu, p.kb_rate_sigma), 0, 1
                )
            else:
                X[i, IX_KB_RATE] = 0.0
            X[i, IX_KEY_HOLD] = max(0,
                self.rng.normal(p.key_hold_mu, p.key_hold_sigma)
            )
            X[i, IX_KEY_LAT] = max(0,
                self.rng.normal(p.key_lat_mu, p.key_lat_sigma)
            )

            # Mouse
            X[i, IX_MOUSE_SPD] = max(0,
                self.rng.normal(p.mouse_speed_mu, p.mouse_speed_sigma)
            )
            X[i, IX_MOUSE_DIST] = max(0,
                self.rng.normal(p.mouse_dist_mu, p.mouse_dist_sigma)
            )
            X[i, IX_CLICK_RATE] = max(0,
                self.rng.normal(p.click_rate_mu, p.click_rate_sigma)
            )
            X[i, IX_MOUSE_IDLE] = 1 if self.rng.random() < p.mouse_idle_prob else 0

            # Temporal
            X[i, IX_MULTI_FACE] = max(0,
                self.rng.normal(p.multi_face_mu, p.multi_face_sigma)
            )
            X[i, IX_PROGRESS] = np.clip(
                self.rng.normal(p.progress_mu, p.progress_sigma), 0, 1
            )

        # Add Gaussian noise
        X += self.rng.normal(0, noise_std, X.shape)
        X = np.clip(X, 0, 1)
        X[:, IX_HEAD_YAW] = np.clip(X[:, IX_HEAD_YAW], -1, 1)
        X[:, IX_HEAD_PITCH] = np.clip(X[:, IX_HEAD_PITCH], -1, 1)
        X[:, IX_HEAD_ROLL] = np.clip(X[:, IX_HEAD_ROLL], -1, 1)

        return X

    def _generate_suspicious_batch(
        self, scenario: BehaviorScenario, n: int, noise: float
    ) -> np.ndarray:
        """Generate suspicious behavior samples by perturbing normal profile."""
        p = self.normal_profile
        X = self._generate_normal_batch(n, noise)

        # Apply scenario-specific perturbations
        if scenario == BehaviorScenario.LOOKING_AWAY:
            # Large yaw deviations, reduced face confidence
            X[:, IX_HEAD_YAW] = self.rng.uniform(0.35, 0.85, n)
            X[:, IX_HEAD_PITCH] = self.rng.uniform(-0.5, 0.5, n)
            X[:, IX_FACE_CONF] *= self.rng.uniform(0.5, 0.8, n)

        elif scenario == BehaviorScenario.ABSENCE:
            # No face detection
            X[:, IX_FACE_PRESENT] = 0
            X[:, IX_FACE_CONF] = 0.0
            X[:, IX_ABSENCE_RATIO] = self.rng.uniform(0.3, 0.8, n)
            X[:, IX_HEAD_YAW] = 0.0
            X[:, IX_HEAD_PITCH] = 0.0

        elif scenario == BehaviorScenario.MULTIPLE_PERSON:
            # Multiple faces detected
            X[:, IX_FACE_COUNT] = self.rng.choice([2, 3, 4], n)
            X[:, IX_FACE_CONF] *= self.rng.uniform(0.6, 0.9, n)
            X[:, IX_MULTI_FACE] = self.rng.uniform(0.3, 0.9, n)
            X[:, IX_KB_RATE] *= self.rng.uniform(0.3, 0.7, n)

        elif scenario == BehaviorScenario.AUDIO_ACTIVITY:
            # Elevated audio signals
            X[:, IX_AUDIO_RMS] = self.rng.uniform(0.25, 0.70, n)
            X[:, IX_AUDIO_DB] = self.rng.uniform(0.20, 0.65, n)
            X[:, IX_AUDIO_ZCR] = self.rng.uniform(0.15, 0.60, n)
            X[:, IX_AUDIO_FLAG] = self.rng.choice([0, 1], n, p=[0.2, 0.8])
            X[:, IX_SPECTRAL] = self.rng.uniform(0.30, 0.80, n)

        elif scenario == BehaviorScenario.TAB_SWITCH:
            # Reduced keyboard activity, irregular patterns
            X[:, IX_KB_RATE] = self.rng.uniform(0.05, 0.25, n)
            X[:, IX_KB_IDLE] = self.rng.choice([0, 1], n, p=[0.3, 0.7])
            X[:, IX_KEY_LAT] *= self.rng.uniform(1.5, 3.0, n)
            X[:, IX_MOUSE_SPD] *= self.rng.uniform(0.5, 1.5, n)

        elif scenario == BehaviorScenario.UNUSUAL_TYPING:
            # Abnormally high or low typing rates, erratic hold times
            X[:, IX_KB_RATE] = self.rng.uniform(0.80, 1.0, n)
            X[:, IX_KEY_HOLD] = self.rng.uniform(0.01, 0.06, n)
            X[:, IX_KEY_LAT] = self.rng.uniform(0.01, 0.04, n)
            X[:, IX_KB_IDLE] = 0

        elif scenario == BehaviorScenario.UNUSUAL_MOUSE:
            # Abnormally high mouse speed or idle periods
            X[:, IX_MOUSE_SPD] = self.rng.uniform(0.60, 1.0, n)
            X[:, IX_MOUSE_DIST] = self.rng.uniform(0.50, 1.0, n)
            X[:, IX_CLICK_RATE] = self.rng.uniform(0.20, 0.50, n)
            X[:, IX_MOUSE_IDLE] = self.rng.choice([0, 1], n, p=[0.7, 0.3])

        elif scenario == BehaviorScenario.MULTI_SIGNAL:
            # Multiple signals active simultaneously — highest risk
            X[:, IX_HEAD_YAW] = self.rng.uniform(0.25, 0.65, n)
            X[:, IX_AUDIO_RMS] = self.rng.uniform(0.15, 0.50, n)
            X[:, IX_AUDIO_FLAG] = self.rng.choice([0, 1], n, p=[0.3, 0.7])
            X[:, IX_KB_RATE] *= self.rng.uniform(0.4, 0.8, n)
            X[:, IX_MOUSE_SPD] *= self.rng.uniform(1.2, 2.0, n)
            X[:, IX_CLICK_RATE] *= self.rng.uniform(1.5, 3.0, n)

        # Re-clip to valid ranges
        X = np.clip(X, 0, 1)
        X[:, IX_HEAD_YAW] = np.clip(X[:, IX_HEAD_YAW], -1, 1)
        X[:, IX_HEAD_PITCH] = np.clip(X[:, IX_HEAD_PITCH], -1, 1)
        X[:, IX_HEAD_ROLL] = np.clip(X[:, IX_HEAD_ROLL], -1, 1)

        return X

    # ─── Feature Statistics ─────────────────────────────────────────────────

    @staticmethod
    def compute_statistics(X: np.ndarray, y: np.ndarray) -> dict:
        """Compute dataset statistics for documentation and reporting."""
        n_samples, n_features = X.shape
        n_pos = int(np.sum(y))
        n_neg = int(np.sum(y == 0))

        stats = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_ratio": round(n_pos / n_samples, 4),
            "feature_means_normal": X[y == 0].mean(axis=0).round(4).tolist(),
            "feature_means_suspicious": X[y == 1].mean(axis=0).round(4).tolist(),
            "feature_stds": X.std(axis=0).round(4).tolist(),
            "per_feature_separation": (
                np.abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
                / (X.std(axis=0) + 1e-8)
            ).round(4).tolist(),
        }
        return stats


# ─── Enums ──────────────────────────────────────────────────────────────────

class BehaviorScenario(Enum):
    """Controlled behavioral anomaly scenarios for training data generation."""
    NORMAL = "normal"
    LOOKING_AWAY = "looking_away"
    ABSENCE = "absence"
    MULTIPLE_PERSON = "multiple_person"
    AUDIO_ACTIVITY = "audio_activity"
    TAB_SWITCH = "tab_switch"
    UNUSUAL_TYPING = "unusual_typing"
    UNUSUAL_MOUSE = "unusual_mouse"
    MULTI_SIGNAL = "multi_signal"


# ─── Convenience Functions ──────────────────────────────────────────────────

def generate_training_data(
    n_normal: int = 500,
    n_suspicious: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-call dataset generation."""
    gen = TrainingDataGenerator(seed=seed)
    return gen.generate_dataset(n_normal=n_normal, n_suspicious=n_suspicious)


def generate_evaluation_data(
    n_samples: int = 200,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate held-out evaluation data (different seed for distribution shift)."""
    gen = TrainingDataGenerator(seed=seed)
    return gen.generate_dataset(
        n_normal=n_samples // 2,
        n_suspicious=n_samples // 2,
    )
