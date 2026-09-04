"""Configuration loader for the exam proctoring agent."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file with environment variable overrides."""
    load_dotenv()

    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with environment variables if present
    _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: dict) -> None:
    """Override config values with environment variables."""
    env_map = {
        "FACE_CONFIDENCE_THRESHOLD": ("proctoring", "face_confidence_threshold", float),
        "MAX_FACE_ABSENCE_SECONDS": ("proctoring", "max_face_absence_seconds", int),
        "HEAD_TURN_THRESHOLD_DEGREES": ("proctoring", "head_turn_threshold_degrees", float),
        "ALPHA_DECAY": ("risk", "alpha_decay", float),
        "ESCALATION_THRESHOLD": ("risk", "escalation_threshold", float),
        "REVIEW_THRESHOLD": ("risk", "review_threshold", float),
        "TFIDF_MIN_SIMILARITY": ("grading", "tfidf_min_similarity", float),
        "KEYWORD_COVERAGE_THRESHOLD": ("grading", "keyword_coverage_threshold", float),
        "NUMERICAL_TOLERANCE": ("grading", "numerical_tolerance", float),
        "DATABASE_PATH": ("database", "path", str),
        "LOG_LEVEL": ("logging", "level", str),
        "LOG_FILE": ("logging", "file", str),
    }

    for env_var, (section, key, cast_type) in env_map.items():
        value = os.getenv(env_var)
        if value is not None:
            if config.get(section) is None:
                config[section] = {}
            config[section][key] = cast_type(value)


# Singleton config instance
_config = None


def get_config() -> dict:
    """Get the singleton config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
