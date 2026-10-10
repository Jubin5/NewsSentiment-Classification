"""Load the trained pipeline and predict news sentiment."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"


def load_model():
    """Load the trained scikit-learn pipeline or explain how to create it."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained pipeline not found at {MODEL_PATH}. "
            "Run the training script first."
        )
    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str) -> str:
    """Predict sentiment for a non-empty headline."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Headline must be a non-empty string.")
    return str(load_model().predict([text.strip()])[0])
