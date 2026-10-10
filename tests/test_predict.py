"""Basic tests for sentiment prediction input validation."""
import pytest

from src.predict import predict_sentiment


def test_empty_headline_is_rejected() -> None:
    """Empty headlines should fail before attempting to load a model."""
    with pytest.raises(ValueError, match="non-empty"):
        predict_sentiment("   ")
