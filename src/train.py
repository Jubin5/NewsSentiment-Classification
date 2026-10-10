"""Train and evaluate a news sentiment classifier.

Expected input CSV columns:
- text: news headline or article text
- sentiment: sentiment label (positive, negative, or neutral)

Run:
    python -m src.train --data data/news_sentiment.csv

The script evaluates only on held-out data and writes metrics and plots to reports/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"
RANDOM_STATE = 42


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate a labelled CSV dataset."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Provide a CSV with 'text' and "
            "'sentiment' columns using --data."
        )

    frame = pd.read_csv(path)
    missing_columns = {TEXT_COLUMN, LABEL_COLUMN} - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {list(frame.columns)}"
        )

    frame = frame[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].fillna("").astype(str).str.strip()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].fillna("").astype(str).str.strip().str.lower()
    frame = frame[(frame[TEXT_COLUMN] != "") & (frame[LABEL_COLUMN] != "")]
    frame = frame.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)

    if frame.empty:
        raise ValueError("No usable rows remain after removing empty values.")
    if frame[LABEL_COLUMN].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

    LOGGER.info("Loaded %d usable records.", len(frame))
    LOGGER.info("Class distribution:\n%s", frame[LABEL_COLUMN].value_counts().to_string())
    return frame


def build_pipeline() -> Pipeline:
    """Create a reproducible TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_evaluation(
    y_true: pd.Series,
    y_pred: Any,
    labels: list[str],
    test_size: int,
    train_size: int,
) -> dict[str, Any]:
    """Save aggregate and per-class metrics, classification report, and plots."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "model": "TF-IDF + Logistic Regression",
        "random_state": RANDOM_STATE,
        "train_samples": train_size,
        "test_samples": test_size,
        "classes": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=labels, zero_division=0),
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("News Sentiment Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    metric_names = ["Accuracy", "Macro F1", "Weighted F1"]
    metric_values = [results["accuracy"], results["f1_macro"], results["f1_weighted"]]
    plt.figure(figsize=(7, 4))
    sns.barplot(x=metric_names, y=metric_values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Held-out Test Performance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "summary_metrics.png", dpi=160)
    plt.close()

    return results


def main() -> None:
    """Train, test, save the model, and report evaluation metrics."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "news_sentiment.csv",
        help="Path to a labelled CSV with text and sentiment columns.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5.")

    frame = load_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[TEXT_COLUMN],
        frame[LABEL_COLUMN],
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=frame[LABEL_COLUMN],
    )

    pipeline = build_pipeline()
    LOGGER.info("Training on %d samples; testing on %d samples.", len(x_train), len(x_test))
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    labels = sorted(frame[LABEL_COLUMN].unique().tolist())
    results = save_evaluation(y_test, predictions, labels, len(x_test), len(x_train))

    import joblib

    joblib.dump(pipeline, MODEL_DIR / "sentiment_pipeline.joblib")
    LOGGER.info("Evaluation results:")
    LOGGER.info("Accuracy: %.4f", results["accuracy"])
    LOGGER.info("Macro precision: %.4f", results["precision_macro"])
    LOGGER.info("Macro recall: %.4f", results["recall_macro"])
    LOGGER.info("Macro F1-score: %.4f", results["f1_macro"])
    LOGGER.info("Weighted F1-score: %.4f", results["f1_weighted"])
    LOGGER.info("Full report saved to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
