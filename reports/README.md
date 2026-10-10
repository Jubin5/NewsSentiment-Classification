# Evaluation outputs

Run `python -m src.train --data data/news_sentiment.csv` to generate:

- `metrics.json`: aggregate and class-wise metrics.
- `classification_report.txt`: precision, recall, F1-score, and support per class.
- `confusion_matrix.png`: confusion-matrix heatmap.
- `summary_metrics.png`: accuracy and F1-score visualization.

These outputs are computed from the held-out test set. They are not committed in advance because valid measurements require the real labelled dataset.
