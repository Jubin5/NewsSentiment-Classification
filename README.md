# News Sentiment Classification

A reproducible news headline sentiment classifier using TF-IDF features and Logistic Regression, with a Streamlit app and evaluation pipeline.

## Improvements in this rebuild

- Reports **accuracy, precision, recall, macro F1-score, and weighted F1-score**.
- Produces class-wise metrics, a confusion matrix, and a metrics JSON file.
- Uses a stratified hold-out test split and a fixed random seed.
- Fits text vectorization only on training data using a scikit-learn pipeline.
- Validates the labelled dataset and removes empty rows and exact text/label duplicates.

## Project structure

```text
├── app_py.py
├── data/
│   └── README.md
├── models/                  # Trained model saved here (not committed)
├── reports/                 # Generated evaluation results
├── src/
│   ├── __init__.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_predict.py
├── requirements.txt
└── .gitignore
```

## 1. Set up (Windows PowerShell)

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Add your labelled dataset

Place your CSV at `data/news_sentiment.csv`. It must include:

| Column | Meaning |
|---|---|
| `text` | News headline or article text |
| `sentiment` | Ground-truth class, e.g. `positive`, `negative`, `neutral` |

See [data/README.md](data/README.md) for a schema example. If the dataset uses other column names, update `TEXT_COLUMN` and `LABEL_COLUMN` in `src/train.py`.

**Important:** the current repository did not contain the original labelled training dataset or training script. Metrics cannot be calculated honestly until you supply the dataset or identify its source. Examples in the schema are illustrative only and must not be used to claim evaluation performance.

## 3. Train and evaluate

```powershell
python -m src.train --data data/news_sentiment.csv
```

The script trains TF-IDF + Logistic Regression and evaluates it on a stratified held-out test set (default 20%). It saves the fitted pipeline to `models/sentiment_pipeline.joblib` and generates:

- `reports/metrics.json` — accuracy, macro/weighted precision, recall and F1, class labels, test/train sizes and confusion matrix counts.
- `reports/classification_report.txt` — class-wise precision, recall, F1 and support.
- `reports/confusion_matrix.png` — confusion matrix heatmap.
- `reports/summary_metrics.png` — accuracy and F1 comparison plot.

Scores depend on the actual dataset and will only be available after the command runs successfully. They are not pre-filled or guaranteed.

## 4. Run Streamlit

```powershell
streamlit run app_py.py
```

The UI loads the newly trained pipeline from `models/sentiment_pipeline.joblib`. Train the model first. The evaluation section reads the generated metrics from `reports/metrics.json`.

## 5. Tests

```powershell
pytest
```

## Metric guide

- **Accuracy:** fraction of predictions that match their true labels.
- **Precision:** fraction of predictions for a given class that are correct.
- **Recall:** fraction of true instances of a class that were detected.
- **F1-score:** harmonic mean of precision and recall.
- **Macro average:** treats each class equally, useful when classes are imbalanced.
- **Weighted average:** averages class scores based on class support.
- **Confusion matrix:** counts correct and incorrect predictions for each class.

## Limitations

Performance depends on label quality, class balance, data leakage, and how similar future headlines are to the training data. A random split is a baseline; for duplicate-heavy data or time-sensitive news, use duplicate-aware or chronological validation too.
