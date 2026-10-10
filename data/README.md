# Dataset

Place the labelled dataset here as `news_sentiment.csv`.

Required columns:

- `text`: news headline or article text
- `sentiment`: ground-truth sentiment label (for example, `positive`, `negative`, or `neutral`)

Example schema:

```csv
text,sentiment
"Stocks rise after positive economic data",positive
"Company reports record losses amid weak demand",negative
"Central bank announces its scheduled policy meeting",neutral
```

These rows are illustrative examples only, not a proper training or evaluation dataset. Do not use them to claim model accuracy. The repository did not include the original labelled dataset, so add the actual dataset (or identify its source) before training.
