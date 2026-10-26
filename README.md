# Political speech vocabulary (2022–2024)

Exam-preparation NLP project: compare vocabulary in Biden and Trump transcripts using transcript-level TF-IDF.

## Data

The dataset was found on Kaggle (https://www.kaggle.com/datasets/beridzeg45/usa-political-speeches/data).

## Run

```bash
pip install -r requirements.txt
python analyze.py
```

Outputs are `results/distinctive_terms.png` and `results/top_terms.csv`.

## Interpretation

A positive TF-IDF difference means a term has a greater mean transcript-level TF-IDF score for that politician.
