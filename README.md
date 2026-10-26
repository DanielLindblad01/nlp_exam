# Political speech vocabulary (2022–2024)

Exam-preparation NLP project: compare characteristic vocabulary in Biden and Trump transcripts using transcript-level TF-IDF.

## Data

The dataset was supplied in `data/archive.zip` (files `biden.csv` and `trump.csv`), sourced from Rev.com transcript URLs as recorded in the files. The extraction is limited to segments explicitly attributed to the named politician. Review the source website's licensing terms before publicly redistributing data; the ZIP file is included here for local reproducibility but you may omit it from a public GitHub repository.

## Run

```bash
pip install -r requirements.txt
python analyze.py
```

Outputs are `results/distinctive_terms.png` and `results/top_terms.csv`.

## Interpretation

A positive TF-IDF difference means a term has a greater mean transcript-level TF-IDF score for that politician. These are lexical patterns rather than causal or psychological measures. The 2022–2024 records can still differ in topic, format and transcription quality.
