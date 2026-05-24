# Data Sources

## UCI Bank Marketing Dataset
- Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Used for: ML model — predicting campaign conversion (term deposit subscription)
- File: data/raw/bank-full.csv (not committed, download via ingest.py)

## CFPB Consumer Complaint Database
- Source: https://www.consumerfinance.gov/data-research/consumer-complaints/
- Used for: LLM/RAG — complaint intelligence Q&A
- File: data/raw/complaints_sample.csv (not committed, download via ingest.py)

## Note
Raw data files are excluded from git (.gitignore).
Run `python src/data_pipeline/ingest.py` to download them.