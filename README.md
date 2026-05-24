# Data Directory

## Sources

### ML Lane — UCI Bank Marketing Dataset
- **Source**: https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Target**: `y` — whether client subscribed to a term deposit (yes/no)
- **Records**: ~45,211 rows, 16 features
- **Download**: Run `python src/data_pipeline/ingest.py`
### RAG Lane — CFPB Consumer Complaints
- **Source**: https://www.consumerfinance.gov/data-research/consumer-complaints/
- **API**: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/
- **Sample size**: 5,000–10,000 records (narratives only)
- **Download**: Run `python src/data_pipeline/ingest.py`

## Directory Structure
```
data/
  raw/           # Original downloaded data (small samples only)
  processed/     # Cleaned, validated, feature-engineered data
  samples/       # Very small samples for CI (100 rows)
  README.md
```
