# Architecture — Customer Intelligence Platform

## Overview

Two services — an ML conversion scorer and a RAG complaint assistant — share a single governed pipeline with versioned data, a CI gate, monitoring, and one integration endpoint that ties both together.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│   UCI Bank Marketing (45k rows)    CFPB Complaints (5k rows)   │
└────────────────┬────────────────────────────┬───────────────────┘
                 │                            │
                 ▼                            ▼
        ingest.py + validate.py        ingest.py + chunk
        features.py (reusable)         build_index.py (FAISS)
                 │                            │
                 ▼                            ▼
┌───────────────────────┐      ┌──────────────────────────┐
│      ML LANE          │      │        RAG LANE           │
│                       │      │                           │
│  train.py             │      │  sentence-transformers    │
│  baseline: LogReg     │      │  embed → FAISS index      │
│  improved: XGBoost    │      │                           │
│  MLflow tracking      │      │  retrieve.py              │
│  evaluate.py          │      │  similarity threshold     │
│  promotion gate       │      │  answer.py (Groq LLM)     │
│  (PR-AUC +3pp)        │      │  cited evidence IDs       │
└──────────┬────────────┘      └────────────┬─────────────┘
           │                                │
           ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI  (serve.py)                          │
│                                                                 │
│  GET  /health          → status, model version, index version  │
│  POST /predict         → probability, decision, model version  │
│  POST /batch-score     → scored CSV batch                      │
│  POST /ask-complaints  → answer, evidence IDs, sufficiency     │
│  POST /customer-intel  → ML band + complaint themes (cited)    │
│  GET  /metrics         → latency, counts, RAG stats            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        Docker Compose                GitHub Actions CI
        (local deployment)            tests + validation
                 │                    promotion gate
                 ▼                           │
        monitoring/                          ▼
        ml_drift.py    ←──────────  blocks worse model
        rag_monitor.py
```

---

## Component Breakdown

### Data Pipeline
- `ingest.py` — downloads UCI Bank Marketing (nested zip handled) and CFPB complaints via 4-strategy fallback. Saves sha256 manifest for reproducibility.
- `validate.py` — schema checks, missing value rules, 5+ business rule validations using Pandera/custom checks. Fails loudly so bad data never reaches training.
- `features.py` — reusable, tested feature functions shared between train time and serve time. No train-serving skew.

### ML Lane
- `train.py` — trains a Logistic Regression baseline and an XGBoost improved model. Logs params, metrics and artifacts to MLflow.
- `evaluate.py` — reports ROC-AUC, PR-AUC, F1, calibration and confusion matrix. Runs the relative promotion gate: XGBoost must beat baseline by at least 3 PR-AUC percentage points or it is blocked.
- Promoted model artifact saved to `mlruns/` and loaded at serve time by version string.

### RAG Lane
- `build_index.py` — chunks complaint narratives (512 token chunks, 50 token overlap), embeds with `sentence-transformers/all-MiniLM-L6-v2`, builds a FAISS flat index. Persists index + document ID map.
- `retrieve.py` — cosine similarity retrieval with a hard threshold (0.35). Returns top-k chunks with IDs. Refuses to answer if no chunk crosses the threshold.
- `answer.py` — constructs a grounded prompt from retrieved chunks, calls Groq LLM (llama3-8b-8192), returns answer + cited record IDs + evidence sufficiency note.
- `rag_eval.py` — 10 question eval set with expected evidence. Records PASS/FAIL per question.

### Serving
- `serve.py` — FastAPI app with all 5 endpoints. Pydantic schemas for input validation. Model and index loaded once at startup.
- `/customer-intel` is the integration endpoint: runs ML prediction to get a conversion band, then retrieves complaint themes for the matching product/issue segment, returning both with cited IDs.

### CI/CD
- GitHub Actions workflow on every push: installs dependencies, runs `pytest tests/`, runs data validation on `data/samples/bank_sample.csv`, runs the promotion gate check.
- A deliberately worse model run is included in `tests/` to demonstrate the gate blocks it.

### Monitoring
- `ml_drift.py` — generates an Evidently drift report by comparing the training distribution against a synthetically shifted dataset. Saved as HTML report.
- `rag_monitor.py` — logs retrieval hit-rate, empty-retrieval count, average top-k similarity score, refusal rate, token count and latency per request.

### Deployment
- Docker Compose runs both the FastAPI service and MLflow UI locally.
- `Dockerfile` packages the FastAPI app with model artifacts and FAISS index baked in.

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector store | FAISS (local) | Free, fast, reproducible — no cloud account needed |
| LLM | Groq (llama3-8b) | Free tier, low latency, no GPU required |
| ML tracking | MLflow local | Zero cost, full artifact versioning |
| Embeddings | all-MiniLM-L6-v2 | Runs on CPU, good quality for complaint text |
| Promotion gate | Relative PR-AUC +3pp | Tied to baseline so threshold is meaningful |
| Deployment | Docker Compose | Reproducible from fresh clone, no cloud credits burned |

---

## Data Flow Summary

```
Fresh clone
    → pip install -r requirements.txt
    → python src/data_pipeline/ingest.py        (downloads data)
    → python src/data_pipeline/validate.py      (checks data)
    → python src/training/train.py              (trains + tracks)
    → python src/training/evaluate.py           (gates + promotes)
    → python src/rag/build_index.py             (builds FAISS)
    → uvicorn src.serving.serve:app             (serves API)
    → python monitoring/ml_drift.py             (drift report)
    → python monitoring/rag_monitor.py          (RAG metrics)
```
