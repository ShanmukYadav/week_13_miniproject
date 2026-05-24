# Decision Log — Customer Intelligence Platform

This log records the approaches I tried, what I rejected, and why. A shallow demo hides failures; this log shows the actual engineering decisions.

---

## Model Selection

**What I tried:**
Logistic Regression as baseline, then Random Forest, then XGBoost as the improved candidate.

**What I rejected:**
Random Forest initially looked promising but training time was 3-4x longer than XGBoost for marginal gain on PR-AUC. Neural network approaches (sklearn MLPClassifier) were tried briefly but required too much hyperparameter tuning for the time available and gave no meaningful improvement on this tabular dataset.

**What I chose:**
XGBoost as the improved model. It consistently beat Logistic Regression by more than 3 PR-AUC points on the bank marketing dataset, which is the promotion gate threshold. It also handles class imbalance better out of the box via `scale_pos_weight`.

**Known limitation:**
The promotion gate is relative (beat baseline by 3pp), not absolute. If the baseline itself is weak, a mediocre improved model could still pass. A future hardening would add an absolute floor (e.g. PR-AUC must exceed 0.45 regardless).

---

## Metrics Choice

**What I rejected:**
Accuracy alone. The bank marketing dataset is imbalanced (~88% no, ~12% yes). A model that always predicts "no" would score 88% accuracy and be completely useless for campaign targeting.

**What I chose:**
PR-AUC as the primary gate metric, supported by ROC-AUC, F1, calibration plot, and confusion matrix. PR-AUC is the right metric for imbalanced classification where the positive class (subscription) is what the business cares about. Calibration matters because the output is a probability used for campaign scoring, not just a binary decision.

---

## Vector Store

**What I rejected:**
Chroma — had dependency conflicts with the Python version on the dev machine during initial setup. Pinecone and Azure AI Search — both require paid accounts or complex setup, which conflicts with the zero-cost constraint.

**What I chose:**
FAISS flat index (CPU). Runs entirely in-process, persists to disk, loads in under a second, and handles 5,000 complaint chunks comfortably. The index + document ID map are saved so the demo is reproducible from a fresh clone without re-embedding.

---

## LLM Choice

**What I rejected:**
OpenAI GPT-4 — costs money per token, not reproducible on a free tier. Azure OpenAI — requires approved access which takes time. Local LLMs (Ollama/llama.cpp) — too slow on CPU for a demo with sub-10-second response time.

**What I chose:**
Groq API with llama3-8b-8192. Free tier gives enough requests for development and demo. Latency is under 2 seconds per answer. The model is capable enough for grounded summarisation of complaint narratives when given good retrieved context.

---

## Chunking Strategy

**What I tried:**
Fixed 256-token chunks, then 512-token chunks, then 512 with 50-token overlap.

**What I rejected:**
256-token chunks were too short — complaint narratives often span multiple paragraphs and the key evidence was split across chunks, causing the retriever to miss relevant context.

**What I chose:**
512-token chunks with 50-token overlap. This keeps full complaint narratives mostly intact while still allowing fine-grained retrieval. The overlap prevents important sentences at chunk boundaries from being lost.

---

## Deployment

**What I rejected:**
Full Azure deployment — Azure ML managed endpoints require quota that is not guaranteed on a student account. Cloud Run on GCP — not enough time to set up service account credentials and container registry before the deadline.

**What I chose:**
Docker Compose for local deployment. Both services (FastAPI + MLflow UI) run with a single `docker-compose up`. This is reproducible from a fresh clone and demonstrates real containerisation. The Dockerfile bakes in the model artifact and FAISS index so the container is self-contained.

**Honest limitation:**
This is not a cloud deployment. A senior engineer would flag this immediately. The next step would be pushing the Docker image to Azure Container Registry and deploying to Azure Container Apps, which would take roughly 2 hours with proper quota.

---

## Promotion Gate Threshold

**Why 3 PR-AUC percentage points:**
This was calibrated against the actual baseline performance. At 3pp, a model that is only marginally better (noise-level improvement) still gets blocked. At 5pp, the gate became too strict and would have blocked a genuinely better model due to random seed variation in train/test splits.

**What fails if you tighten by another 2 points:**
The improved model would be blocked on approximately 30% of runs due to natural variance from the train/test split. The gate would become unreliable — sometimes passing the same model, sometimes blocking it — which is worse than a slightly lenient threshold.

---

## RAG Refusal Logic

**What I rejected:**
Answering every question regardless of retrieval quality. Tested this early on — when no relevant complaints exist for a query, the LLM would hallucinate plausible-sounding but entirely fabricated complaint summaries. This is unacceptable for a system that claims to be evidence-grounded.

**What I chose:**
Hard similarity threshold of 0.35 cosine similarity. If no retrieved chunk crosses this threshold, the API returns a refusal with an explanation rather than an answer. This was validated against the 10-question eval set — all 10 questions that have real evidence in the corpus return answers; queries about topics not in the corpus correctly refuse.

---

## Known Risks if This Went Live Tomorrow

1. **No authentication on the API.** Any caller can hit /predict or /ask-complaints. In production this needs API key auth or OAuth at minimum.
2. **FAISS index is not updated automatically.** New complaints are never indexed until someone manually runs build_index.py. A production system needs an incremental indexing pipeline.
3. **Groq free tier rate limits.** Under load, the /ask-complaints endpoint would hit rate limits and return errors. Production would need a paid tier or a fallback LLM.
4. **Model drift is detected but not automatically remediated.** ml_drift.py generates a report but nothing triggers a retrain. A human has to notice and act.
5. **No PII audit beyond CFPB's own redaction.** The complaint narratives rely on CFPB having already redacted personal details. This assumption has not been independently verified.

---

## What a Senior MLOps Engineer Would Criticise First

1. No automated retraining trigger — drift is detected but the loop is not closed.
2. Docker Compose instead of a real cloud deployment with health checks, autoscaling, and rollback.
3. FAISS does not support concurrent writes — fine for a demo, breaks under real traffic.
4. MLflow is running locally with no remote artifact store — if the machine is wiped, all experiment history is gone.
5. No request logging to a persistent store — /metrics only shows in-memory stats that reset on restart.
