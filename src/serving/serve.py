"""
serve.py — FastAPI service: /predict, /health, /batch-score, /ask-complaints, /customer-intel, /metrics
Usage: uvicorn src.serving.serve:app --reload --port 8000
"""
import sys
import os
import json
import time
import joblib
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

from src.serving.schemas import (
    CustomerFeatures, PredictResponse, BatchScoreResponse,
    ComplaintQuestion, ComplaintAnswer,
    CustomerIntelRequest, CustomerIntelResponse, ComplaintTheme,
    HealthResponse, MetricsResponse,
)
from src.data_pipeline.features import build_features, load_encoding_maps

ARTIFACTS_DIR = Path("artifacts")

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
_model = None
_model_meta = {}
_rag_loaded = False
_metrics = {
    "total_requests": 0,
    "predict_requests": 0,
    "rag_requests": 0,
    "error_count": 0,
    "latencies": [],
    "predictions": [],
    "rag_scores": [],
    "empty_retrievals": 0,
    "refusals": 0,
}


def load_ml_model():
    global _model, _model_meta
    model_path = ARTIFACTS_DIR / "model.joblib"
    meta_path = ARTIFACTS_DIR / "model_meta.json"

    if not model_path.exists():
        print("[serve] WARNING: model.joblib not found. Run train.py first.")
        return

    _model = joblib.load(model_path)
    if meta_path.exists():
        with open(meta_path) as f:
            _model_meta = json.load(f)
    load_encoding_maps()
    print(f"[serve] ML model loaded: {_model_meta.get('model_version', 'unknown')}")


def load_rag():
    global _rag_loaded
    try:
        from src.rag.retrieve import load_index
        load_index()
        _rag_loaded = True
        print("[serve] RAG index loaded.")
    except Exception as e:
        print(f"[serve] WARNING: Could not load RAG index: {e}")
        _rag_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ml_model()
    load_rag()
    yield


app = FastAPI(
    title="Customer Intelligence Platform",
    description="ML + RAG service for Meridian Financial",
    version="1.0.0",
    lifespan=lifespan,
)


def customer_to_df(customer: CustomerFeatures) -> pd.DataFrame:
    return pd.DataFrame([customer.model_dump()])


def get_conversion_band(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    elif prob >= 0.35:
        return "medium"
    else:
        return "low"


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    from src.rag.retrieve import get_index_version
    index_ver = "unknown"
    try:
        index_ver = get_index_version()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        model_version=_model_meta.get("model_version", "not_loaded"),
        vector_index_version=index_ver,
        ml_model_loaded=_model is not None,
        rag_index_loaded=_rag_loaded,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerFeatures):
    start = time.time()
    _metrics["total_requests"] += 1
    _metrics["predict_requests"] += 1

    if _model is None:
        _metrics["error_count"] += 1
        raise HTTPException(status_code=503, detail="ML model not loaded. Run train.py first.")

    try:
        df = customer_to_df(customer)
        X = build_features(df, fit=False)

        # Align columns to training features
        expected_cols = _model_meta.get("feature_columns", [])
        if expected_cols:
            for c in expected_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[expected_cols]

        prob = float(_model.predict_proba(X)[0][1])
        threshold = _model_meta.get("threshold", 0.5)
        pred = int(prob >= threshold)
        band = get_conversion_band(prob)

        _metrics["latencies"].append(time.time() - start)
        _metrics["predictions"].append(pred)

        return PredictResponse(
            prediction=pred,
            probability=round(prob, 4),
            threshold=threshold,
            decision="SUBSCRIBE" if pred == 1 else "NO_SUBSCRIBE",
            model_version=_model_meta.get("model_version", "unknown"),
            conversion_band=band,
        )

    except Exception as e:
        _metrics["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-score", response_model=BatchScoreResponse)
async def batch_score(file: UploadFile = File(...)):
    _metrics["total_requests"] += 1

    if _model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        df = pd.read_csv(tmp_path)
        X = build_features(df, fit=False)

        expected_cols = _model_meta.get("feature_columns", [])
        if expected_cols:
            for c in expected_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[expected_cols]

        threshold = _model_meta.get("threshold", 0.5)
        probs = _model.predict_proba(X)[:, 1]
        bands = [get_conversion_band(p) for p in probs]

        output_path = f"data/batch_output_{int(time.time())}.csv"
        df["probability"] = probs
        df["conversion_band"] = bands
        df.to_csv(output_path, index=False)

        return BatchScoreResponse(
            total=len(df),
            high_conversion=bands.count("high"),
            medium_conversion=bands.count("medium"),
            low_conversion=bands.count("low"),
            output_path=output_path,
        )
    except Exception as e:
        _metrics["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-complaints", response_model=ComplaintAnswer)
def ask_complaints(req: ComplaintQuestion):
    _metrics["total_requests"] += 1
    _metrics["rag_requests"] += 1

    if not _rag_loaded:
        raise HTTPException(status_code=503, detail="RAG index not loaded. Run build_index.py first.")

    try:
        from src.rag.retrieve import retrieve
        from src.rag.answer import generate_answer

        ret = retrieve(
            query=req.question,
            top_k=req.top_k or 5,
            product_filter=req.product,
            company_filter=req.company,
        )

        if ret["refused"]:
            _metrics["refusals"] += 1
            _metrics["empty_retrievals"] += 1

        _metrics["rag_scores"].append(ret["max_score"])

        ans = generate_answer(
            question=req.question,
            chunks=ret["chunks"],
            refused=ret["refused"],
            refusal_reason=ret.get("reason", ""),
        )

        return ComplaintAnswer(**ans)

    except Exception as e:
        _metrics["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/customer-intel", response_model=CustomerIntelResponse)
def customer_intel(req: CustomerIntelRequest):
    _metrics["total_requests"] += 1

    if _model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    # Step 1: ML prediction
    pred_resp = predict(req.customer)

    # Step 2: RAG complaint themes
    themes = []
    total_complaints = 0

    if _rag_loaded:
        try:
            from src.rag.retrieve import retrieve

            # Build query from filter
            query_parts = ["customer complaints"]
            if req.product:
                query_parts.append(req.product)
            if req.issue:
                query_parts.append(req.issue)
            theme_query = " ".join(query_parts)

            ret = retrieve(
                query=theme_query,
                top_k=15,
                product_filter=req.product,
            )

            # Group by product/issue to create themes
            if not ret["refused"] and ret["chunks"]:
                from collections import defaultdict
                theme_groups = defaultdict(list)
                for chunk in ret["chunks"]:
                    key = chunk.get("product", "General") + " - " + chunk.get("issue", "Other")
                    theme_groups[key].append(chunk.get("complaint_id", ""))

                total_complaints = len(ret["chunks"])
                themes = [
                    ComplaintTheme(
                        theme=theme,
                        count=len(ids),
                        evidence_ids=list(set(ids))[:5],
                    )
                    for theme, ids in sorted(theme_groups.items(), key=lambda x: -len(x[1]))[:5]
                ]
        except Exception as e:
            print(f"[serve] RAG error in /customer-intel: {e}")

    return CustomerIntelResponse(
        conversion_band=pred_resp.conversion_band,
        conversion_probability=pred_resp.probability,
        model_version=pred_resp.model_version,
        complaint_themes=themes,
        total_complaints_found=total_complaints,
        segment_filter={
            "product": req.product,
            "issue": req.issue,
            "date_from": req.date_from,
            "date_to": req.date_to,
        },
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    latencies = _metrics["latencies"]
    rag_scores = _metrics["rag_scores"]
    preds = _metrics["predictions"]

    return MetricsResponse(
        total_requests=_metrics["total_requests"],
        predict_requests=_metrics["predict_requests"],
        rag_requests=_metrics["rag_requests"],
        error_count=_metrics["error_count"],
        avg_latency_ms=round(np.mean(latencies) * 1000, 2) if latencies else 0.0,
        prediction_distribution={
            "subscribe": sum(preds),
            "no_subscribe": len(preds) - sum(preds),
        },
        rag_retrieval_stats={
            "avg_top_score": round(float(np.mean(rag_scores)), 4) if rag_scores else 0.0,
            "empty_retrievals": _metrics["empty_retrievals"],
            "refusals": _metrics["refusals"],
            "total_rag_queries": _metrics["rag_requests"],
            "hit_rate": round(
                (_metrics["rag_requests"] - _metrics["empty_retrievals"]) / max(1, _metrics["rag_requests"]), 4
            ),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.serve:app", host="0.0.0.0", port=8000, reload=True)
