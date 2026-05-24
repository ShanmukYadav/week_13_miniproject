"""
monitoring/rag_monitor.py — RAG monitoring: hit-rate, latency, refusal rate, token counts.
Usage: python monitoring/rag_monitor.py
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPORTS_DIR = Path("docs")
REPORTS_DIR.mkdir(exist_ok=True)

# Test queries for monitoring run
MONITOR_QUERIES = [
    {"query": "mortgage payment problems", "product": "Mortgage"},
    {"query": "credit card billing dispute", "product": "Credit card"},
    {"query": "debt collection harassment calls", "product": None},
    {"query": "checking account unauthorized transactions", "product": None},
    {"query": "student loan repayment issues", "product": "Student loan"},
    {"query": "foreclosure prevention loan modification", "product": "Mortgage"},
    {"query": "credit report errors inaccurate information", "product": None},
    # Adversarial / off-topic
    {"query": "how to make spaghetti carbonara", "product": None},
    {"query": "what is the weather forecast for tomorrow", "product": None},
    {"query": "cryptocurrency bitcoin investment advice", "product": None},
]


def run_monitoring() -> dict:
    print("[rag_monitor] Loading RAG index...")
    from src.rag.retrieve import load_index, retrieve
    from src.rag.answer import generate_answer

    load_index()

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_queries": 0,
        "hit_count": 0,
        "empty_retrieval_count": 0,
        "refusal_count": 0,
        "latencies_ms": [],
        "retrieval_scores": [],
        "token_counts": [],
        "query_results": [],
    }

    print(f"[rag_monitor] Running {len(MONITOR_QUERIES)} monitoring queries...\n")

    for item in MONITOR_QUERIES:
        query = item["query"]
        product = item.get("product")

        start = time.time()
        ret = retrieve(query, product_filter=product)
        ans = generate_answer(query, ret["chunks"], ret["refused"], ret.get("reason", ""))
        elapsed_ms = round((time.time() - start) * 1000, 2)

        metrics["total_queries"] += 1
        metrics["latencies_ms"].append(elapsed_ms)
        metrics["retrieval_scores"].append(ret["max_score"])
        metrics["token_counts"].append(ans.get("token_count", 0))

        if ret["refused"]:
            metrics["refusal_count"] += 1
            metrics["empty_retrieval_count"] += 1
            hit = False
        else:
            metrics["hit_count"] += 1
            hit = True

        result_entry = {
            "query": query,
            "product_filter": product,
            "hit": hit,
            "refused": ret["refused"],
            "max_score": round(ret["max_score"], 4),
            "n_chunks_retrieved": len(ret["chunks"]),
            "evidence_ids": [c.get("complaint_id", "") for c in ret["chunks"][:3]],
            "answer_preview": ans["answer"][:150] + "...",
            "sufficiency": ans["evidence_sufficiency"],
            "latency_ms": elapsed_ms,
            "token_count": ans.get("token_count", 0),
        }
        metrics["query_results"].append(result_entry)

        status = "✓ HIT" if hit else "✗ MISS/REFUSED"
        print(f"  {status} | score={ret['max_score']:.3f} | {elapsed_ms:.0f}ms | {query[:50]}...")

    # Aggregate stats
    scores = metrics["retrieval_scores"]
    latencies = metrics["latencies_ms"]
    tokens = metrics["token_counts"]

    hit_rate = metrics["hit_count"] / metrics["total_queries"]
    avg_score = float(np.mean(scores)) if scores else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    avg_tokens = float(np.mean([t for t in tokens if t > 0])) if any(t > 0 for t in tokens) else 0.0
    refusal_rate = metrics["refusal_count"] / metrics["total_queries"]

    summary = {
        "hit_rate": round(hit_rate, 4),
        "empty_retrieval_count": metrics["empty_retrieval_count"],
        "refusal_count": metrics["refusal_count"],
        "refusal_rate": round(refusal_rate, 4),
        "avg_top_k_score": round(avg_score, 4),
        "min_score": round(float(min(scores)), 4) if scores else 0.0,
        "max_score": round(float(max(scores)), 4) if scores else 0.0,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "avg_token_count": round(avg_tokens, 1),
        "total_tokens_used": sum(tokens),
        "health": "OK" if hit_rate >= 0.6 and avg_latency < 5000 else "DEGRADED",
    }
    metrics["summary"] = summary

    # Print summary
    print("\n── RAG Monitoring Summary ──")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save report
    report_path = REPORTS_DIR / "rag_monitoring_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[rag_monitor] ✓ Report saved → {report_path}")

    return metrics


if __name__ == "__main__":
    run_monitoring()
