"""
answer.py — Generate grounded LLM answers from retrieved complaint chunks using Groq.
"""
import os
import time
import json
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PROMPT_VERSION = "v1.2"
GROQ_MODEL = "llama3-8b-8192"
MAX_TOKENS = 512

try:
    from groq import Groq
    _groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    _groq_client = None


SYSTEM_PROMPT = """You are a complaint intelligence analyst for Meridian Financial.
Your job is to answer questions about customer complaints using ONLY the provided complaint excerpts.

Rules:
1. Base your answer ONLY on the provided complaint excerpts. Do not invent facts.
2. Always reference specific complaints by their IDs (e.g., "Complaint #12345 states...").
3. If the evidence is insufficient, say so clearly.
4. Do NOT provide legal or financial advice.
5. Do NOT expose personal information.
6. Be concise: 3-5 sentences maximum.
7. End with: "Evidence IDs: [list of IDs used]"
"""


def build_context(chunks: list) -> str:
    """Format retrieved chunks into a context block."""
    parts = []
    for i, chunk in enumerate(chunks):
        cid = chunk.get("complaint_id", f"chunk_{i}")
        product = chunk.get("product", "")
        score = chunk.get("similarity_score", 0)
        text = chunk.get("text", "")[:500]  # Truncate to avoid token overflow
        parts.append(f"[Complaint #{cid} | Product: {product} | Relevance: {score:.2f}]\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: list,
    refused: bool = False,
    refusal_reason: str = "",
) -> dict:
    """
    Generate a grounded answer using Groq.
    Returns dict with answer, evidence_ids, token_count, latency_ms.
    """
    start_time = time.time()

    # Handle refusal
    if refused or not chunks:
        return {
            "answer": f"I cannot answer this question from the available complaint records. {refusal_reason}",
            "evidence_ids": [],
            "evidence_sufficiency": "INSUFFICIENT - REFUSED",
            "prompt_version": PROMPT_VERSION,
            "retrieval_score": 0.0,
            "token_count": 0,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
        }

    if not _groq_client:
        # Fallback: rule-based summary without LLM
        evidence_ids = list({c.get("complaint_id", "") for c in chunks})
        themes = list({c.get("product", "Unknown") for c in chunks})
        answer = (
            f"Based on {len(chunks)} complaint records, the main themes relate to: {', '.join(themes)}. "
            f"Key complaints include issues described in records #{', #'.join(evidence_ids[:3])}. "
            f"[Note: LLM generation unavailable — set GROQ_API_KEY for full analysis.] "
            f"Evidence IDs: {evidence_ids}"
        )
        return {
            "answer": answer,
            "evidence_ids": evidence_ids,
            "evidence_sufficiency": "PARTIAL",
            "prompt_version": PROMPT_VERSION + "_fallback",
            "retrieval_score": float(chunks[0].get("similarity_score", 0)),
            "token_count": 0,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
        }

    context = build_context(chunks)
    user_message = f"""Question: {question}

Complaint Evidence:
{context}

Answer the question using only the above evidence. Be specific and cite complaint IDs."""

    try:
        response = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1,
        )
        answer_text = response.choices[0].message.content.strip()
        token_count = response.usage.total_tokens if response.usage else 0

        # Extract evidence IDs from answer + context
        evidence_ids = list({c.get("complaint_id", "") for c in chunks if c.get("complaint_id")})

        # Determine sufficiency
        max_score = max(c.get("similarity_score", 0) for c in chunks)
        if max_score >= 0.5:
            sufficiency = "SUFFICIENT"
        elif max_score >= 0.35:
            sufficiency = "PARTIAL"
        else:
            sufficiency = "INSUFFICIENT"

    except Exception as e:
        answer_text = f"LLM generation failed: {str(e)}"
        evidence_ids = []
        token_count = 0
        sufficiency = "ERROR"

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "answer": answer_text,
        "evidence_ids": evidence_ids,
        "evidence_sufficiency": sufficiency,
        "prompt_version": PROMPT_VERSION,
        "retrieval_score": float(chunks[0].get("similarity_score", 0) if chunks else 0),
        "token_count": token_count,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    from src.rag.retrieve import load_index, retrieve
    load_index()

    q = "What are common mortgage payment and foreclosure complaints?"
    result = retrieve(q)
    ans = generate_answer(q, result["chunks"], result["refused"], result.get("reason", ""))
    print(f"Q: {q}")
    print(f"A: {ans['answer']}")
    print(f"Evidence IDs: {ans['evidence_ids']}")
    print(f"Sufficiency: {ans['evidence_sufficiency']}")
    print(f"Latency: {ans['latency_ms']}ms | Tokens: {ans['token_count']}")
