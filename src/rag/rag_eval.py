"""
rag_eval.py — 10-question RAG evaluation with pass/fail and evidence checks.
Usage: python src/rag/rag_eval.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EVAL_QUESTIONS = [
    {
        "id": "Q01",
        "question": "What are the most common complaints about mortgage payments?",
        "expected_topics": ["payment", "mortgage", "foreclosure", "loan"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
    {
        "id": "Q02",
        "question": "What issues do customers report about credit card billing errors?",
        "expected_topics": ["credit", "billing", "charge", "dispute"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
    {
        "id": "Q03",
        "question": "How do customers describe problems with debt collection harassment?",
        "expected_topics": ["debt", "collection", "harass", "call"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
    {
        "id": "Q04",
        "question": "What complaints exist about student loan servicer communication?",
        "expected_topics": ["student", "loan", "servicer", "communication"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
    {
        "id": "Q05",
        "question": "What are customer complaints regarding checking account fraud?",
        "expected_topics": ["fraud", "account", "unauthorized", "transaction"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
    {
        "id": "Q06",
        "question": "What issues do people report about credit report inaccuracies?",
        "expected_topics": ["credit", "report", "inaccurate", "bureau"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
    {
        "id": "Q07",
        "question": "What complaints mention difficulty getting a loan modification?",
        "expected_topics": ["modification", "loan", "payment", "hardship"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
    {
        "id": "Q08",
        "question": "How are complaints about bank account closures described?",
        "expected_topics": ["account", "close", "bank", "notice"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
    # Refusal test: irrelevant topic
    {
        "id": "Q09",
        "question": "What is the quantum entanglement theory in particle physics?",
        "expected_topics": [],
        "expect_evidence": False,
        "expect_refusal": True,
    },
    # Partial/borderline test
    {
        "id": "Q10",
        "question": "What complaints exist about insurance products sold by banks?",
        "expected_topics": ["insurance", "bank", "product"],
        "expect_evidence": True,
        "expect_refusal": False,
    },
]


def run_eval() -> dict:
    from src.rag.retrieve import load_index, retrieve
    from src.rag.answer import generate_answer

    print("[rag_eval] Loading index...")
    load_index()

    results = []
    passed = 0
    failed = 0

    print("\n── RAG Evaluation ──\n")

    for q_item in EVAL_QUESTIONS:
        qid = q_item["id"]
        question = q_item["question"]
        expect_refusal = q_item["expect_refusal"]
        expect_evidence = q_item["expect_evidence"]
        expected_topics = q_item["expected_topics"]

        # Retrieve
        ret = retrieve(question)
        ans = generate_answer(question, ret["chunks"], ret["refused"], ret.get("reason", ""))

        # Evaluate
        test_results = {}

        # Test 1: Refusal behavior matches expectation
        if expect_refusal:
            t1 = ret["refused"]
            test_results["refusal_correct"] = t1
        else:
            t1 = not ret["refused"]
            test_results["not_refused"] = t1

        # Test 2: Evidence IDs present when expected
        if expect_evidence:
            t2 = len(ans["evidence_ids"]) > 0
            test_results["evidence_ids_present"] = t2
        else:
            t2 = True
            test_results["evidence_ids_na"] = True

        # Test 3: Answer mentions at least one expected topic (if not refusal)
        if expected_topics and not ret["refused"]:
            answer_lower = ans["answer"].lower()
            topic_hit = any(t.lower() in answer_lower for t in expected_topics)
            test_results["topic_relevance"] = topic_hit
            t3 = topic_hit
        else:
            t3 = True
            test_results["topic_relevance_na"] = True

        # Test 4: Sufficiency note present
        t4 = bool(ans.get("evidence_sufficiency"))
        test_results["sufficiency_note_present"] = t4

        # Overall pass
        all_pass = all([t1, t2, t3, t4])
        if all_pass:
            passed += 1
        else:
            failed += 1

        result_entry = {
            "id": qid,
            "question": question,
            "pass": all_pass,
            "answer_preview": ans["answer"][:200] + ("..." if len(ans["answer"]) > 200 else ""),
            "evidence_ids": ans["evidence_ids"][:5],
            "evidence_sufficiency": ans["evidence_sufficiency"],
            "retrieval_score": ret["max_score"],
            "refused": ret["refused"],
            "test_details": test_results,
            "latency_ms": ans["latency_ms"],
        }
        results.append(result_entry)

        status = "✓ PASS" if all_pass else "✗ FAIL"
        print(f"  {status} | {qid}: {question[:60]}...")
        if not all_pass:
            print(f"         Tests: {test_results}")

    # Summary
    total = len(EVAL_QUESTIONS)
    pass_rate = passed / total

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 4),
        "results": results,
    }

    print(f"\n── Summary ──")
    print(f"  Passed: {passed}/{total} ({pass_rate:.0%})")
    print(f"  Failed: {failed}/{total}")

    # Save report
    report_path = Path("docs/rag_eval_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[rag_eval] Report saved → {report_path}")

    return report


if __name__ == "__main__":
    report = run_eval()
    sys.exit(0 if report["pass_rate"] >= 0.7 else 1)
