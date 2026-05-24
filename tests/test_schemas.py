"""tests/test_schemas.py — Pydantic schema validation tests."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.serving.schemas import CustomerFeatures, ComplaintQuestion


VALID_CUSTOMER = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 200,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def test_valid_customer_accepted():
    c = CustomerFeatures(**VALID_CUSTOMER)
    assert c.age == 35


def test_age_too_low_rejected():
    bad = {**VALID_CUSTOMER, "age": 10}
    with pytest.raises(Exception):
        CustomerFeatures(**bad)


def test_age_too_high_rejected():
    bad = {**VALID_CUSTOMER, "age": 150}
    with pytest.raises(Exception):
        CustomerFeatures(**bad)


def test_missing_required_field_rejected():
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "age"}
    with pytest.raises(Exception):
        CustomerFeatures(**bad)


def test_complaint_question_valid():
    q = ComplaintQuestion(question="What are the main mortgage complaints?")
    assert q.question.startswith("What")


def test_complaint_question_too_short():
    with pytest.raises(Exception):
        ComplaintQuestion(question="hi")


def test_complaint_question_with_filters():
    q = ComplaintQuestion(
        question="What mortgage issues exist?",
        product="Mortgage",
        company="Wells Fargo",
        top_k=10,
    )
    assert q.product == "Mortgage"
    assert q.top_k == 10
