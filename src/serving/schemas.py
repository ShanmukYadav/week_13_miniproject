"""
schemas.py — Pydantic schemas for all API endpoints.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


# ── /predict ──────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    job: str = Field(..., example="management")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="tertiary")
    default: str = Field(..., example="no")
    balance: int = Field(..., example=1500)
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")
    contact: str = Field(..., example="cellular")
    day: int = Field(..., ge=1, le=31, example=15)
    month: str = Field(..., example="may")
    duration: int = Field(..., ge=0, example=200)
    campaign: int = Field(..., ge=1, example=2)
    pdays: int = Field(..., example=-1)
    previous: int = Field(..., ge=0, example=0)
    poutcome: str = Field(..., example="unknown")

    @field_validator("age")
    @classmethod
    def age_valid(cls, v):
        if not (18 <= v <= 100):
            raise ValueError("age must be between 18 and 100")
        return v


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    decision: str  # "SUBSCRIBE" or "NO_SUBSCRIBE"
    model_version: str
    conversion_band: str  # "high", "medium", "low"


# ── /batch-score ──────────────────────────────────────────
class BatchScoreResponse(BaseModel):
    total: int
    high_conversion: int
    medium_conversion: int
    low_conversion: int
    output_path: str


# ── /ask-complaints ───────────────────────────────────────
class ComplaintQuestion(BaseModel):
    question: str = Field(..., min_length=5, max_length=500, example="What are the main mortgage complaints?")
    product: Optional[str] = Field(None, example="Mortgage")
    company: Optional[str] = Field(None, example="Bank of America")
    date_from: Optional[str] = Field(None, example="2023-01-01")
    date_to: Optional[str] = Field(None, example="2024-01-01")
    top_k: Optional[int] = Field(5, ge=1, le=20)


class ComplaintAnswer(BaseModel):
    answer: str
    evidence_ids: List[str]
    evidence_sufficiency: str  # "SUFFICIENT" / "PARTIAL" / "INSUFFICIENT - REFUSED"
    prompt_version: str
    retrieval_score: float
    token_count: int
    latency_ms: float


# ── /customer-intel ───────────────────────────────────────
class CustomerIntelRequest(BaseModel):
    customer: CustomerFeatures
    product: Optional[str] = Field(None, example="Mortgage")
    issue: Optional[str] = Field(None, example="payment")
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class ComplaintTheme(BaseModel):
    theme: str
    count: int
    evidence_ids: List[str]


class CustomerIntelResponse(BaseModel):
    conversion_band: str
    conversion_probability: float
    model_version: str
    complaint_themes: List[ComplaintTheme]
    total_complaints_found: int
    segment_filter: dict


# ── /health ───────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_version: str
    vector_index_version: str
    ml_model_loaded: bool
    rag_index_loaded: bool


# ── /metrics ──────────────────────────────────────────────
class MetricsResponse(BaseModel):
    total_requests: int
    predict_requests: int
    rag_requests: int
    error_count: int
    avg_latency_ms: float
    prediction_distribution: dict
    rag_retrieval_stats: dict
