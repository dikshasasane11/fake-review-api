"""
Pydantic schemas for request & response validation.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Request Schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text   : str = Field(..., min_length=3, description="Review text to classify")
    rating : int = Field(4, ge=1, le=5,    description="Star rating (1–5)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text"  : "AMAZING hotel!! Best stay EVER!! Perfect in every way!!!",
                    "rating": 5,
                }
            ]
        }
    }


class BatchPredictRequest(BaseModel):
    reviews: list[PredictRequest] = Field(
        ..., min_length=1, max_length=50,
        description="List of reviews (max 50)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reviews": [
                        {"text": "AMAZING!!! BEST HOTEL EVER!!!", "rating": 5},
                        {"text": "Stayed 3 nights. Clean room, friendly staff. "
                                 "Breakfast was decent. Minor noise issue.", "rating": 4},
                    ]
                }
            ]
        }
    }


# ── Response Schemas ──────────────────────────────────────────────────────────

class TextSignals(BaseModel):
    word_count         : int
    char_count         : int
    avg_word_len       : float
    exclamation_count  : int
    unique_word_ratio  : float
    sentence_count     : int
    avg_sentence_len   : float
    pronoun_count      : int
    superlative_count  : int


class PredictResponse(BaseModel):
    label       : str                  # "FAKE" | "GENUINE"
    is_fake     : bool
    confidence  : Optional[float]      # 0.0 – 1.0
    signals     : TextSignals
    latency_ms  : Optional[float]


class BatchPredictResponse(BaseModel):
    predictions : list[PredictResponse]
    total       : int
    fake_count  : int
    real_count  : int
    latency_ms  : float


class ModelInfoResponse(BaseModel):
    model_name      : str
    algorithm       : str
    tfidf_features  : int
    handcrafted_features : int
    total_features  : int
    training_samples: int
    test_accuracy   : float
    test_precision  : float
    test_recall     : float
    test_f1         : float
    top_fake_features : list[list]
    top_real_features : list[list]


class HealthResponse(BaseModel):
    status      : str
    model_ready : bool
