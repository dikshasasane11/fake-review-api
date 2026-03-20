"""
Fake Review Detection API
==========================
FastAPI service that exposes the fake-review ML model via REST endpoints.

Endpoints
---------
  GET  /               → health check
  GET  /docs           → Swagger UI (auto-generated)
  POST /predict        → classify a single review
  POST /predict/batch  → classify up to 50 reviews at once
  GET  /model/info     → model metadata & feature info
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
import os

from models import FakeReviewDetector
from schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ModelInfoResponse, HealthResponse,
)

# ── Startup: train model once, reuse across all requests ─────────────────────

detector: FakeReviewDetector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    print("Training model on startup …")
    detector = FakeReviewDetector()
    detector.train("tripadvisor_hotel_reviews.csv")
    print("Model ready.")
    yield
    print("Shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Fake Review Detector",
    description = "ML-powered API to detect fake hotel reviews using TF-IDF + Logistic Regression.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serves the main web page."""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns API status and whether the model is loaded."""
    return HealthResponse(
        status      = "ok",
        model_ready = detector is not None and detector.is_trained,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Classify a single review as FAKE or GENUINE.

    - **text**: the review text (min 3 characters)
    - **rating**: star rating 1–5 (default 4)
    """
    if not detector or not detector.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    t0     = time.perf_counter()
    result = detector.predict(req.text, req.rating)
    ms     = round((time.perf_counter() - t0) * 1000, 2)

    return PredictResponse(
        label       = result["label"],
        is_fake     = result["is_fake"],
        confidence  = result["confidence"],
        signals     = result["signals"],
        latency_ms  = ms,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(req: BatchPredictRequest):
    """
    Classify up to 50 reviews in a single request.

    Each item needs a **text** field; **rating** is optional (defaults to 4).
    """
    if not detector or not detector.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready yet.")
    if len(req.reviews) > 50:
        raise HTTPException(status_code=400, detail="Max 50 reviews per batch.")

    t0      = time.perf_counter()
    results = [detector.predict(r.text, r.rating) for r in req.reviews]
    ms      = round((time.perf_counter() - t0) * 1000, 2)

    predictions = [
        PredictResponse(
            label      = r["label"],
            is_fake    = r["is_fake"],
            confidence = r["confidence"],
            signals    = r["signals"],
            latency_ms = None,
        )
        for r in results
    ]

    fake_count = sum(1 for r in results if r["is_fake"])

    return BatchPredictResponse(
        predictions = predictions,
        total       = len(results),
        fake_count  = fake_count,
        real_count  = len(results) - fake_count,
        latency_ms  = ms,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Returns model metadata, training stats, and top features."""
    if not detector or not detector.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    info = detector.get_info()
    return ModelInfoResponse(**info)
