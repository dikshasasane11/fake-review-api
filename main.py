"""
Fake Review Detection API
==========================
FastAPI service that exposes the fake-review ML model via REST endpoints.

Endpoints
---------
  GET  /               -> serve frontend
  GET  /dashboard      -> analytics dashboard
  GET  /health         -> health check
  GET  /docs           -> Swagger UI
  POST /predict        -> classify a single review
  POST /predict/batch  -> classify up to 50 reviews at once
  GET  /model/info     -> model metadata & feature info
  GET  /analytics      -> usage analytics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager
import time
import os

from models import FakeReviewDetector
from database import Database
from schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ModelInfoResponse, HealthResponse,
)

# -- Startup ------------------------------------------------------------------

detector: FakeReviewDetector = None
db: Database = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, db
    db = Database()
    db.init()
    print("Database ready.")
    print("Training model on startup ...")
    detector = FakeReviewDetector()
    detector.train("tripadvisor_hotel_reviews.csv")
    print("Model ready.")
    yield
    print("Shutting down.")

# -- App ----------------------------------------------------------------------

app = FastAPI(
    title       = "Fake Review Detector",
    description = "ML-powered API to detect fake hotel reviews.",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

BASE = os.path.dirname(os.path.abspath(__file__))

# -- Routes -------------------------------------------------------------------

@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serves the main web page."""
    return FileResponse(os.path.join(BASE, "index.html"))


@app.get("/dashboard", tags=["Frontend"])
def serve_dashboard():
    """Serves the analytics dashboard."""
    return FileResponse(os.path.join(BASE, "dashboard.html"))


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns API status and whether the model is loaded."""
    return HealthResponse(
        status      = "ok",
        model_ready = detector is not None and detector.is_trained,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """Classify a single review as FAKE or GENUINE."""
    if not detector or not detector.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready yet.")

    t0     = time.perf_counter()
    result = detector.predict(req.text, req.rating)
    ms     = round((time.perf_counter() - t0) * 1000, 2)

    db.log_prediction(
        text       = req.text,
        rating     = req.rating,
        is_fake    = result["is_fake"],
        confidence = result["confidence"],
        latency_ms = ms,
        signals    = result["signals"],
    )

    return PredictResponse(
        label       = result["label"],
        is_fake     = result["is_fake"],
        confidence  = result["confidence"],
        signals     = result["signals"],
        latency_ms  = ms,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(req: BatchPredictRequest):
    """Classify up to 50 reviews in a single request."""
    if not detector or not detector.is_trained:
        raise HTTPException(status_code=503, detail="Model not ready yet.")
    if len(req.reviews) > 50:
        raise HTTPException(status_code=400, detail="Max 50 reviews per batch.")

    t0      = time.perf_counter()
    results = [detector.predict(r.text, r.rating) for r in req.reviews]
    ms      = round((time.perf_counter() - t0) * 1000, 2)

    for r_req, r_res in zip(req.reviews, results):
        db.log_prediction(
            text       = r_req.text,
            rating     = r_req.rating,
            is_fake    = r_res["is_fake"],
            confidence = r_res["confidence"],
            latency_ms = ms / len(results),
            signals    = r_res["signals"],
        )

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
    return ModelInfoResponse(**detector.get_info())


@app.get("/analytics", tags=["Analytics"])
def analytics():
    """Returns usage analytics for the dashboard."""
    return db.get_analytics()
