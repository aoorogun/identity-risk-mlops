from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from .config import load_config
from .modeling import actions, load_artifacts


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_type: str
    country_risk_tier: str
    channel: str
    face_match_score: float
    liveness_score: float
    image_quality_score: float
    device_integrity_score: float
    ip_risk_score: float
    requests_24h: float
    retry_count: float
    document_age_days: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(os.getenv("PR_CONFIG"))
    root = os.getenv("PR_MODEL_DIR", config["paths"]["artifacts"])
    try:
        app.state.model, app.state.anomaly, app.state.metadata = load_artifacts(root)
        app.state.ready = True
    except FileNotFoundError:
        app.state.ready = False
    yield


app = FastAPI(title="Identity Verification Risk API", version="1.0.0", lifespan=lifespan)


@app.get("/health/ready")
def ready(request: Request) -> dict[str, Any]:
    if not request.app.state.ready:
        raise HTTPException(503, "Model unavailable")
    return {"status": "ready", "version": request.app.state.metadata["version"]}


@app.post("/v1/assess")
def assess(event: Event, request: Request) -> dict[str, Any]:
    if not request.app.state.ready:
        raise HTTPException(503, "Model unavailable")
    frame = pd.DataFrame([event.model_dump()])
    meta = request.app.state.metadata
    probability = float(request.app.state.model.predict_proba(frame[meta["features"]])[0, 1])
    thresholds = meta["policy_thresholds"]
    action = actions([probability], thresholds["approve_below"], thresholds["reject_at_or_above"])[
        0
    ]
    anomaly = bool(request.app.state.anomaly.predict(frame[meta["numeric_features"]])[0] == -1)
    return {
        "failure_probability": probability,
        "recommended_action": action,
        "traffic_anomaly": anomaly,
        "model_version": meta["version"],
    }
