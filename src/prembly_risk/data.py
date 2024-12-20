from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate(config: dict[str, Any]) -> pd.DataFrame:
    """Generate privacy-safe verification events with concept drift and latent risk."""
    rng = np.random.default_rng(config["project"]["seed"])
    n = config["data"]["rows"]
    start = pd.Timestamp(config["data"]["start"])
    day = rng.integers(0, config["data"]["days"], n)
    frame = pd.DataFrame(
        {
            "event_id": [f"evt_{value:08d}" for value in range(n)],
            "event_time": start
            + pd.to_timedelta(day, unit="D")
            + pd.to_timedelta(rng.integers(0, 86400, n), unit="s"),
            "document_type": rng.choice(
                ["passport", "national_id", "drivers_license", "residence_permit"],
                n,
                p=[0.22, 0.48, 0.22, 0.08],
            ),
            "country_risk_tier": rng.choice(["low", "medium", "high"], n, p=[0.42, 0.43, 0.15]),
            "channel": rng.choice(["web", "android", "ios", "api"], n, p=[0.28, 0.34, 0.22, 0.16]),
        }
    )
    frame["face_match_score"] = np.clip(rng.beta(9, 2, n), 0, 1)
    frame["liveness_score"] = np.clip(rng.beta(10, 1.8, n), 0, 1)
    frame["image_quality_score"] = np.clip(rng.beta(7, 2.5, n), 0, 1)
    frame["device_integrity_score"] = np.clip(rng.beta(8, 2, n), 0, 1)
    frame["ip_risk_score"] = np.clip(rng.beta(2, 7, n), 0, 1)
    frame["requests_24h"] = np.maximum(1, rng.negative_binomial(3, 0.55, n) + 1)
    frame["retry_count"] = np.minimum(rng.poisson(0.45, n), 6)
    frame["document_age_days"] = rng.integers(10, 3650, n)
    frame["latency_ms"] = np.maximum(100, rng.lognormal(7.2, 0.42, n))
    high = (frame["country_risk_tier"] == "high").astype(float)
    post_drift = (frame["event_time"] >= pd.Timestamp("2024-03-01")).astype(float)
    logit = (
        2.2
        - 3.2 * frame["face_match_score"]
        - 2.1 * frame["liveness_score"]
        - 1.4 * frame["image_quality_score"]
        - 0.8 * frame["device_integrity_score"]
        + 4.0 * frame["ip_risk_score"]
        + 0.12 * frame["requests_24h"]
        + 0.55 * frame["retry_count"]
        + 0.7 * high
        + 0.35 * post_drift
        + 0.25 * (frame["document_type"] == "residence_permit")
    )
    probability = 1 / (1 + np.exp(-logit))
    frame["verification_failed"] = rng.binomial(1, probability)
    frame["manual_review_minutes"] = np.where(
        frame["verification_failed"].eq(1), rng.gamma(3, 2.5, n), rng.gamma(2, 1.5, n)
    )
    frame = frame.sort_values("event_time").reset_index(drop=True)
    path = Path(config["paths"]["data"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["event_time"])
    required = {
        "event_id",
        "event_time",
        "verification_failed",
        "face_match_score",
        "liveness_score",
    }
    if missing := required - set(frame):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame["event_id"].duplicated().any():
        raise ValueError("Dataset is empty or event IDs are duplicated")
    if not frame["verification_failed"].isin([0, 1]).all():
        raise ValueError("Target must be binary")
    return frame.sort_values("event_time").reset_index(drop=True)


def checksum(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
