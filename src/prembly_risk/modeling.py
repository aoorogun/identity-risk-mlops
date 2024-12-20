from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import checksum, load_data


def make_model(name: str, categorical: list[str], numeric: list[str], seed: int) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=3000, C=0.6)
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "histogram_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=240,
            learning_rate=0.06,
            max_leaf_nodes=24,
            l2_regularization=0.3,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown candidate: {name}")
    return Pipeline([("preprocess", prep), ("model", estimator)])


def classification_metrics(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "max_f1": float(np.max(2 * precision * recall / np.maximum(precision + recall, 1e-9))),
    }


def actions(probability: np.ndarray, approve_below: float, reject_at_or_above: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    return np.where(
        probability < approve_below,
        "approve",
        np.where(probability >= reject_at_or_above, "reject", "review"),
    )


def policy_cost(
    y: np.ndarray, p: np.ndarray, low: float, high: float, costs: dict[str, float]
) -> float:
    action = actions(p, low, high)
    false_approve = np.sum((action == "approve") & (y == 1))
    false_reject = np.sum((action == "reject") & (y == 0))
    review = np.sum(action == "review")
    return float(
        false_approve * costs["false_approval"]
        + false_reject * costs["false_rejection"]
        + review * costs["manual_review"]
    )


def train(config: dict[str, Any]) -> dict[str, Any]:
    frame = load_data(config["paths"]["data"])
    categorical, numeric = config["features"]["categorical"], config["features"]["numeric"]
    features, target = categorical + numeric, config["features"]["target"]
    test_start = pd.Timestamp(config["data"]["test_start"])
    validation_start = test_start - pd.Timedelta(days=config["training"]["validation_days"])
    train_frame = frame[frame.event_time < validation_start]
    validation = frame[(frame.event_time >= validation_start) & (frame.event_time < test_start)]
    test = frame[frame.event_time >= test_start]
    candidates, fitted = {}, {}
    for name in config["training"]["candidates"]:
        model = make_model(name, categorical, numeric, config["project"]["seed"])
        model.fit(train_frame[features], train_frame[target])
        p = model.predict_proba(validation[features])[:, 1]
        candidates[name] = classification_metrics(validation[target], p)
        fitted[name] = model
    selected = max(candidates, key=lambda name: candidates[name]["average_precision"])
    low, high = config["training"]["review_thresholds"]
    combined = pd.concat([train_frame, validation])
    model = make_model(selected, categorical, numeric, config["project"]["seed"])
    model.fit(combined[features], combined[target])
    test_p = model.predict_proba(test[features])[:, 1]
    test_actions = actions(test_p, low, high)
    anomaly = IsolationForest(
        n_estimators=200, contamination=0.02, random_state=config["project"]["seed"]
    )
    anomaly.fit(combined[numeric])
    output = test[["event_time", target, "country_risk_tier"]].copy()
    output["failure_probability"] = test_p
    output["action"] = test_actions
    output["anomaly"] = anomaly.predict(test[numeric]) == -1
    group_metrics = {}
    for group, values in output.groupby("country_risk_tier"):
        group_metrics[group] = classification_metrics(values[target], values["failure_probability"])
        group_metrics[group]["review_rate"] = float((values["action"] == "review").mean())
    metadata = {
        "version": config["project"]["version"],
        "selected_model": selected,
        "features": features,
        "numeric_features": numeric,
        "candidates": candidates,
        "test_metrics": classification_metrics(test[target], test_p),
        "policy_thresholds": {"approve_below": low, "reject_at_or_above": high},
        "policy_cost": policy_cost(
            test[target].to_numpy(), test_p, low, high, config["training"]["costs"]
        ),
        "action_rates": {
            name: float(np.mean(test_actions == name)) for name in ["approve", "review", "reject"]
        },
        "group_metrics": group_metrics,
        "train_end": str(combined.event_time.max()),
        "test_start": config["data"]["test_start"],
        "test_rows": len(test),
        "data_sha256": checksum(config["paths"]["data"]),
    }
    artifact_dir, report_dir = Path(config["paths"]["artifacts"]), Path(config["paths"]["reports"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "risk_model.joblib")
    joblib.dump(anomaly, artifact_dir / "anomaly_model.joblib")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output.to_csv(report_dir / "test_predictions.csv", index=False)
    return metadata


def load_artifacts(path: str | Path):
    root = Path(path)
    return (
        joblib.load(root / "risk_model.joblib"),
        joblib.load(root / "anomaly_model.joblib"),
        json.loads((root / "metadata.json").read_text()),
    )
