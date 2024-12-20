from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_data


def psi(reference: pd.Series, current: pd.Series) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)
    b = np.clip(np.histogram(current, edges)[0] / len(current), 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def monitor(config: dict[str, Any]) -> dict[str, Any]:
    frame = load_data(config["paths"]["data"])
    midpoint = frame.event_time.quantile(0.75)
    reference, current = frame[frame.event_time < midpoint], frame[frame.event_time >= midpoint]
    scores = {c: psi(reference[c], current[c]) for c in config["features"]["numeric"]}
    maximum = max(scores.values())
    mcfg = config["monitoring"]
    status = (
        "critical"
        if maximum >= mcfg["psi_critical"]
        else "warning"
        if maximum >= mcfg["psi_warning"]
        else "healthy"
    )
    result = {"status": status, "maximum_psi": maximum, "feature_psi": scores}
    (Path(config["paths"]["reports"]) / "drift_report.json").write_text(
        json.dumps(result, indent=2)
    )
    return result
