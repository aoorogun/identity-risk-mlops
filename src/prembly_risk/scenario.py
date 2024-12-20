from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_data
from .modeling import actions, load_artifacts


def run_scenario(config: dict[str, Any], scenario_path: str | Path) -> dict[str, Any]:
    scenario = json.loads(Path(scenario_path).read_text())
    frame = load_data(config["paths"]["data"]).tail(5000)
    model, _, metadata = load_artifacts(config["paths"]["artifacts"])
    baseline = model.predict_proba(frame[metadata["features"]])[:, 1]
    shocked = frame.copy()
    for feature, change in scenario["feature_shocks"].items():
        if feature not in config["features"]["numeric"]:
            raise ValueError(f"Feature cannot be shocked: {feature}")
        if change["operation"] == "add":
            shocked[feature] += change["value"]
        elif change["operation"] == "multiply":
            shocked[feature] *= change["value"]
        else:
            raise ValueError("Only add and multiply operations are allowed")
        if feature.endswith("score"):
            shocked[feature] = shocked[feature].clip(0, 1)
    scenario_probability = model.predict_proba(shocked[metadata["features"]])[:, 1]
    base_policy = metadata["policy_thresholds"]
    policy = scenario["policy"]
    base_actions = actions(
        baseline, base_policy["approve_below"], base_policy["reject_at_or_above"]
    )
    scenario_actions = actions(
        scenario_probability, policy["approve_below"], policy["reject_at_or_above"]
    )
    multiplier = scenario["traffic_multiplier"]
    result = {
        "scenario_id": scenario["scenario_id"],
        "sample_rows": len(frame),
        "traffic_multiplier": multiplier,
        "baseline_failure_probability": float(np.mean(baseline)),
        "scenario_failure_probability": float(np.mean(scenario_probability)),
        "baseline_review_volume": int(np.sum(base_actions == "review")),
        "scenario_review_volume_scaled": int(
            round(np.sum(scenario_actions == "review") * multiplier)
        ),
        "baseline_reject_rate": float(np.mean(base_actions == "reject")),
        "scenario_reject_rate": float(np.mean(scenario_actions == "reject")),
    }
    path = Path(config["paths"]["reports"]) / "scenario_result.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
