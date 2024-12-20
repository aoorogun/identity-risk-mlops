from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else ROOT / "configs/base.yaml"
    if not source.is_absolute():
        source = ROOT / source
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key, value in config["paths"].items():
        candidate = Path(value)
        config["paths"][key] = str(candidate if candidate.is_absolute() else ROOT / candidate)
    return config
