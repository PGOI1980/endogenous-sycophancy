from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def save_json(payload: Any, filename: str) -> Path:
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path

def load_json(filename: str) -> Any:
    path = RESULTS_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
