from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aquasentinel.paths import resolve_path


def load_processed(cfg: dict) -> dict:
    d = resolve_path(cfg["paths"]["processed_dir"])
    sensors = pd.read_parquet(d / "sensors.parquet")
    leaks = pd.read_parquet(d / "leaks.parquet") if (d / "leaks.parquet").exists() else pd.DataFrame()
    faults = pd.read_parquet(d / "faults.parquet") if (d / "faults.parquet").exists() else pd.DataFrame()
    network = json.loads((d / "network.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "dataset_meta.json").read_text(encoding="utf-8"))
    return {
        "sensors": sensors,
        "leaks": leaks,
        "faults": faults,
        "network": network,
        "meta": meta,
        "dir": d,
    }
