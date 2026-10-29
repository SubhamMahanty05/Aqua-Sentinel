from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


def _pipe_table(network: dict) -> pd.DataFrame:
    import networkx as nx

    from aquasentinel.graph.network_graph import networkx_from_json

    g = networkx_from_json(network)
    btw = nx.betweenness_centrality(g)
    rows = []
    for e in network["edges"]:
        if e.get("link_type") and e["link_type"] != "Pipe":
            # still include; pumps/valves can sit near leaks
            pass
        rows.append(
            {
                "pipe_id": e["id"],
                "start": e["start"],
                "end": e["end"],
                "length": float(e.get("length") or 0.0),
                "diameter": float(e.get("diameter") or 0.0),
                "betweenness": 0.5 * (btw.get(e["start"], 0.0) + btw.get(e["end"], 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_risk_frame(network: dict, leaks: pd.DataFrame, sensors: pd.DataFrame, normal_ids: list[str]) -> pd.DataFrame:
    pipes = _pipe_table(network)
    # Pre-leak / normal hydraulic summary from pressure sensors (shared features).
    normal = sensors[sensors["scenario_id"].isin(normal_ids) & (sensors["kind"] == "pressure")]
    stats = normal.groupby("sensor_id")["value"].agg(["mean", "std"]).reset_index()
    mean_p = float(stats["mean"].mean()) if len(stats) else 0.0
    std_p = float(stats["std"].mean()) if len(stats) else 0.0
    leak_nodes = set(leaks["junction_id"].astype(str)) if len(leaks) else set()
    # One row per pipe; label 1 if either endpoint was a simulated leak junction.
    pipes["mean_pressure_network"] = mean_p
    pipes["std_pressure_network"] = std_p
    pipes["y"] = pipes.apply(lambda r: int(str(r["start"]) in leak_nodes or str(r["end"]) in leak_nodes), axis=1)
    return pipes


def train_risk_model(frame: pd.DataFrame, seed: int = 42) -> dict:
    feature_cols = ["length", "diameter", "betweenness", "mean_pressure_network", "std_pressure_network"]
    X = frame[feature_cols].fillna(0.0).to_numpy()
    y = frame["y"].to_numpy()
    dummy = DummyClassifier(strategy="prior", random_state=seed)
    dummy.fit(X, y)
    clf = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=80)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    dproba = dummy.predict_proba(X)[:, 1]
    metrics = {
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else None,
        "brier": float(brier_score_loss(y, proba)),
        "dummy_roc_auc": float(roc_auc_score(y, dproba)) if len(np.unique(y)) > 1 else None,
        "dummy_pr_auc": float(average_precision_score(y, dproba)) if len(np.unique(y)) > 1 else None,
        "feature_importances": {c: float(i) for c, i in zip(feature_cols, clf.feature_importances_)},
        "note": "Labels mark pipes incident to simulated leak junctions on this dataset, not real remaining useful life.",
    }
    return {"model": clf, "dummy": dummy, "features": feature_cols, "metrics": metrics, "proba": proba, "frame": frame}
