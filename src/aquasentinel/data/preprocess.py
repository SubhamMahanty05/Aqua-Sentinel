from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def pivot_sensors(sensors: pd.DataFrame) -> pd.DataFrame:
    df = sensors.copy()
    df["feature"] = df["kind"] + ":" + df["sensor_id"].astype(str)
    wide = df.pivot_table(
        index=["scenario_id", "scenario_type", "time_s"],
        columns="feature",
        values="value",
        aggfunc="mean",
    )
    wide = wide.sort_index()
    return wide


def split_scenarios(sensors: pd.DataFrame, seed: int = 42) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    types = sensors.groupby("scenario_id")["scenario_type"].first()
    normal = types[types == "normal"].index.tolist()
    leak = types[types == "leak"].index.tolist()
    fault = types[types == "sensor_fault"].index.tolist()
    rng.shuffle(leak)
    n_val = max(1, len(leak) // 5) if leak else 0
    n_test = max(1, len(leak) // 5) if leak else 0
    test = leak[:n_test]
    val = leak[n_test : n_test + n_val]
    train_leak = leak[n_test + n_val :]
    train = normal + train_leak
    return {"train": train, "val": val, "test": test, "fault": fault, "normal": normal, "leak": leak}


def fit_scaler(wide: pd.DataFrame, train_ids: list[str], normal_ids: list[str]) -> StandardScaler:
    idx = wide.index.get_level_values("scenario_id")
    fit_ids = [s for s in train_ids if s in set(normal_ids)]
    if not fit_ids:
        fit_ids = train_ids
    X = wide.loc[idx.isin(fit_ids)].fillna(0.0).to_numpy()
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def transform_wide(wide: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    filled = wide.fillna(0.0)
    arr = scaler.transform(filled.to_numpy())
    return pd.DataFrame(arr, index=filled.index, columns=filled.columns)


def leak_time_mask(index: pd.Index, leaks: pd.DataFrame) -> np.ndarray:
    leak_map = {r.scenario_id: (int(r.start_s), int(r.end_s)) for r in leaks.itertuples()}
    y = np.zeros(len(index), dtype=np.int32)
    for i, (sid, _stype, t) in enumerate(index):
        if sid in leak_map:
            start, end = leak_map[sid]
            if start <= int(t) <= end:
                y[i] = 1
    return y


def build_windows(
    wide_scaled: pd.DataFrame,
    window: int,
    leaks: pd.DataFrame,
    scenario_ids: list[str] | None = None,
) -> dict:
    idx = wide_scaled.index
    sids = idx.get_level_values("scenario_id")
    if scenario_ids is not None:
        keep = sids.isin(scenario_ids)
        data = wide_scaled.loc[keep]
    else:
        data = wide_scaled
    Xs = []
    ys = []
    meta = []
    y_time = leak_time_mask(data.index, leaks)
    # Group by scenario
    for sid, grp in data.groupby(level="scenario_id", sort=False):
        arr = grp.to_numpy()
        times = grp.index.get_level_values("time_s").to_numpy()
        stype = grp.index.get_level_values("scenario_type")[0]
        labels = y_time[data.index.get_level_values("scenario_id") == sid]
        if len(arr) < window:
            continue
        for t in range(window - 1, len(arr)):
            sl = arr[t - window + 1 : t + 1]
            Xs.append(sl)
            ys.append(int(labels[t]))
            meta.append({"scenario_id": sid, "scenario_type": stype, "time_s": int(times[t])})
    X = np.stack(Xs) if Xs else np.zeros((0, window, data.shape[1]))
    y = np.asarray(ys, dtype=np.int32)
    return {
        "X": X.astype(np.float32),
        "y": y,
        "meta": meta,
        "feature_names": list(data.columns),
    }
