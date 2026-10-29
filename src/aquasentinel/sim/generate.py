from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import wntr
from tqdm import tqdm

from aquasentinel.paths import project_root, resolve_path


def load_or_copy_network(cfg: dict) -> tuple[wntr.network.WaterNetworkModel, Path]:
    dest = resolve_path(cfg["network_inp"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    name = cfg.get("network_name", "Net3")
    wn = wntr.network.WaterNetworkModel(name)
    if not dest.exists():
        # Reconstruct a portable INP from the loaded model.
        wntr.network.write_inpfile(wn, str(dest))
    else:
        wn = wntr.network.WaterNetworkModel(str(dest))
    return wn, dest


def _configure_time(wn: wntr.network.WaterNetworkModel, cfg: dict) -> None:
    duration = int(cfg["duration_hours"] * 3600)
    hyd = int(cfg["hydraulic_timestep_min"] * 60)
    rpt = int(cfg.get("report_timestep_min", cfg["hydraulic_timestep_min"]) * 60)
    wn.options.time.duration = duration
    wn.options.time.hydraulic_timestep = hyd
    wn.options.time.quality_timestep = hyd
    wn.options.time.report_timestep = rpt
    wn.options.time.pattern_timestep = 3600
    wn.options.time.rule_timestep = hyd


def _junctions(wn: wntr.network.WaterNetworkModel) -> list[str]:
    return list(wn.junction_name_list)


def _select_sensors(wn: wntr.network.WaterNetworkModel, cfg: dict, rng: np.random.Generator) -> dict:
    junctions = _junctions(wn)
    pipes = list(wn.pipe_name_list)
    n_p = min(int(cfg["n_pressure_sensors"]), len(junctions))
    n_f = min(int(cfg["n_flow_sensors"]), len(pipes))
    # Prefer spread: sort by x-coordinate then stride.
    coords = []
    for j in junctions:
        node = wn.get_node(j)
        xy = getattr(node, "coordinates", (0.0, 0.0)) or (0.0, 0.0)
        coords.append((j, float(xy[0]), float(xy[1])))
    coords.sort(key=lambda t: (t[1], t[2]))
    step = max(len(coords) // n_p, 1)
    pressure = [coords[i][0] for i in range(0, len(coords), step)][:n_p]
    if len(pressure) < n_p:
        extras = [j for j in junctions if j not in pressure]
        pressure.extend(extras[: n_p - len(pressure)])
    flow = list(pipes[:n_f]) if n_f else []
    if len(flow) < n_f:
        flow = list(rng.choice(pipes, size=n_f, replace=False))
    return {"pressure": pressure, "flow": flow}


def _network_json(wn: wntr.network.WaterNetworkModel, sensors: dict) -> dict:
    nodes = []
    for name in wn.node_name_list:
        node = wn.get_node(name)
        xy = getattr(node, "coordinates", (0.0, 0.0)) or (0.0, 0.0)
        ntype = node.node_type
        nodes.append(
            {
                "id": name,
                "type": ntype,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "elevation": float(getattr(node, "elevation", 0.0) or 0.0),
            }
        )
    edges = []
    for name, link in wn.links():
        edges.append(
            {
                "id": name,
                "start": link.start_node_name,
                "end": link.end_node_name,
                "length": float(getattr(link, "length", 0.0) or 0.0),
                "diameter": float(getattr(link, "diameter", 0.0) or 0.0),
                "link_type": link.link_type,
            }
        )
    return {"nodes": nodes, "edges": edges, "sensors": sensors}


def _results_to_rows(
    results,
    scenario_id: str,
    scenario_type: str,
    sensors: dict,
) -> list[dict]:
    rows = []
    pressure = results.node["pressure"]
    flow = results.link["flowrate"]
    for t in pressure.index:
        for sid in sensors["pressure"]:
            if sid in pressure.columns:
                val = pressure.at[t, sid]
                rows.append(
                    {
                        "time_s": int(t),
                        "scenario_id": scenario_id,
                        "scenario_type": scenario_type,
                        "sensor_id": sid,
                        "kind": "pressure",
                        "value": float(val) if pd.notna(val) else np.nan,
                    }
                )
        for sid in sensors["flow"]:
            if sid in flow.columns:
                val = flow.at[t, sid]
                rows.append(
                    {
                        "time_s": int(t),
                        "scenario_id": scenario_id,
                        "scenario_type": scenario_type,
                        "sensor_id": sid,
                        "kind": "flow",
                        "value": float(val) if pd.notna(val) else np.nan,
                    }
                )
    return rows


def _run_hydraulics(wn: wntr.network.WaterNetworkModel):
    sim = wntr.sim.WNTRSimulator(wn)
    return sim.run_sim()


def _fresh_wn(inp: Path, cfg: dict) -> wntr.network.WaterNetworkModel:
    wn = wntr.network.WaterNetworkModel(str(inp))
    _configure_time(wn, cfg)
    return wn


def _apply_faults(
    df: pd.DataFrame,
    scenario_id: str,
    rng: np.random.Generator,
    cfg: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy()
    sensors = out["sensor_id"].unique()
    pick = rng.choice(sensors)
    kind = rng.choice(["stuck", "spike", "dropout"])
    mask = out["sensor_id"] == pick
    meta = {"scenario_id": scenario_id, "sensor_id": str(pick), "fault_type": kind}
    if kind == "stuck":
        stuck = float(out.loc[mask, "value"].iloc[0])
        out.loc[mask, "value"] = stuck
    elif kind == "spike":
        noise = rng.normal(0, float(cfg.get("fault_noise_std", 0.35)), size=int(mask.sum()))
        scale = out.loc[mask, "value"].abs().mean() + 1e-6
        out.loc[mask, "value"] = out.loc[mask, "value"].to_numpy() + noise * scale
    else:
        times = sorted(out.loc[mask, "time_s"].unique())
        drop = set(times[len(times) // 3 : 2 * len(times) // 3])
        out.loc[mask & out["time_s"].isin(drop), "value"] = np.nan
        meta["dropped_times"] = list(map(int, drop))
    return out, [meta]


def generate_dataset(cfg: dict, quick: bool = False) -> dict:
    rng = np.random.default_rng(int(cfg["seed"]))
    wn0, inp = load_or_copy_network(cfg)
    _configure_time(wn0, cfg)
    sensors = _select_sensors(wn0, cfg, rng)
    processed = resolve_path(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    leaks: list[dict] = []
    faults: list[dict] = []
    junctions = _junctions(wn0)
    n_normal = 1 if quick else int(cfg.get("n_normal_days", 2))
    duration = int(cfg["duration_hours"] * 3600)

    for i in range(n_normal):
        sid = f"normal_{i:02d}"
        wn = _fresh_wn(inp, cfg)
        # Slight demand jitter so normal is not a single identical trace.
        if i > 0:
            for jname in junctions:
                node = wn.get_node(jname)
                base = float(node.demand_timeseries_list[0].base_value)
                node.demand_timeseries_list[0].base_value = max(base * (1.0 + 0.05 * (i % 3)), 0.0)
        results = _run_hydraulics(wn)
        rows.extend(_results_to_rows(results, sid, "normal", sensors))

    leak_areas = [cfg["leak_areas_m2"][0]] if quick else list(cfg["leak_areas_m2"])
    starts = [cfg["leak_start_hours"][0]] if quick else list(cfg["leak_start_hours"])
    n_j = 1 if quick else min(int(cfg["n_leak_junctions"]), len(junctions))
    leak_js = list(rng.choice(junctions, size=n_j, replace=False))

    for jid in tqdm(leak_js, desc="leak scenarios"):
        for area in leak_areas:
            for start_h in starts:
                sid = f"leak_{jid}_a{area}_t{start_h}"
                wn = _fresh_wn(inp, cfg)
                start = int(start_h * 3600)
                node = wn.get_node(jid)
                node.add_leak(wn, area=float(area), start_time=start, end_time=duration)
                results = _run_hydraulics(wn)
                rows.extend(_results_to_rows(results, sid, "leak", sensors))
                leaks.append(
                    {
                        "scenario_id": sid,
                        "junction_id": jid,
                        "area_m2": float(area),
                        "start_s": start,
                        "end_s": duration,
                    }
                )

    n_faults = 1 if quick else int(cfg.get("n_fault_scenarios", 3))
    normal_ids = sorted({r["scenario_id"] for r in rows if r["scenario_type"] == "normal"})
    base_normal = pd.DataFrame([r for r in rows if r["scenario_id"] == normal_ids[0]])
    for k in range(n_faults):
        sid = f"fault_{k:02d}"
        df_f, meta = _apply_faults(base_normal, sid, rng, cfg)
        df_f["scenario_id"] = sid
        df_f["scenario_type"] = "sensor_fault"
        rows.extend(df_f.to_dict("records"))
        faults.extend(meta)

    sensors_df = pd.DataFrame(rows)
    leaks_df = pd.DataFrame(leaks)
    faults_df = pd.DataFrame(faults)
    net = _network_json(wn0, sensors)

    sensors_df.to_parquet(processed / "sensors.parquet", index=False)
    leaks_df.to_parquet(processed / "leaks.parquet", index=False)
    faults_df.to_parquet(processed / "faults.parquet", index=False)
    (processed / "network.json").write_text(json.dumps(net, indent=2), encoding="utf-8")
    meta = {
        "n_rows": int(len(sensors_df)),
        "scenarios": sorted(sensors_df["scenario_id"].unique().tolist()),
        "sensors": sensors,
        "quick": quick,
        "inp": str(inp),
        "wntr_version": getattr(wntr, "__version__", "unknown"),
    }
    (processed / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
