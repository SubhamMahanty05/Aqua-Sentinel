from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import torch
import torch.nn as nn
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sensors.parquet"
METRICS = ROOT / "artifacts" / "metrics" / "lstm_metrics.json"
SCALER = ROOT / "artifacts" / "models" / "sensor_scaler.joblib"
MODEL = ROOT / "artifacts" / "models" / "lstm_autoencoder.pt"
NET3 = ROOT / "data" / "networks" / "Net3.inp"
FEATURES = ROOT / "artifacts" / "models" / "sensor_features.json"

st.set_page_config(page_title="AquaSentinel AI", page_icon="💧", layout="wide")
st.title("💧 AquaSentinel AI — Live Simulation")
st.caption("Net3 hydraulic digital-twin demo with interactive leak simulation")
st.warning("Simulated research prototype. Pressure changes are not proof of real-world leaks.")

# Existing dataset overview
st.header("1. Existing sensor data")
if DATA.exists():
    df = pd.read_parquet(DATA)
    a, b, c = st.columns(3)
    a.metric("Sensor records", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    c.metric("Network", "Net3 — simulated")
    with st.expander("Inspect dataset"):
        st.write("Columns:", df.columns.tolist())
        st.dataframe(df.head(100), width="stretch")

    value_col = next((x for x in ["value", "reading", "measurement", "pressure", "flow"] if x in df.columns), None)
    time_col = next((x for x in ["time_s", "timestamp", "time", "step"] if x in df.columns), None)
    sensor_col = next((x for x in ["sensor_id", "sensor", "node_id"] if x in df.columns), None)
    if value_col:
        p = df.copy()
        if time_col is None:
            p["_record"] = np.arange(len(p))
            time_col = "_record"
        if sensor_col:
            choices = sorted(p[sensor_col].dropna().astype(str).unique())
            selected = st.selectbox("Dataset sensor", ["All sensors"] + choices, key="dataset_sensor")
            if selected != "All sensors":
                p = p[p[sensor_col].astype(str) == selected]
            elif len(choices) > 10:
                p = p[p[sensor_col].astype(str).isin(choices[:10])]
        p[value_col] = pd.to_numeric(p[value_col], errors="coerce")
        p = p.dropna(subset=[value_col]).sort_values(time_col)
        color = sensor_col if sensor_col and p[sensor_col].nunique() <= 10 and ("selected" not in locals() or selected == "All sensors") else None
        st.plotly_chart(px.line(p, x=time_col, y=value_col, color=color, title="Recorded sensor measurements"), width="stretch")
else:
    st.info("Dataset parquet not found; hydraulic simulation can still run if Net3.inp exists.")


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        context = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.decoder(context)
        return self.output(y)

    

def score_lstm_anomalies(pressure_df, model_path, scaler_path, features_path):
    """Experimental LSTM reconstruction-error scoring for simulated pressures."""
    required = [model_path, scaler_path, features_path]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(
            "LSTM checkpoint, scaler, or sensor feature file is missing."
        )

    features = [str(f) for f in json.loads(
        features_path.read_text(encoding="utf-8")
    )]
    scaler = joblib.load(scaler_path)

    # Align columns with the exact training feature order.
    pressure = pressure_df.copy()
    pressure.columns = pressure.columns.astype(str)

    missing = [f for f in features if f not in pressure.columns]

    if missing:
        raise ValueError(
            f"WNTR simulation is missing trained features: {missing[:10]}"
        )

    aligned = pressure.loc[:, features].astype(float)

    if len(aligned) < 12:
        raise ValueError(
            f"Need at least 12 simulation time steps; received {len(aligned)}."
        )

    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Pressure inputs contain NaN or infinite values.")

    if len(features) != scaler.n_features_in_:
        raise ValueError(
            f"Feature mismatch: JSON has {len(features)} features, "
            f"but scaler expects {scaler.n_features_in_}."
        )

    # Scale using the saved training scaler.
    scaled = scaler.transform(aligned)

    # Show diagnostics without changing model inputs or the threshold.
    st.write("### LSTM input diagnostics")
    st.write(
        "Raw pressure range:",
        f"{aligned.to_numpy().min():.4f} to "
        f"{aligned.to_numpy().max():.4f}"
    )
    st.write(
        "Scaled input range:",
        f"{scaled.min():.4f} to {scaled.max():.4f}"
    )
    st.write("Sensor features:", len(features))
    st.write("Simulation time steps:", len(aligned))

    # Build overlapping windows matching training window size.
    window_size = 12
    windows = np.stack([
        scaled[i:i + window_size]
        for i in range(len(scaled) - window_size + 1)
    ]).astype(np.float32)

    model = LSTMAutoencoder(n_features=len(features), hidden=32)
    state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(windows)
        reconstructed = model(inputs)
        errors = (
            (reconstructed - inputs)
            .pow(2)
            .mean(dim=(1, 2))
            .cpu()
            .numpy()
        )

    # WNTR result indices are normally in seconds.
    if isinstance(pressure_df.index, pd.TimedeltaIndex):
        time_hours = pressure_df.index.total_seconds() / 3600
    else:
        time_hours = np.asarray(pressure_df.index, dtype=float) / 3600

    # Each score belongs to the final timestep of its window.
    end_hours = np.asarray(time_hours[window_size - 1:])

    if len(end_hours) != len(errors):
        raise ValueError("Could not align LSTM scores with simulation times.")

    return pd.DataFrame({
        "Window ending (hours)": end_hours,
        "Reconstruction error": errors,
    })



# Live hydraulic simulation
st.header("2. Run a simulated leak")
if not NET3.exists():
    st.error(f"Net3 file not found: {NET3}")
else:
    try:
        import wntr
        probe = wntr.network.WaterNetworkModel(str(NET3))
        junctions = list(probe.junction_name_list)
        if not junctions:
            st.error("No junctions found in Net3.inp.")
        else:
            x, y, z = st.columns(3)
            with x:
                leak_node = st.selectbox("Leak junction", junctions, index=min(10, len(junctions)-1))
            with y:
                leak_coeff = st.select_slider("Leak severity", options=[0.001, 0.005, 0.01, 0.02, 0.05], value=0.01, format_func=lambda v: f"{v:g}")
            with z:
                duration_hr = st.selectbox("Duration (hours)", [1, 2, 3, 6], index=1)

            if st.button("▶ Run normal vs leak simulation", type="primary", width="stretch"):
                with st.spinner("Running normal baseline and leak scenario with WNTR…"):
                    wn_normal = wntr.network.WaterNetworkModel(str(NET3))
                    wn_normal.options.time.duration = int(duration_hr * 3600)
                    wn_normal.options.time.hydraulic_timestep = 300
                    wn_normal.options.time.report_timestep = 300
                    normal = wntr.sim.WNTRSimulator(wn_normal).run_sim()

                    wn_leak = wntr.network.WaterNetworkModel(str(NET3))
                    wn_leak.options.time.duration = int(duration_hr * 3600)
                    wn_leak.options.time.hydraulic_timestep = 300
                    wn_leak.options.time.report_timestep = 300
                    leak_junction = wn_leak.get_node(leak_node)
                    leak_junction.add_leak(
                        wn_leak,
                        area=0.001,
                        discharge_coeff=0.75,
                        start_time=0,
                        end_time=int(duration_hr * 3600),
                    )
                    leak = wntr.sim.WNTRSimulator(wn_leak).run_sim()

                npres = normal.node["pressure"]
                lpres = leak.node["pressure"]
                
                try:
                    lstm_scores = score_lstm_anomalies(
                        lpres if False else leak.node["pressure"],
                        MODEL,
                        SCALER,
                        FEATURES,
                    )
                    st.subheader("Experimental LSTM anomaly scoring")
                    st.line_chart(
                        lstm_scores.set_index("Window ending (hours)")["Reconstruction error"]
                    )
                    st.caption(
                        "Experimental reconstruction error; this is not a validated leak "
                        "classification or a calibrated anomaly probability."
                    )
                except (ValueError, FileNotFoundError, RuntimeError) as e:
                    st.warning(f"LSTM scoring unavailable for this simulation: {e}")

                if leak_node not in npres.columns or leak_node not in lpres.columns:
                    st.error("Selected junction was not found in simulation pressure results.")
                else:
                    delta = npres[leak_node] - lpres[leak_node]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Leak junction", str(leak_node))
                    m2.metric("Max pressure reduction", f"{float(delta.max()):.3f}")
                    m3.metric("Mean pressure reduction", f"{float(delta.mean()):.3f}")

                    pressure_df = pd.DataFrame({
                        "Time (hours)": npres.index / 3600,
                        "Normal baseline": npres[leak_node].to_numpy(),
                        "Leak scenario": lpres[leak_node].to_numpy()
                    })
                    fig = px.line(pressure_df, x="Time (hours)", y=["Normal baseline", "Leak scenario"],
                                  title=f"Pressure at junction {leak_node}: normal vs leak")
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, width="stretch")

                    common = [n for n in npres.columns if n in lpres.columns]
                    comp = pd.DataFrame({
                        "Junction": common,
                        "Normal pressure": npres.iloc[-1][common].to_numpy(),
                        "Leak pressure": lpres.iloc[-1][common].to_numpy()
                    })
                    comp["Pressure reduction"] = comp["Normal pressure"] - comp["Leak pressure"]
                    comp = comp.sort_values("Pressure reduction", ascending=False)
                    st.subheader("Largest pressure reductions at final simulation time")
                    st.dataframe(comp.head(15).round(4), width="stretch")
                    st.plotly_chart(px.bar(comp.head(15).sort_values("Pressure reduction"),
                                           x="Pressure reduction", y="Junction", orientation="h",
                                           title="Top junctions by pressure reduction"), width="stretch")
                    st.success("Hydraulic simulation complete. These are simulated effects, not validated leak predictions.")
            else:
                st.info("Choose a junction and leak severity, then click Run normal vs leak simulation.")
    except ImportError:
        st.error("WNTR is missing. In the project terminal run: python -m pip install wntr")
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.caption("If needed, try a shorter duration or lower leak severity.")

# Existing LSTM status
st.header("3. LSTM model status")
st.write("Checkpoint:", "Available" if MODEL.exists() else "Not found")
if METRICS.exists():
    try:
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        if metrics.get("scenario_results"):
            st.dataframe(pd.DataFrame(metrics["scenario_results"]), width="stretch")
        else:
            a, b, c = st.columns(3)
            a.metric("Training epochs", metrics.get("epochs", "—"))
            b.metric("Scored windows", metrics.get("windows", "—"))
            c.metric("Flagged windows", metrics.get("flagged_windows", "—"))
    except Exception as e:
        st.warning(f"Could not read metrics: {e}")
st.caption("The LSTM metrics shown here are separate from this new hydraulic simulation unless explicitly evaluated on its output.")
