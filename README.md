# AquaSentinel AI 💧

### AI-Powered Water Distribution Network Monitoring and Leak Anomaly Detection

AquaSentinel AI combines WNTR hydraulic simulation with an LSTM Autoencoder to explore pressure-based anomaly detection in water distribution networks.

## Features
- WNTR Net3 hydraulic simulation.
- Normal-versus-leak pressure comparison.
- Configurable leak junction, severity, and simulation duration.
- Interactive Streamlit dashboard.
- Experimental LSTM reconstruction-error scoring.
- Saved model, scaler, sensor feature list, and metrics.

## Technology Stack
Python · WNTR · PyTorch · Streamlit · Pandas · NumPy · Scikit-learn · Plotly

## Project Structure
```text
AquaSentinel-AI/
├── app/streamlit_app.py
├── artifacts/
│   ├── metrics/
│   └── models/
├── configs/
├── data/
│   ├── networks/
│   └── processed/
├── notebooks/AquaSentinel_DeepLearning.ipynb
├── src/
├── requirements.txt
└── README.md
```

## Setup and Run

```powershell
git clone https://github.com/SubhamMahanty05/Aqua-Sentinel.git
cd Aqua-Sentinel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Preliminary WNTR LSTM Experiment

| Measurement | Result |
|---|---:|
| Normal mean reconstruction error | 0.457939 |
| Leak mean reconstruction error | 0.613537 |
| Normal-derived 95th-percentile threshold | 0.575751 |
| Normal windows flagged | 1 / 14 (7.14%) |
| Leak windows flagged | 9 / 14 (64.29%) |

These are preliminary results from one simulated normal run and one simulated leak run. They are not independent benchmark results and do not establish real-world leak detection performance.

## Limitations
- Evaluation currently covers very few simulated runs.
- The threshold was calibrated using the normal run being evaluated.
- Leak detection needs testing across multiple leak locations, severities, durations, and network conditions.
- Sensor-fault detection has not yet been validated independently.
- Reconstruction error is an anomaly score, not a calibrated leak probability.

## Future Work
- Validate multiple independent normal, leak, and sensor-fault scenarios.
- Test different leak locations, sizes, and start times.
- Add real-time sensor data ingestion.
- Improve model calibration and anomaly classification.
- Deploy the dashboard.
