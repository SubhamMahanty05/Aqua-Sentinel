# AquaSentinel AI — Deep Learning demo

Copy `notebooks/AquaSentinel_DeepLearning.ipynb` into your project's `notebooks` folder and `app/streamlit_app.py` into your existing `app` folder.

From the project root in the activated `.venv`:
```powershell
python -m pip install torch jupyter ipykernel
python scripts/generate_dataset.py --quick
jupyter notebook notebooks/AquaSentinel_DeepLearning.ipynb
```
Run all notebook cells to train and save the LSTM autoencoder and metrics. Then run:
```powershell
streamlit run app/streamlit_app.py
```

This is a simulated-data demonstration, not validated real-world leak detection.
