from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(self, n_estimators: int = 200, contamination: float = 0.08, seed: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )

    def fit_normal(self, X: np.ndarray) -> None:
        flat = X.reshape(X.shape[0], -1)
        self.model.fit(flat)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        flat = X.reshape(X.shape[0], -1)
        # decision_function: higher = more normal; invert
        return -self.model.decision_function(flat).astype(np.float64)
