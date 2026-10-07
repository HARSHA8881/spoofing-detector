"""Isolation Forest: unsupervised, flags orders that are easy to isolate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features import FEATURES

LOG_FEATURES = ["size_ratio", "lifetime_s", "opp_exec_rel", "cancel_to_trade", "n_large_cancels"]
FIT_ROWS = 200_000


class IsolationForestDetector:
    name = "iforest"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = IsolationForest(n_estimators=200, max_samples=1024, random_state=seed, n_jobs=-1)

    def _prepare(self, orders: pd.DataFrame) -> pd.DataFrame:
        X = orders[FEATURES].astype(float).copy()
        X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES].clip(lower=0))
        return X

    def fit(self, orders: pd.DataFrame) -> "IsolationForestDetector":
        X = self._prepare(orders)
        self.median = X.median()
        self.iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, np.nan).fillna(X.std()).replace(0, 1)
        X = X.fillna(self.median)
        self.model.fit(X.sample(min(FIT_ROWS, len(X)), random_state=self.seed))
        return self

    def score(self, orders: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(self._prepare(orders).fillna(self.median))

    def explain(self, orders: pd.DataFrame, top: int = 3) -> list[str]:
        """Which features sit furthest from the typical order (robust z-scores)."""
        z = (self._prepare(orders).fillna(self.median) - self.median) / self.iqr
        out = []
        for (_, o), (_, row) in zip(orders.iterrows(), z.iterrows()):
            far = row.abs().sort_values(ascending=False).index[:top]
            out.append("unusual: " + "; ".join(
                f"{f} = {o[f]:.3g} ({row[f]:+.1f} IQRs from typical)" for f in far))
        return out
