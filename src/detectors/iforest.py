"""Isolation Forest: unsupervised, flags orders that are easy to isolate.

Fitted on every order it mostly measures "rare", and most rare orders are not
spoofs. So it is fitted only on episode candidates (large unfilled cancels) and
only on the features that describe how such an order behaved. Orders that are
not candidates get the lowest score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

USED = ["level_share_add", "depth_share_add", "lifetime_s", "dist_ticks", "dist_change_ticks",
        "opp_exec_rel", "imb_jump_add", "ep_cancel_together", "t_since_trade_side"]
LOG_FEATURES = ["lifetime_s", "opp_exec_rel", "ep_cancel_together", "t_since_trade_side"]


class IsolationForestDetector:
    name = "iforest"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = IsolationForest(n_estimators=200, max_samples=1024, random_state=seed, n_jobs=-1)

    def _prepare(self, orders: pd.DataFrame) -> pd.DataFrame:
        X = orders[USED].astype(float).copy()
        X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES].clip(lower=0))
        return X

    def fit(self, orders: pd.DataFrame) -> "IsolationForestDetector":
        X = self._prepare(orders[orders.candidate])
        self.median = X.median()
        self.iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, np.nan).fillna(X.std()).replace(0, 1)
        self.model.fit(X.fillna(self.median))
        return self

    def score(self, orders: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(orders))
        cand = orders.candidate.to_numpy()
        if cand.any():
            scores[cand] = -self.model.score_samples(self._prepare(orders[cand]).fillna(self.median))
        return scores

    def explain(self, orders: pd.DataFrame, top: int = 3) -> list[str]:
        """Which features sit furthest from the typical large cancel (robust z-scores)."""
        z = (self._prepare(orders).fillna(self.median) - self.median) / self.iqr
        out = []
        for (_, o), (_, row) in zip(orders.iterrows(), z.iterrows()):
            far = row.abs().sort_values(ascending=False).index[:top]
            out.append("unusual for a large cancel: " + "; ".join(
                f"{f} = {o[f]:.3g} ({row[f]:+.1f} IQRs from typical)" for f in far))
        return out
