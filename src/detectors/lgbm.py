"""Gradient boosting trained on the injected labels, explained with SHAP values."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import FEATURES


class LightGBMDetector:
    name = "lgbm"

    def __init__(self, seed: int = 0):
        self.model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, verbose=-1)

    def fit(self, orders: pd.DataFrame) -> "LightGBMDetector":
        y = orders.label.to_numpy()
        # positives are ~1 in 4,000: upweight them, but not all the way to balance
        self.model.set_params(scale_pos_weight=np.sqrt((y == 0).sum() / max(y.sum(), 1)))
        self.model.fit(orders[FEATURES].astype(float), y)
        return self

    def score(self, orders: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(orders[FEATURES].astype(float))[:, 1]

    def shap_values(self, orders: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions in log-odds, one column per feature.

        LightGBM computes these natively (pred_contrib); they are the same values
        shap.TreeExplainer returns.
        """
        contrib = self.model.booster_.predict(orders[FEATURES].astype(float), pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=orders.index)

    def explain(self, orders: pd.DataFrame, top: int = 3) -> list[str]:
        shap = self.shap_values(orders)
        out = []
        for (_, o), (_, row) in zip(orders.iterrows(), shap.iterrows()):
            strongest = row.abs().sort_values(ascending=False).index[:top]
            out.append("SHAP: " + "; ".join(f"{f} = {o[f]:.3g} ({row[f]:+.2f})" for f in strongest))
        return out
