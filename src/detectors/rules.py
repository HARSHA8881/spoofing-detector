"""Rule baseline: big, brief, cancelled unfilled, with trades on the other side."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

SIZE_GRID = (3, 5, 8, 10, 15, 20)
LIFETIME_GRID = (1, 2, 5, 10, 30)


class RuleDetector:
    name = "rules"

    def __init__(self, min_size_ratio: float = 5.0, max_lifetime_s: float = 5.0):
        self.min_size_ratio = min_size_ratio
        self.max_lifetime_s = max_lifetime_s

    def conditions(self, orders: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "large": orders.size_ratio > self.min_size_ratio,
            "brief": orders.lifetime_s < self.max_lifetime_s,
            "cancelled_unfilled": (orders.status == "deleted") & (orders.fill_ratio == 0),
            "opposite_trades": orders.opp_exec_qty > 0,
        })

    def flag(self, orders: pd.DataFrame) -> np.ndarray:
        return self.conditions(orders).all(axis=1).to_numpy()

    def fit(self, orders: pd.DataFrame) -> "RuleDetector":
        """Pick the two thresholds with the best F1 on the training labels."""
        y = orders.label.to_numpy().astype(bool)
        best = -1.0
        for size, life in product(SIZE_GRID, LIFETIME_GRID):
            flagged = RuleDetector(size, life).flag(orders)
            hits = (flagged & y).sum()
            f1 = 2 * hits / max(flagged.sum() + y.sum(), 1)
            if f1 > best:
                best, self.min_size_ratio, self.max_lifetime_s = f1, size, life
        return self

    def score(self, orders: pd.DataFrame) -> np.ndarray:
        """Number of conditions met; ties broken by size so the ranking is usable."""
        met = self.conditions(orders).sum(axis=1).to_numpy()
        return met + orders.size_ratio.to_numpy() / (1 + orders.size_ratio.to_numpy())

    def explain(self, orders: pd.DataFrame) -> list[str]:
        cond = self.conditions(orders)
        out = []
        for (_, o), (_, c) in zip(orders.iterrows(), cond.iterrows()):
            parts = [
                f"size {o.size_ratio:.1f}x median ({'>' if c.large else 'not >'} {self.min_size_ratio:g}x)",
                f"lived {o.lifetime_s:.2f}s ({'<' if c.brief else 'not <'} {self.max_lifetime_s:g}s)",
                "cancelled with no fill" if c.cancelled_unfilled else f"status {o.status}, fill {o.fill_ratio:.0%}",
                f"{o.opp_exec_qty:.0f} shares traded on the opposite side during its life",
            ]
            verdict = "all 4 rules fired" if c.all() else f"{int(c.sum())} of 4 rules fired"
            out.append(f"{verdict}: " + "; ".join(parts))
        return out
