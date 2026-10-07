"""Rule baseline: big for its stock, brief, cancelled unfilled, trades on the other side.

"Big" is a percentile inside the stock-day rather than a fixed multiple of the
median, so a deep one-tick stock does not flood the alert list. The score is a
scorecard (a sum of 0-1 parts), so alerts inside the flagged set are ranked by
how strongly they fit instead of by size alone.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

SIZE_PCT_GRID = (0.95, 0.98, 0.99, 0.995, 0.998)
LIFETIME_GRID = (1, 2, 5, 10, 30)


class RuleDetector:
    name = "rules"

    def __init__(self, min_size_pct: float = 0.99, max_lifetime_s: float = 5.0):
        self.min_size_pct = min_size_pct
        self.max_lifetime_s = max_lifetime_s

    def conditions(self, orders: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "large": orders.size_pct >= self.min_size_pct,
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
        for size, life in product(SIZE_PCT_GRID, LIFETIME_GRID):
            flagged = RuleDetector(size, life).flag(orders)
            hits = (flagged & y).sum()
            f1 = 2 * hits / max(flagged.sum() + y.sum(), 1)
            if f1 > best:
                best, self.min_size_pct, self.max_lifetime_s = f1, size, life
        return self

    def parts(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Scorecard: every part is between 0 and 1."""
        cond = self.conditions(orders)
        return pd.DataFrame({
            "size for this stock": orders.size_pct,
            "share of visible depth": orders.depth_share_add.fillna(0),
            "brief": 1 / (1 + orders.lifetime_s.fillna(np.inf) / self.max_lifetime_s),
            "cancelled unfilled": cond.cancelled_unfilled.astype(float),
            "opposite trades": cond.opposite_trades.astype(float),
            "cancelled with others": orders.ep_cancel_together.clip(upper=2) / 2,
        })

    def score(self, orders: pd.DataFrame) -> np.ndarray:
        return self.parts(orders).sum(axis=1).to_numpy()

    def explain(self, orders: pd.DataFrame) -> list[str]:
        cond = self.conditions(orders)
        out = []
        for (_, o), (_, c) in zip(orders.iterrows(), cond.iterrows()):
            parts = [
                f"larger than {o.size_pct:.1%} of the day's orders "
                f"({'meets' if c.large else 'below'} the {self.min_size_pct:.1%} bar)",
                f"lived {o.lifetime_s:.2f}s ({'<' if c.brief else 'not <'} {self.max_lifetime_s:g}s)",
                "cancelled with no fill" if c.cancelled_unfilled else f"status {o.status}, fill {o.fill_ratio:.0%}",
                f"{o.opp_exec_qty:.0f} shares traded on the opposite side during its life",
            ]
            verdict = "all 4 rules fired" if c.all() else f"{int(c.sum())} of 4 rules fired"
            out.append(f"{verdict}: " + "; ".join(parts))
        return out
