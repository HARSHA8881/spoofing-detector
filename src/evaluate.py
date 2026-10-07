"""Metrics for a rare-event ranking problem.

Accuracy is useless here (never flagging anything scores 99.9%+), so everything
is about the ranking: PR-AUC, and precision / recall in the top k alerts an
analyst would actually review in a day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

KS = (20, 50, 100)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores (stable, so ties resolve by position)."""
    return np.argsort(-np.asarray(scores), kind="stable")[:k]


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    return float(np.asarray(y)[top_k(scores, k)].mean())


def recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    y = np.asarray(y)
    return float(y[top_k(scores, k)].sum() / max(y.sum(), 1))


def hard_negative_fpr(is_hard_neg: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of the legitimate large cancels that land in the top k alerts."""
    is_hard_neg = np.asarray(is_hard_neg)
    return float(is_hard_neg[top_k(scores, k)].sum() / max(is_hard_neg.sum(), 1))


def top_episodes(orders: pd.DataFrame, scores: np.ndarray, k: int) -> np.ndarray:
    """Alert groups an analyst would open first: each group ranked by its best order."""
    ranked = orders.alert_group.to_numpy()[np.argsort(-np.asarray(scores), kind="stable")]
    _, first = np.unique(ranked, return_index=True)
    return ranked[np.sort(first)[:k]]


def episode_metrics(orders: pd.DataFrame, scores: np.ndarray, k: int) -> tuple[float, float]:
    """Precision and recall when the unit reviewed is an episode, not an order.

    Precision: share of the top k alert groups that contain an injected spoof.
    Recall: share of the injected episodes with an order in those groups.
    """
    groups = top_episodes(orders, scores, k)
    spoof = orders[orders.label == 1]
    hit = spoof[spoof.alert_group.isin(groups)]
    precision = np.isin(groups, spoof.alert_group.to_numpy()).mean()
    return float(precision), float(hit.episode.nunique() / max(spoof.episode.nunique(), 1))


def evaluate_day(orders: pd.DataFrame, scores: np.ndarray, ks=KS) -> dict:
    """Metrics for one stock-day. ``orders`` needs ``label`` and ``role``."""
    y = orders.label.to_numpy()
    hard = (orders.role == "hard_neg").to_numpy()
    out = {"n_orders": len(orders), "n_spoof": int(y.sum()), "n_hard_neg": int(hard.sum()),
           "pr_auc": float(average_precision_score(y, scores))}
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(y, scores, k)
        out[f"recall@{k}"] = recall_at_k(y, scores, k)
        out[f"hard_neg_fpr@{k}"] = hard_negative_fpr(hard, scores, k)
        if "alert_group" in orders:
            out[f"ep_precision@{k}"], out[f"ep_recall@{k}"] = episode_metrics(orders, scores, k)
    return out


def bootstrap(y: np.ndarray, scores: np.ndarray, ks=KS, n_boot: int = 200, seed: int = 0) -> dict:
    """Bootstrap replicates of PR-AUC and precision@k for one stock-day.

    Poisson bootstrap: every order gets a random weight (how many times it was
    "drawn"). The orders are sorted by score once, so each replicate is O(n).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)[np.argsort(-np.asarray(scores), kind="stable")].astype(float)
    out = {"pr_auc": np.empty(n_boot), **{f"precision@{k}": np.empty(n_boot) for k in ks}}
    for b in range(n_boot):
        w = rng.poisson(1.0, len(y))
        hits, seen = np.cumsum(w * y), np.cumsum(w)
        precision = hits / np.maximum(seen, 1)
        out["pr_auc"][b] = (precision * w * y).sum() / max(hits[-1], 1)
        for k in ks:
            out[f"precision@{k}"][b] = precision[min(np.searchsorted(seen, k), len(y) - 1)]
    return out


def level_alert_overlap(alerts: pd.DataFrame, orders: pd.DataFrame, slack_s: float = 1.0) -> dict:
    """How level-based alerts (no order ids) line up with injected spoof orders.

    An alert matches a spoof order on the same side whose life covers the alert
    time, give or take ``slack_s`` for the sampling grid.
    """
    spoof = orders[orders.label == 1]
    caught = np.zeros(len(spoof), dtype=bool)
    matched = np.zeros(len(alerts), dtype=bool)
    for direction in (-1, 1):
        a_idx = np.flatnonzero((alerts.direction == direction).to_numpy())
        times = alerts.time.to_numpy()[a_idx]
        for j in np.flatnonzero((spoof.direction == direction).to_numpy()):
            inside = (times >= spoof.t_add.iloc[j] - slack_s) & (times <= spoof.t_end.iloc[j] + slack_s)
            caught[j] = inside.any()
            matched[a_idx[inside]] = True
    return {"n_alerts": len(alerts), "alert_precision": float(matched.mean()) if len(alerts) else 0.0,
            "spoof_recall": float(caught.mean()) if len(spoof) else 0.0}
