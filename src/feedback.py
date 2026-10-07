"""Reviewer feedback: save an analyst's verdicts and retrain with them.

A simple form of active learning. The verdicts on a day's alerts are added to
the training set with extra weight, LightGBM is refitted, and the day is
rescored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.detectors import LightGBMDetector

LABELS = Path("artifacts") / "review_labels.csv"
REVIEW_WEIGHT = 25.0  # one reviewed alert counts as much as 25 training rows


def load_labels(path: Path = LABELS) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({"ticker": pd.Series(dtype=str), "order_id": pd.Series(dtype="int64"),
                             "verdict": pd.Series(dtype=str)})
    return pd.read_csv(path)


def save_label(ticker: str, order_id: int, verdict: str, path: Path = LABELS) -> pd.DataFrame:
    """Record one verdict ("spoof" or "not spoof"); a second verdict on the same order replaces the first."""
    labels = load_labels(path)
    labels = labels[~((labels.ticker == ticker) & (labels.order_id == order_id))]
    labels = pd.concat([labels, pd.DataFrame([{"ticker": ticker, "order_id": order_id, "verdict": verdict}])])
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    return labels


def retrain(train: pd.DataFrame, day: pd.DataFrame, labels: pd.DataFrame):
    """Refit LightGBM on the training set plus the reviewed orders of ``day``.

    ``day`` is indexed by order id. Returns (detector, scores for every order in day).
    """
    reviewed = day.loc[day.index.intersection(labels.order_id)].copy()
    verdict = labels.set_index("order_id").verdict.reindex(reviewed.index)
    reviewed["label"] = (verdict == "spoof").astype("int8").to_numpy()
    frame = pd.concat([train, reviewed])
    weight = np.r_[np.ones(len(train)), np.full(len(reviewed), REVIEW_WEIGHT)]
    detector = LightGBMDetector().fit(frame, sample_weight=weight)
    return detector, pd.Series(detector.score(day), index=day.index)
