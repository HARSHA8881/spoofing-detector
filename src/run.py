"""End-to-end experiment: inject, rebuild the book, featurise, train, evaluate.

    uv run python -m src.run

Train and test use different stock-days AND different injection settings, so a
model cannot score well by memorising how the training episodes were built.
Outputs land in ``artifacts/``; the dashboard reads them from there.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate
from src.detectors import IsolationForestDetector, LightGBMDetector, RuleDetector
from src.features import FEATURES, build_features, level_jump_features
from src.inject import InjectionConfig, inject
from src.load import book_columns, find_stock_days, load_messages, load_orderbook
from src.replay import check_against_official, overlay_book

DATA, OUT = Path("data"), Path("artifacts")
DEPTH = 10
ALERTS_KEPT = 200  # per detector and day, with explanations, for the dashboard

TRAIN = {
    "AAPL": InjectionConfig(k_choices=(10, 20), hold_s=(0.5, 3.0), levels_away=(1, 2), seed=11),
    "GOOG": InjectionConfig(k_choices=(10, 20), hold_s=(0.5, 3.0), levels_away=(1, 2), seed=12),
    "INTC": InjectionConfig(k_choices=(10, 20), hold_s=(0.5, 3.0), levels_away=(1, 2), seed=13),
}
TEST = {  # unseen days, smaller orders, longer holds, one level deeper
    "AMZN": InjectionConfig(k_choices=(5, 8, 15), hold_s=(1.0, 5.0), levels_away=(1, 3), seed=21),
    "MSFT": InjectionConfig(k_choices=(5, 8, 15), hold_s=(1.0, 5.0), levels_away=(1, 3), seed=22),
}


def prepare(ticker: str, cfg: InjectionConfig, files: dict, keep_raw: bool) -> pd.DataFrame:
    msg_path, book_path = files[ticker]
    msgs, official = load_messages(msg_path), load_orderbook(book_path, DEPTH)
    engine_ok = check_against_official(msgs, official)
    injected = inject(msgs, official, cfg)
    book = overlay_book(injected, official, DEPTH)
    orders = build_features(injected, book)
    orders.insert(0, "ticker", ticker)
    print(f"{ticker}: {len(msgs):,} messages, engine matches official book on {engine_ok:.4%} "
          f"of events, {int(orders.label.sum())} spoof orders, "
          f"{int((orders.role == 'hard_neg').sum())} hard negatives")
    if keep_raw:
        injected.to_parquet(OUT / "messages" / f"{ticker}.parquet")
        pd.DataFrame(book, columns=book_columns(DEPTH)).to_parquet(OUT / "book" / f"{ticker}.parquet")
    return orders


def level_based_check(ticker: str, orders: pd.DataFrame) -> dict:
    """What a 5-level feed without order ids (the Kite path) would have caught."""
    msgs = pd.read_parquet(OUT / "messages" / f"{ticker}.parquet", columns=["time"])
    book = pd.read_parquet(OUT / "book" / f"{ticker}.parquet").to_numpy()
    lv = level_jump_features(msgs.time.to_numpy(), book, depth=5)
    alerts = lv[(lv.level_jump_z >= 4) & (lv.revert_s <= 6)]
    return {"ticker": ticker, **evaluate.level_alert_overlap(alerts, orders)}


def score_days(days: dict[str, pd.DataFrame], detectors: list, save: bool) -> pd.DataFrame:
    """Score every test day with every detector; one metrics row per (detector, day)."""
    rules = detectors[0]
    rows, alerts = [], []
    for ticker, orders in days.items():
        scored = orders[orders.role != "genuine"]
        for det in detectors:
            scores = det.score(scored)
            rows.append({"detector": det.name, "ticker": ticker, **evaluate.evaluate_day(scored, scores)})
            if not save:
                continue
            orders[f"score_{det.name}"] = pd.Series(scores, index=scored.index)
            best = evaluate.top_k(scores, ALERTS_KEPT)
            top = scored.iloc[best]
            detail = det.shap_values(top) if det.name == "lgbm" else top[FEATURES].astype(float)
            alerts.append(pd.DataFrame({
                "ticker": ticker, "detector": det.name, "rank": np.arange(1, len(top) + 1),
                "order_id": top.index, "score": scores[best], "why": det.explain(top),
                "detail": [json.dumps(d) for d in detail.round(4).to_dict("records")]}))
        flagged = rules.flag(scored)
        y, hard = scored.label.to_numpy() == 1, (scored.role == "hard_neg").to_numpy()
        rows.append({"detector": "rules (flag)", "ticker": ticker, "n_alerts": int(flagged.sum()),
                     "precision": float(y[flagged].mean()) if flagged.any() else 0.0,
                     "recall": float(flagged[y].mean()), "hard_neg_fpr": float(flagged[hard].mean())})
        if save:
            orders.reset_index().to_parquet(OUT / "features" / f"{ticker}.parquet")
    if save:
        pd.concat(alerts).to_parquet(OUT / "alerts.parquet")
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    for sub in ("messages", "book", "features"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    files = find_stock_days(DATA)

    train = pd.concat([prepare(t, cfg, files, keep_raw=False) for t, cfg in TRAIN.items()])
    # ablation first, so the raw files kept for the dashboard are the unseen-settings ones:
    # the same test days injected with the settings the model was trained on
    seen_cfg = {t: replace(TRAIN["AAPL"], seed=cfg.seed) for t, cfg in TEST.items()}
    seen = {t: prepare(t, cfg, files, keep_raw=False) for t, cfg in seen_cfg.items()}
    test = {t: prepare(t, cfg, files, keep_raw=True) for t, cfg in TEST.items()}
    # the spoofer's own small filled order is part of an episode but is neither
    # a spoof order nor an innocent one, so it is left out of training and scoring
    train = train[train.role != "genuine"]

    detectors = [RuleDetector().fit(train), IsolationForestDetector().fit(train),
                 LightGBMDetector().fit(train)]
    rules = detectors[0]
    print(f"rules tuned on train: size_ratio > {rules.min_size_ratio}, lifetime < {rules.max_lifetime_s}s")

    per_day = score_days(test, detectors, save=True)
    per_day.to_csv(OUT / "results_per_day.csv", index=False)
    metrics = ["pr_auc"] + [f"{m}@{k}" for k in evaluate.KS for m in ("precision", "recall", "hard_neg_fpr")]
    order = [d.name for d in detectors]
    summary = per_day.groupby("detector")[metrics].mean().loc[order]
    summary.to_csv(OUT / "results.csv")
    seen_summary = score_days(seen, detectors, save=False).groupby("detector")[metrics].mean().loc[order]
    seen_summary.to_csv(OUT / "results_seen_settings.csv")
    level = pd.DataFrame([level_based_check(t, o[o.role != "genuine"]) for t, o in test.items()])
    level.to_csv(OUT / "results_level_based.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    base = per_day[per_day.detector == "rules"].set_index("ticker")
    print("\nTest days:", ", ".join(f"{t}: {int(r.n_spoof)} spoof orders in {int(r.n_orders):,}"
                                    for t, r in base.iterrows()))
    print("\nUnseen days, unseen injection settings (mean over test days)\n", summary.round(3).to_string())
    print("\nUnseen days, training injection settings\n", seen_summary.round(3).to_string())
    print("\nRule flags as fired\n", per_day[per_day.detector == "rules (flag)"]
          .dropna(axis=1).round(3).to_string(index=False))
    print("\nLevel-based alerts (no order ids)\n", level.round(3).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
