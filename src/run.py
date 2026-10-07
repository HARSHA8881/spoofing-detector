"""End-to-end experiment: inject, rebuild the book, featurise, train, evaluate.

    uv run python -m src.run             # 5 injection seeds, a few minutes
    uv run python -m src.run --seeds 1   # quick

Train and test use different stock-days AND different injection settings, so a
model cannot score well by memorising how the training episodes were built.
Every result is "after the fact": an order is scored once it has ended.
Outputs land in ``artifacts/``; the dashboard reads them from there.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate, report
from src.detectors import (IsolationForestDetector, LightGBMDetector, LightGBMRanker,
                           RuleDetector)
from src.features import (FEATURE_GROUPS, FEATURES, REALTIME_FEATURES, build_features,
                          level_jump_features)
from src.inject import InjectionConfig, inject
from src.load import book_columns, find_stock_days, load_messages, load_orderbook
from src.replay import check_against_official, overlay_book

DATA, OUT = Path("data"), Path("artifacts")
DEPTH = 10
ALERTS_KEPT = 200  # per detector and day, with explanations, for the dashboard
HEADLINE = ["pr_auc", "precision@20", "precision@50", "precision@100", "recall@100",
            "hard_neg_fpr@50", "ep_precision@20", "ep_precision@50", "ep_recall@50"]

NARROW = dict(k_choices=(10, 20), hold_s=(0.5, 3.0), levels_away=(1, 2))
# a wide spread of settings that covers the test ones
WIDE = dict(k_choices=(3, 5, 8, 10, 15, 20, 30), hold_s=(0.2, 10.0), levels_away=(1, 3))
# unseen days, smaller orders, longer holds, one level deeper
UNSEEN = dict(k_choices=(5, 8, 15), hold_s=(1.0, 5.0), levels_away=(1, 3))
TRAIN_DAYS, TEST_DAYS = ("AAPL", "GOOG", "INTC"), ("AMZN", "MSFT")


def configs(days, settings: dict, seed: int, offset: int) -> dict[str, InjectionConfig]:
    return {t: InjectionConfig(**settings, seed=1000 * seed + offset + i) for i, t in enumerate(days)}


class Days:
    """Loads each stock-day once and builds injected, featurised versions of it."""

    def __init__(self):
        self.files = find_stock_days(DATA)
        self.raw = {}

    def load(self, ticker: str):
        if ticker not in self.raw:
            msg_path, book_path = self.files[ticker]
            msgs, official = load_messages(msg_path), load_orderbook(book_path, DEPTH)
            print(f"{ticker}: {len(msgs):,} messages, engine matches the official book on "
                  f"{check_against_official(msgs, official):.4%} of events")
            self.raw[ticker] = (msgs, official)
        return self.raw[ticker]

    def prepare(self, ticker: str, cfg: InjectionConfig, keep_raw: bool = False) -> pd.DataFrame:
        msgs, official = self.load(ticker)
        injected = inject(msgs, official, cfg)
        book = overlay_book(injected, official, DEPTH)
        orders = build_features(injected, book)
        orders.insert(0, "ticker", ticker)
        if keep_raw:
            injected.to_parquet(OUT / "messages" / f"{ticker}.parquet")
            pd.DataFrame(book, columns=book_columns(DEPTH)).to_parquet(OUT / "book" / f"{ticker}.parquet")
        return orders

    def many(self, cfgs: dict[str, InjectionConfig], keep_raw: bool = False) -> dict[str, pd.DataFrame]:
        return {t: self.prepare(t, cfg, keep_raw) for t, cfg in cfgs.items()}


def scoreable(orders: pd.DataFrame) -> pd.DataFrame:
    """The spoofer's own small filled order is part of an episode but is neither a
    spoof order nor an innocent one, so it is left out of training and scoring."""
    return orders[orders.role != "genuine"]


def training_frame(days: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return scoreable(pd.concat(days.values()))


def fit_all(train: pd.DataFrame) -> list:
    return [RuleDetector().fit(train), IsolationForestDetector().fit(train),
            LightGBMDetector().fit(train), LightGBMRanker().fit(train)]


def score_days(days: dict[str, pd.DataFrame], detectors: list) -> pd.DataFrame:
    """One metrics row per (detector, day). Scores are written into the day frames."""
    rows = []
    for ticker, orders in days.items():
        scored = scoreable(orders)
        for det in detectors:
            scores = det.score(scored)
            orders[f"score_{det.name}"] = pd.Series(scores, index=scored.index)
            rows.append({"detector": det.name, "ticker": ticker, **evaluate.evaluate_day(scored, scores)})
    return pd.DataFrame(rows)


def mean_over_days(per_day: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return per_day.groupby("detector")[HEADLINE].mean().loc[names]


def confidence_intervals(days: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """95% bootstrap interval of each day-averaged metric."""
    rows = []
    for name in names:
        reps = [evaluate.bootstrap(scoreable(o).label.to_numpy(), scoreable(o)[f"score_{name}"].to_numpy())
                for o in days.values()]
        for metric in reps[0]:
            mean = np.mean([r[metric] for r in reps], axis=0)
            rows.append({"detector": name, "metric": metric, "low": np.percentile(mean, 2.5),
                         "high": np.percentile(mean, 97.5)})
    return pd.DataFrame(rows)


def rule_flags(days: dict[str, pd.DataFrame], rules: RuleDetector) -> pd.DataFrame:
    rows = []
    for ticker, orders in days.items():
        scored = scoreable(orders)
        flagged = rules.flag(scored)
        y, hard = scored.label.to_numpy() == 1, (scored.role == "hard_neg").to_numpy()
        rows.append({"ticker": ticker, "n_alerts": int(flagged.sum()),
                     "precision": float(y[flagged].mean()) if flagged.any() else 0.0,
                     "recall": float(flagged[y].mean()), "hard_neg_fpr": float(flagged[hard].mean())})
    return pd.DataFrame(rows)


def recall_by_reason(days: dict[str, pd.DataFrame], name: str, k: int = 100) -> pd.DataFrame:
    """Is each injected order (by why it was cancelled) inside its day's top k?"""
    frames = []
    for orders in days.values():
        scored = scoreable(orders)
        rank = scored[f"score_{name}"].rank(ascending=False, method="first")
        injected = scored[scored.reason != ""]
        frames.append(pd.DataFrame({"reason": injected.reason, "in_top_k": rank[injected.index] <= k}))
    return pd.concat(frames)


def ablation(train: pd.DataFrame, days: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Retrain LightGBM without one feature group at a time."""
    variants = {"all features": FEATURES, "real-time features only": REALTIME_FEATURES}
    variants |= {f"without {g}": [f for f in FEATURES if f not in cols] for g, cols in FEATURE_GROUPS.items()}
    rows = []
    for label, features in variants.items():
        det = LightGBMDetector(features=features).fit(train)
        per_day = pd.DataFrame([evaluate.evaluate_day(scoreable(o), det.score(scoreable(o)))
                                for o in days.values()])
        rows.append({"variant": label, "n_features": len(features), **per_day[HEADLINE].mean()})
    out = pd.DataFrame(rows)
    out["pr_auc_change"] = out.pr_auc - out.pr_auc.iloc[0]
    return out


def save_dashboard_files(days: dict[str, pd.DataFrame], detectors: list, train: pd.DataFrame) -> None:
    alerts = []
    for ticker, orders in days.items():
        scored = scoreable(orders)
        for det in detectors:
            scores = scored[f"score_{det.name}"].to_numpy()
            best = evaluate.top_k(scores, ALERTS_KEPT)
            top = scored.iloc[best]
            detail = det.shap_values(top) if det.name.startswith("lgbm") else top[FEATURES].astype(float)
            alerts.append(pd.DataFrame({
                "ticker": ticker, "detector": det.name, "rank": np.arange(1, len(top) + 1),
                "order_id": top.index, "score": scores[best], "why": det.explain(top),
                "detail": [json.dumps(d) for d in detail.round(4).to_dict("records")]}))
        orders.reset_index().to_parquet(OUT / "features" / f"{ticker}.parquet")
    pd.concat(alerts).to_parquet(OUT / "alerts.parquet")
    # the training set, so the dashboard can retrain with a reviewer's labels
    train.reset_index().to_parquet(OUT / "features" / "train.parquet")


def level_based_check(ticker: str, orders: pd.DataFrame) -> dict:
    """What a 5-level feed without order ids (the Kite path) would have caught."""
    msgs = pd.read_parquet(OUT / "messages" / f"{ticker}.parquet", columns=["time"])
    book = pd.read_parquet(OUT / "book" / f"{ticker}.parquet").to_numpy()
    lv = level_jump_features(msgs.time.to_numpy(), book, depth=5)
    alerts = lv[(lv.level_jump_z >= 4) & (lv.revert_s <= 6)]
    return {"ticker": ticker, **evaluate.level_alert_overlap(alerts, orders)}


def main(n_seeds: int) -> None:
    t0 = time.time()
    for sub in ("messages", "book", "features"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    days = Days()
    per_seed, reasons = [], []

    for seed in range(n_seeds):
        train = training_frame(days.many(configs(TRAIN_DAYS, NARROW, seed, 11)))
        test = days.many(configs(TEST_DAYS, UNSEEN, seed, 21), keep_raw=seed == 0)
        detectors = fit_all(train)
        names = [d.name for d in detectors]
        per_day = score_days(test, detectors)
        per_seed.append(mean_over_days(per_day, names).assign(seed=seed))
        reasons.append(recall_by_reason(test, "lgbm"))
        print(f"seed {seed}: " + ", ".join(f"{n} PR-AUC {v:.3f}" for n, v in per_seed[-1].pr_auc.items()))
        if seed > 0:
            continue

        # everything below is reported for seed 0 only
        rules = detectors[0]
        print(f"rules tuned on train: size_pct >= {rules.min_size_pct}, lifetime < {rules.max_lifetime_s}s")
        per_day.to_csv(OUT / "results_per_day.csv", index=False)
        summary = mean_over_days(per_day, names)
        summary.to_csv(OUT / "results.csv")
        ci = confidence_intervals(test, names)
        ci.to_csv(OUT / "results_ci.csv", index=False)
        flags = rule_flags(test, rules)
        flags.to_csv(OUT / "results_rule_flags.csv", index=False)
        save_dashboard_files(test, detectors, train)

        lgbm = detectors[2]
        cases = report.error_cases(test, lgbm)
        (OUT / "error_analysis.json").write_text(json.dumps(cases, indent=1))
        (OUT / "error_analysis.md").write_text(report.error_markdown(cases))

        abl = ablation(train, test)
        abl.to_csv(OUT / "results_ablation.csv", index=False)

        # follow-ups: same test days with the training settings, and wide training
        seen = days.many(configs(TEST_DAYS, NARROW, seed, 21))
        wide_train = training_frame(days.many(configs(TRAIN_DAYS, WIDE, seed, 31)))
        wide = LightGBMDetector().fit(wide_train)
        follow = pd.DataFrame({
            "narrow training, unseen settings": summary.loc["lgbm"],
            "wide training, unseen settings": mean_over_days(score_days(test.copy(), [wide]), ["lgbm"]).loc["lgbm"],
            "narrow training, training settings": mean_over_days(score_days(seen, [lgbm]), ["lgbm"]).loc["lgbm"],
        }).T
        # scoring with the wide model overwrote score_lgbm in the frames; put the main scores back
        score_days(test, [lgbm])
        follow.to_csv(OUT / "results_followup.csv")
        level = pd.DataFrame([level_based_check(t, scoreable(o)) for t, o in test.items()])
        level.to_csv(OUT / "results_level_based.csv", index=False)

        base = per_day[per_day.detector == "rules"].set_index("ticker")
        print("\nTest days:", ", ".join(f"{t}: {int(r.n_spoof)} spoof orders in {int(r.n_orders):,}"
                                        for t, r in base.iterrows()))
        print("\nSeed 0, mean over test days\n", summary.round(3).to_string())
        print("\n95% bootstrap intervals\n", ci.pivot(index="detector", columns="metric", values=["low", "high"])
              .round(3).loc[names].to_string())
        print("\nRule flags as fired\n", flags.round(3).to_string(index=False))
        print("\nLightGBM ablation\n", abl[["variant", "n_features", "pr_auc", "pr_auc_change", "precision@50",
                                           "hard_neg_fpr@50"]].round(3).to_string(index=False))
        print("\nLightGBM follow-ups\n", follow.round(3).to_string())
        print("\nLevel-based alerts (no order ids)\n", level.round(3).to_string(index=False))

    seeds = pd.concat(per_seed).reset_index()
    seeds.to_csv(OUT / "results_per_seed.csv", index=False)
    names = list(per_seed[0].index)
    agg = seeds.groupby("detector")[HEADLINE].agg(["mean", "std"]).loc[names]
    agg.to_csv(OUT / "results_seeds.csv")
    by_reason = pd.concat(reasons).groupby("reason").in_top_k.agg(["mean", "size"])
    by_reason.columns = ["in_top_100", "n_orders"]
    by_reason.to_csv(OUT / "results_by_reason.csv")
    print(f"\nMean and std over {n_seeds} injection seeds\n", agg.round(3).to_string())
    print(f"\nLightGBM: share of injected orders in the day's top 100, by why they were cancelled "
          f"({n_seeds} seeds pooled)\n", by_reason.round(3).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    main(parser.parse_args().seeds)
