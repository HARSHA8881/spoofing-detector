# Spoofing detector

A surveillance research tool for limit order books. It reads NASDAQ order-level
data (LOBSTER), plants labelled synthetic spoofing episodes in it, rebuilds the
book, turns every order into a feature row, and compares three detectors on
days and injection settings they have not seen. A Streamlit dashboard shows the
alerts for a human to review.

It flags order patterns. It does not accuse anyone, and it does not trade.

## Results

Test set: AMZN and MSFT on 2012-06-21 (462k orders, 271 injected spoof orders,
188 hard negatives). Trained on AAPL, GOOG and INTC with different injection
settings. Mean over the two test days:

| Detector | PR-AUC | Precision@20 | Precision@50 | Precision@100 | Recall@100 | Hard-negative FPR@50 |
|---|---|---|---|---|---|---|
| Rule baseline | 0.146 | 0.03 | 0.15 | 0.20 | 0.17 | 0.08 |
| Isolation Forest | 0.006 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 |
| LightGBM | **0.258** | **0.50** | **0.58** | **0.43** | **0.34** | 0.09 |

The base rate is 0.06%, so a random ranking has PR-AUC 0.0006.

What the numbers say:

- **LightGBM is the only detector an analyst could work from**, and it is far
  from solved: 4 in 10 of its top 50 alerts per day are not injected spoofs.
- **It learned the injection style.** On the same test days injected with the
  *training* settings, its PR-AUC is 0.620 and precision@100 is 0.69. Moving to
  smaller orders (k = 5-15 instead of 10-20), longer holds and one level deeper
  costs more than half of that. On MSFT it finds 3% of spoof orders under 6x
  the median size in its top 100.
- **Hard negatives are a real problem.** A large order cancelled after the
  price moved away looks almost the same as a spoof without trader IDs. On AMZN
  31 of LightGBM's top 100 alerts are hard negatives.
- **The rule baseline flags too much.** Tuned on training days it picked
  `size_ratio > 10` and `lifetime < 5s`. As fired it gives 98 alerts on AMZN
  (39% precision, 34% recall) but 904 on MSFT (7.5% precision). Its ranking
  inside the flagged set is weak, hence the low precision@20.
- **Isolation Forest is no better than chance here.** It flags whatever is
  rare, and most rare orders are not spoofs.
- **Level-based alerts (no order IDs, the Kite-style path)** overlap 65-85% of
  spoof orders but only 4-5% of those alerts are spoofs. Without order IDs the
  signal is too weak to use alone.

Per-day numbers are in `artifacts/results_per_day.csv` after a run.

## Run it

```bash
uv sync
uv run python -m src.run          # about 20 seconds, writes artifacts/
uv run pytest -q                  # 32 tests
uv run streamlit run app/dashboard.py
```

Data: download the free 10-level sample files (AAPL, AMZN, GOOG, INTC, MSFT,
2012-06-21) from the LOBSTER sample page and unzip them into `data/`.

## How it works

| Step | File | Notes |
|---|---|---|
| Load | `src/load.py` | Prices stay integers (dollars x 10,000) so they are exact book keys |
| Book engine | `src/replay.py` | Hash map of orders plus a sorted map of price levels per side; O(n log L) |
| Injection | `src/inject.py` | Spoof episodes (layering, repeats) and hard negatives; settings and seed in `InjectionConfig` |
| Features | `src/features.py` | One row per order, fully vectorised, no look-ahead in the rolling median |
| Detectors | `src/detectors/` | Rules, Isolation Forest, LightGBM; each has `fit`, `score`, `explain` |
| Evaluation | `src/evaluate.py` | PR-AUC, precision/recall@k, hard-negative FPR |
| Experiment | `src/run.py` | Train/test split by stock-day and by injection settings |
| Dashboard | `app/dashboard.py` | Price timeline with alerts, book heatmap, order life story, explanation |

### Why the book is not rebuilt from messages alone

A LOBSTER file requested with 10 levels only contains events that touched the
top 10 levels. Orders resting at the open have no add message, and an order
that drifts below level 10 can be deleted without the file saying so. A pure
message replay of these files keeps phantom liquidity and matches the official
book at the best price on under 6% of rows.

So the engine is used in two ways that are both exact:

- **Validation.** For every message, the size change the engine predicts at
  that price is compared with the change in the official book. They agree on
  99.99% to 100% of events across the five files.
- **After injection.** Injected orders are passive and leave through their own
  messages, so the new book is the official book plus a replay of the injected
  orders only (`overlay_book`).

On a full-information stream (the synthetic one in the tests) the engine
reproduces the whole book exactly.

### Injection design

- A spoof order is k x the day's median size on its side, 1-3 levels from the
  best price, held 0.5-5 seconds, then cancelled. During the hold a small
  opposite-side order is added at or inside the best price and executed.
- An episode can layer up to 3 orders and repeat up to 3 times.
- A spoof is pulled just before the best price reaches it, so nothing in the
  original data would have traded against it.
- A hard negative is the same kind of large order, cancelled 50-500 ms after
  the mid price moved away, with no opposite-side order of its own.
- History does not react to injected orders. To keep "the price moved" from
  separating the classes by construction, half the spoof episodes are placed
  where the market happened to move the way the spoof would have pushed.

Explanations: LightGBM alerts carry TreeSHAP values, computed by LightGBM's own
`pred_contrib` (the same values `shap.TreeExplainer` returns). Rule alerts list
which rules fired. Isolation Forest alerts list the most unusual features.

## Limitations

- **Synthetic labels.** The detectors are scored on spoofing I wrote. Good
  numbers here show the pipeline can find this pattern, not that it finds real
  spoofing. The seen-versus-unseen gap above is the honest measure of that risk.
- **Known injection artefact.** The opposite-side fill is 1-3x the median order
  size, so 61% of spoof orders have `opp_exec_rel` between 1 and 3 against 3%
  of natural large fast cancels. No single feature gets PR-AUC above 0.02 on
  its own, but the model can lean on this.
- **No trader IDs.** The fake order cannot be linked to the opposite-side trade
  by the same participant. Alerts are order-level patterns, not proof of intent.
- **No market reaction.** An injected order cannot move the real price.
- **One calendar day.** The free samples cover five stocks on one date, so
  "unseen days" means unseen stocks on that date.
- **US equities, not Indian F&O.** The live Kite path is not built. Only its
  level-based feature (`level_jump_features`) exists, tested on LOBSTER levels.
- **Not built from the guide's stack:** DuckDB, MLflow (a results CSV per run
  instead), the `shap` package, Redis, and the optional autoencoder/LSTM.

## Boundaries

- The detector's insights are not to be used to design orders that avoid detection.
- No real firm or person is named or accused from these alerts.
- LOBSTER samples are for learning; `data/` and `artifacts/` are git-ignored and
  derived datasets should not be published without checking LOBSTER's terms.
