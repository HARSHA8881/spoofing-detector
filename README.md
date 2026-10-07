# Spoofing detector

A surveillance research tool for limit order books. It reads NASDAQ order-level
data (LOBSTER), plants labelled synthetic spoofing episodes in it, rebuilds the
book, turns every order into a feature row, and compares three detectors on
days and injection settings they have not seen. A Streamlit dashboard shows the
alerts for a human to review.

It flags order patterns. It does not accuse anyone, and it does not trade.

**Headline:** on stocks it never trained on, the best model puts injected spoof
orders in 84% of its top 50 daily alerts (mean of 5 injection seeds), against a
base rate of 1 spoof in 1,600 orders. Its remaining false alarms are mostly
legitimate-looking large cancels, and only a third of spoofs cancelled on a
plain timer make its top 100.

## Results

Test set: AMZN and MSFT on 2012-06-21 (about 462k orders, 290 injected spoof
orders and 190 hard negatives per seed). Trained on AAPL, GOOG and INTC with
different injection settings. Every number is "after the fact": an order is
scored once it has ended. Mean ± standard deviation over 5 injection seeds,
each averaged over the two test days:

| Detector | PR-AUC | Precision@20 | Precision@50 | Precision@100 | Hard-negative FPR@50 |
|---|---|---|---|---|---|
| Rule scorecard | 0.17 ± 0.01 | 0.20 ± 0.04 | 0.18 ± 0.02 | 0.22 ± 0.02 | 0.00 |
| Isolation Forest | 0.04 ± 0.01 | 0.06 ± 0.03 | 0.05 ± 0.01 | 0.05 ± 0.01 | 0.04 |
| LightGBM classifier | 0.51 ± 0.02 | 0.87 ± 0.09 | 0.79 ± 0.09 | 0.66 ± 0.04 | 0.04 |
| LightGBM ranker (LambdaRank) | **0.56 ± 0.02** | **0.92 ± 0.03** | **0.84 ± 0.06** | **0.68 ± 0.06** | 0.04 |

Scored as episodes (layered orders and quick repeats on one side count as one
alert, ranked by their most suspicious order):

| Detector | Episode precision@20 | Episode precision@50 | Injected episodes found in top 50 |
|---|---|---|---|
| Rule scorecard | 0.27 ± 0.03 | 0.24 ± 0.04 | 0.33 ± 0.03 |
| Isolation Forest | 0.12 ± 0.05 | 0.13 ± 0.02 | 0.18 ± 0.03 |
| LightGBM classifier | 0.77 ± 0.09 | 0.60 ± 0.04 | 0.73 ± 0.02 |
| LightGBM ranker | **0.83 ± 0.05** | **0.61 ± 0.06** | **0.74 ± 0.04** |

95% bootstrap intervals for seed 0 (resampling orders): LightGBM classifier
PR-AUC 0.45-0.56, ranker 0.48-0.57; precision@50 0.73-0.87 and 0.70-0.87. The
ranker's edge over the classifier is inside those intervals on one seed, but it
is ahead on PR-AUC in all five seeds.

### What changed the result

The first version of this project scored 0.25 PR-AUC and 0.52 precision@50 with
ten features. Adding features that describe the moment of the cancel, size
relative to the book around the order, and episode grouping doubled that.
Removing one feature group at a time from the LightGBM classifier (seed 0):

| Variant | PR-AUC | Change | Precision@50 |
|---|---|---|---|
| All 26 features | 0.498 | | 0.81 |
| Without cancel context | 0.268 | -0.230 | 0.48 |
| Without episode features | 0.395 | -0.103 | 0.74 |
| Without placement | 0.408 | -0.091 | 0.73 |
| Without opposite trades | 0.416 | -0.083 | 0.74 |
| Without stock context | 0.442 | -0.056 | 0.85 |
| Without activity window | 0.453 | -0.045 | 0.85 |
| Without size | 0.493 | -0.005 | 0.87 |
| Without life and fill | 0.561 | +0.063 | 0.81 |
| Real-time features only (20) | 0.509 | +0.011 | 0.81 |

Cancel context is the signal that matters most. Changes smaller than about
0.05 are within the seed-to-seed noise, so "life and fill" hurting and "size"
not mattering should not be read as findings. The real-time row uses only
features known the moment the order ends (no day percentiles, day averages or
episode grouping) and loses nothing, so the model could run on a live feed.

### Is the model just finding my injection rule?

Spoofs are cancelled for three different reasons, so this can be checked. Share
of injected orders that reach the LightGBM classifier's top 100 of the day, all
seeds pooled:

| Why the order was cancelled | In top 100 | Orders |
|---|---|---|
| Spoof: best price came toward it | 63% | 145 |
| Spoof: right after its own fill | 53% | 489 |
| Spoof: fixed hold | 33% | 935 |
| Hard negative: price moved away | 15% | 873 |

So yes, partly: spoofs pulled as the price approaches are found twice as often
as spoofs pulled on a timer, which have no cancel-time signature at all. The
timer row is the fairest single number for "spoofing the features were not
designed around".

### Other findings

- **Hard negatives are much better handled than before** (about 4% of them
  reach the top 50, 15% the top 100) but they are still the typical false
  alarm: 3 of the 5 highest-ranked mistakes in `artifacts/error_analysis.md`
  are hard negatives, the other 2 are real orders from the data.
- **Training on a wide range of injection settings made the top of the list
  worse** on seed 0 (PR-AUC 0.39, precision@50 0.32, against 0.50 and 0.81).
  Very small, long-lived training spoofs look like ordinary orders and dilute
  the pattern. This is one seed and has not been repeated.
- **Same test days with the training injection settings:** PR-AUC 0.79. The
  gap to 0.50 is the cost of smaller, longer-held spoofs.
- **The rule scorecard no longer floods MSFT** now that "large" is a per-stock
  percentile (321 flags instead of 907), but it is close to useless there:
  3% precision, 6% recall. On AMZN it flags 199 orders at 45% precision.
- **Isolation Forest stays a negative result** even when fitted only on large
  unfilled cancels with spoof-relevant features: 0.04 PR-AUC.
- **Level-based alerts (no order IDs, the Kite-style path)** overlap 67-94% of
  spoof orders but only 4-6% of those alerts are spoofs.

All tables are regenerated by a run; see `artifacts/results*.csv`.

## Run it

```bash
uv sync
uv run python -m src.run          # 5 injection seeds, about 2 minutes, writes artifacts/
uv run python -m src.run --seeds 1
uv run pytest -q                  # 42 tests
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
| Features | `src/features.py` | One row per order, 26 features in 8 groups, fully vectorised |
| Detectors | `src/detectors/` | Rule scorecard, Isolation Forest, LightGBM classifier and ranker; each has `fit`, `score`, `explain` |
| Evaluation | `src/evaluate.py` | PR-AUC, precision/recall@k, episode-level metrics, hard-negative FPR, bootstrap intervals |
| Experiment | `src/run.py` | Seeds, train/test split by stock-day and injection settings, ablation, follow-ups |
| Reports | `src/report.py` | Order life stories and the error analysis |
| Feedback | `src/feedback.py` | Saves a reviewer's verdicts and retrains LightGBM with them |
| Dashboard | `app/dashboard.py` | Alert review, detector comparison, error analysis |

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
- A spoof is cancelled for one of three reasons, recorded per order: after a
  fixed hold, right after its own fill, or when the best price starts moving
  toward it. Whatever the reason, it is pulled before the best price reaches
  it, so nothing in the original data would have traded against it.
- A hard negative is the same kind of large order, cancelled 50-500 ms after
  the mid price moved away, with no opposite-side order of its own.
- History does not react to injected orders. To keep "the price moved" from
  separating the classes by construction, half the spoof episodes are placed
  where the market happened to move the way the spoof would have pushed.

Explanations: LightGBM alerts carry TreeSHAP values, computed by LightGBM's own
`pred_contrib` (the same values `shap.TreeExplainer` returns). Rule alerts list
which rules fired. Isolation Forest alerts list the most unusual features.

## Dashboard

- **Review alerts:** the day's price with every alert on it, a table of alerts
  (per order or grouped into episodes), and for the selected alert the book
  heatmap, its life story, and the SHAP values or rules behind the score.
- **Feedback loop:** mark an alert "Spoof" or "Not spoof"; verdicts are saved to
  `artifacts/review_labels.csv`. "Retrain with my reviews" refits LightGBM with
  them and adds the result as a detector.
- **Detector comparison:** the tables above as charts, with bootstrap whiskers.
- **Where it fails:** the five worst false positives and five worst misses.

## Limitations

- **Synthetic labels.** The detectors are scored on spoofing I wrote. Good
  numbers here show the pipeline can find this pattern, not that it finds real
  spoofing. The seen-versus-unseen gap above is the honest measure of that risk.
- **Injected fills are simplified.** The spoofer's own opposite-side order is
  one order, fully filled, with a size drawn from that day's real trade sizes.
- **No trader IDs.** The fake order cannot be linked to the opposite-side trade
  by the same participant. Alerts are order-level patterns, not proof of intent.
- **No market reaction.** An injected order cannot move the real price.
- **One calendar day.** The free samples cover five stocks on one date, so
  "unseen days" means unseen stocks on that date.
- **US equities, not Indian F&O.** The live Kite path is not built. Only its
  level-based feature (`level_jump_features`) exists, tested on LOBSTER levels.
- **Percentile and episode features need the whole day.** They are fine for
  after-the-fact surveillance; the real-time feature set drops them.
- **Not built:** an agent-based market simulator (for example ABIDES) in which
  a spoof could really move the price.
- **Not built from the guide's stack:** DuckDB, MLflow (a results CSV per run
  instead), the `shap` package, Redis, and the optional autoencoder/LSTM.

## Boundaries

- The detector's insights are not to be used to design orders that avoid detection.
- No real firm or person is named or accused from these alerts.
- LOBSTER samples are for learning; `data/` and `artifacts/` are git-ignored and
  derived datasets should not be published without checking LOBSTER's terms.
