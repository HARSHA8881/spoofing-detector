import numpy as np
import pandas as pd
import pytest

from src import evaluate
from src.detectors import IsolationForestDetector, LightGBMDetector, RuleDetector
from src.features import FEATURES, build_features
from src.inject import InjectionConfig, inject
from src.replay import overlay_book


@pytest.fixture(scope="module")
def orders(toy_day):
    msgs, book = toy_day
    out = inject(msgs, book, InjectionConfig(n_episodes=(40, 40), n_hard_neg=(30, 30), seed=1))
    orders = build_features(out, overlay_book(out, book, depth=10))
    return orders[orders.role != "genuine"]


def test_rule_flags_only_when_every_condition_holds():
    frame = pd.DataFrame({
        "size_ratio": [12, 12, 2, 12, 12], "lifetime_s": [1, 60, 1, 1, 1],
        "status": ["deleted", "deleted", "deleted", "filled", "deleted"],
        "fill_ratio": [0, 0, 0, 1, 0], "opp_exec_qty": [100, 100, 100, 100, 0]})
    det = RuleDetector(min_size_ratio=5, max_lifetime_s=5)
    assert det.flag(frame).tolist() == [True, False, False, False, False]
    assert det.score(frame).argmax() == 0
    assert det.explain(frame)[0].startswith("all 4 rules fired")


def test_rule_fit_picks_thresholds_from_the_grid(orders):
    det = RuleDetector().fit(orders)
    flagged = det.flag(orders)
    assert flagged.sum() > 0 and orders.label[flagged].mean() > orders.label.mean()


def test_every_detector_scores_and_explains(orders):
    for det in (RuleDetector(), IsolationForestDetector(), LightGBMDetector()):
        scores = det.fit(orders).score(orders)
        assert scores.shape == (len(orders),) and np.isfinite(scores).all()
        why = det.explain(orders.head(3))
        assert len(why) == 3 and all(isinstance(w, str) and w for w in why)


def test_lightgbm_learns_the_injected_pattern_and_shap_adds_up(orders):
    det = LightGBMDetector().fit(orders)
    scores = det.score(orders)
    assert evaluate.precision_at_k(orders.label.to_numpy(), scores, 20) > 0.8  # in-sample sanity
    shap = det.shap_values(orders.head(50))
    assert list(shap.columns) == FEATURES
    raw = det.model.booster_.predict(orders.head(50)[FEATURES].astype(float), raw_score=True)
    base = det.model.booster_.predict(orders.head(1)[FEATURES].astype(float), pred_contrib=True)[0, -1]
    assert np.allclose(shap.sum(axis=1) + base, raw, atol=1e-6)
