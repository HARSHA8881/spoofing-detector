import numpy as np
import pandas as pd
import pytest

from src import evaluate
from src.detectors import (IsolationForestDetector, LightGBMDetector, LightGBMRanker,
                           RuleDetector)
from src.features import FEATURES, build_features
from src.inject import InjectionConfig, inject
from src.replay import overlay_book


@pytest.fixture(scope="module")
def orders(toy_day):
    msgs, book = toy_day
    out = inject(msgs, book, InjectionConfig(n_episodes=(40, 40), n_hard_neg=(30, 30), seed=1))
    orders = build_features(out, overlay_book(out, book, depth=10))
    orders.insert(0, "ticker", "TOY")
    return orders[orders.role != "genuine"]


def test_rule_flags_only_when_every_condition_holds():
    frame = pd.DataFrame({
        "size_pct": [0.999, 0.999, 0.5, 0.999, 0.999], "lifetime_s": [1, 60, 1, 1, 1],
        "depth_share_add": [0.5] * 5, "ep_cancel_together": [1, 0, 0, 0, 0],
        "status": ["deleted", "deleted", "deleted", "filled", "deleted"],
        "fill_ratio": [0, 0, 0, 1, 0], "opp_exec_qty": [100, 100, 100, 100, 0]})
    det = RuleDetector(min_size_pct=0.99, max_lifetime_s=5)
    assert det.flag(frame).tolist() == [True, False, False, False, False]
    assert det.score(frame).argmax() == 0
    assert det.explain(frame)[0].startswith("all 4 rules fired")


def test_rule_fit_picks_thresholds_from_the_grid(orders):
    det = RuleDetector().fit(orders)
    flagged = det.flag(orders)
    assert flagged.sum() > 0 and orders.label[flagged].mean() > orders.label.mean()


def test_every_detector_scores_and_explains(orders):
    for det in (RuleDetector(), IsolationForestDetector(), LightGBMDetector(), LightGBMRanker()):
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


def test_rule_score_ranks_inside_the_flagged_set(orders):
    det = RuleDetector().fit(orders)
    flagged = det.flag(orders)
    assert len(np.unique(det.score(orders)[flagged])) > 1  # not one tied block
    assert det.parts(orders).to_numpy().min() >= 0 and det.parts(orders).to_numpy().max() <= 1


def test_isolation_forest_only_ranks_candidates(orders):
    det = IsolationForestDetector().fit(orders)
    scores = det.score(orders)
    assert (scores[~orders.candidate.to_numpy()] == 0).all()
    assert (scores[orders.candidate.to_numpy()] > 0).all()


def test_lightgbm_can_be_trained_on_a_feature_subset(orders):
    det = LightGBMDetector(features=FEATURES[:5]).fit(orders)
    assert list(det.shap_values(orders.head(2)).columns) == FEATURES[:5]


def test_reviewer_labels_are_saved_and_change_the_model(orders, tmp_path):
    from src import feedback
    path = tmp_path / "labels.csv"
    det = LightGBMDetector().fit(orders)
    top = orders.iloc[evaluate.top_k(det.score(orders), 10)]
    for order_id in top.index:
        feedback.save_label("TOY", int(order_id), "spoof", path)
    labels = feedback.save_label("TOY", int(top.index[0]), "not spoof", path)  # changed my mind
    assert len(labels) == 10 and (labels.verdict == "not spoof").sum() == 1
    assert feedback.load_labels(path).equals(labels.reset_index(drop=True))
    new_det, scores = feedback.retrain(orders, orders, labels)
    assert scores.index.equals(orders.index)
    assert scores[top.index[0]] < det.score(orders.loc[[top.index[0]]])[0]  # the rejected alert drops
