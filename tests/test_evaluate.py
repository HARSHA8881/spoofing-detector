import numpy as np
import pandas as pd
import pytest

from src import evaluate


def test_precision_and_recall_at_k():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    assert evaluate.precision_at_k(y, scores, 2) == 0.5
    assert evaluate.precision_at_k(y, scores, 3) == pytest.approx(2 / 3)
    assert evaluate.recall_at_k(y, scores, 3) == pytest.approx(2 / 3)
    assert evaluate.recall_at_k(y, scores, 6) == 1.0


def test_hard_negative_fpr_counts_legit_cancels_in_the_alert_list():
    hard = np.array([False, True, False, True])
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    assert evaluate.hard_negative_fpr(hard, scores, 2) == 0.5
    assert evaluate.hard_negative_fpr(hard, scores, 4) == 1.0


def test_evaluate_day_reports_rare_event_metrics():
    orders = pd.DataFrame({"label": [1, 0, 0, 0], "role": ["spoof", "hard_neg", "orig", "orig"]})
    perfect = evaluate.evaluate_day(orders, np.array([1.0, 0.5, 0.2, 0.1]), ks=(1, 2))
    assert perfect["pr_auc"] == 1.0 and perfect["precision@1"] == 1.0
    assert perfect["hard_neg_fpr@1"] == 0.0 and perfect["hard_neg_fpr@2"] == 1.0
    never = evaluate.evaluate_day(orders, np.zeros(4), ks=(1,))
    assert never["pr_auc"] == 0.25  # a detector that says nothing is only as good as the base rate


def test_level_alerts_match_spoof_orders_on_the_same_side():
    orders = pd.DataFrame({"label": [1, 1], "direction": [1, -1], "t_add": [10.0, 50.0], "t_end": [12.0, 52.0]})
    alerts = pd.DataFrame({"time": [11.0, 51.0, 80.0], "direction": [1, 1, -1]})
    out = evaluate.level_alert_overlap(alerts, orders)
    assert out["spoof_recall"] == 0.5 and out["alert_precision"] == pytest.approx(1 / 3)


def test_episode_metrics_count_each_group_once():
    orders = pd.DataFrame({
        "label":       [1, 1, 0, 0, 1, 0],
        "episode":     [7, 7, -1, -1, 8, -1],
        "alert_group": [0, 0, 0, -3, 1, -5]})
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.05])
    # top 2 groups are group 0 (three orders, one alert) and the lone order -3
    assert evaluate.top_episodes(orders, scores, 2).tolist() == [0, -3]
    precision, recall = evaluate.episode_metrics(orders, scores, 2)
    assert precision == 0.5 and recall == 0.5      # episode 7 found, episode 8 not
    assert evaluate.episode_metrics(orders, scores, 3) == (pytest.approx(2 / 3), 1.0)


def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    scores = y * 0.5 + rng.random(5000) * 2
    reps = evaluate.bootstrap(y, scores, ks=(50,), n_boot=200)
    point = evaluate.precision_at_k(y, scores, 50)
    low, high = np.percentile(reps["precision@50"], [2.5, 97.5])
    assert low <= point <= high and 0 < high - low < 0.5
    assert reps["pr_auc"].std() > 0
