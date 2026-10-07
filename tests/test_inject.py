import numpy as np
import pytest

from src.features import build_features
from src.inject import InjectionConfig, inject
from src.load import ADD, DELETE, EXEC
from src.replay import overlay_book

CFG = InjectionConfig(n_episodes=(8, 8), n_hard_neg=(10, 10), seed=5)


@pytest.fixture(scope="module")
def injected(toy_day):
    msgs, book = toy_day
    out = inject(msgs, book, CFG)
    return msgs, book, out, overlay_book(out, book, depth=10)


def test_original_messages_are_kept_in_order(injected):
    msgs, _, out, _ = injected
    orig = out[~out.injected]
    assert (orig[msgs.columns].to_numpy() == msgs.to_numpy()).all()
    assert (np.diff(out.time.to_numpy()) >= 0).all()


def test_labels_and_roles(injected):
    _, _, out, _ = injected
    assert set(out.role) == {"orig", "spoof", "genuine", "hard_neg"}
    assert (out.label == out.role.isin(["spoof", "genuine"])).all()
    assert (out.loc[~out.injected, "label"] == 0).all()
    assert (out.loc[out.injected, "episode"] >= 0).all()


def test_big_orders_are_added_then_deleted_and_never_trade(injected):
    _, _, out, _ = injected
    big = out[out.role.isin(["spoof", "hard_neg"])]
    for _, events in big.groupby("order_id"):
        assert events.event_type.tolist() == [ADD, DELETE]
        assert events["size"].nunique() == 1
    assert not (big.event_type == EXEC).any()


def test_genuine_order_is_on_the_other_side_and_fills_during_the_spoof(injected):
    _, _, out, _ = injected
    for _, ep in out[out.label == 1].groupby("episode"):
        spoof, genuine = ep[ep.role == "spoof"], ep[ep.role == "genuine"]
        assert set(genuine.direction) == {-spoof.direction.iloc[0]}
        fills = genuine[genuine.event_type == EXEC]
        assert len(fills) >= 1
        assert fills.time.min() > spoof.time.min() and fills.time.max() < spoof.time.max()


def test_spoof_orders_are_large_and_away_from_the_best_price(injected):
    msgs, _, out, book = injected
    orders = build_features(out, book)
    spoof = orders[orders.role == "spoof"]
    median = msgs[msgs.event_type == ADD]["size"].median()
    assert (spoof.size_add >= 0.8 * min(CFG.k_choices) * median - 50).all()
    assert (spoof.dist_ticks >= 1).all()
    assert (spoof.status == "deleted").all() and (spoof.fill_ratio == 0).all()
    assert (spoof.label == 1).all() and orders.label.sum() == len(spoof)
    assert (spoof.opp_exec_qty > 0).all()


def test_hard_negatives_are_cancelled_after_the_price_moved_away(injected):
    _, _, out, book = injected
    orders = build_features(out, book)
    hard = orders[orders.role == "hard_neg"]
    assert len(hard) > 0 and (hard.label == 0).all()
    # the cancel comes a reaction time after the move, so a few moves have reversed by then
    assert (hard.mid_move_bps > 0).mean() >= 0.8


def test_rebuilt_book_stays_consistent(injected):
    _, _, _, book = injected
    assert (book[:, 0] > book[:, 2]).all()            # never crossed
    assert (book[:, 1::2] >= 0).all()                 # no negative sizes
    occupied = book[:, 5::4] > 0                      # ask levels 2+ that hold size
    assert (np.diff(book[:, 0::4], axis=1)[occupied] > 0).all()  # ask prices rise with level


def test_injection_is_deterministic_and_seed_dependent(toy_day):
    msgs, book = toy_day
    a, b = inject(msgs, book, CFG), inject(msgs, book, CFG)
    assert a.equals(b)
    other = inject(msgs, book, InjectionConfig(n_episodes=(8, 8), n_hard_neg=(10, 10), seed=6))
    assert not a[a.injected].time.reset_index(drop=True).equals(other[other.injected].time.reset_index(drop=True))
