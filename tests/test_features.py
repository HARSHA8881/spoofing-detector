import numpy as np
import pytest

from src.features import build_features, level_jump_features, order_lifecycles
from src.load import ADD, CANCEL, DELETE, EXEC, TICK
from src.replay import replay
from tests.conftest import make_messages

P = 1_000_000  # $100.00


def story():
    """bid 100.00 / ask 100.10, then a big sell two ticks behind the ask that is pulled."""
    return make_messages([
        (10.0, ADD, 1, 100, P, 1),                   # resting bid
        (10.0, ADD, 2, 100, P + 10 * TICK, -1),      # resting ask
        (11.0, ADD, 3, 1000, P + 12 * TICK, -1),     # the big sell
        (11.5, ADD, 4, 50, P + TICK, 1),             # small buy inside the spread ...
        (11.6, EXEC, 4, 50, P + TICK, 1),            # ... which fills
        (12.0, CANCEL, 1, 40, P, 1),
        (13.0, DELETE, 3, 1000, P + 12 * TICK, -1),  # big sell pulled, unfilled
        (14.0, EXEC, 1, 60, P, 1),
        (15.0, DELETE, 99, 10, P - 5 * TICK, 1),     # resting since before the open
    ])


def test_lifecycles():
    orders = order_lifecycles(story())
    assert 99 not in orders.index  # no add message -> dropped, not guessed
    big = orders.loc[3]
    assert big.status == "deleted" and big.lifetime_s == pytest.approx(2.0)
    assert big.filled_qty == 0 and big.cancelled_qty == 1000 and big.fill_ratio == 0
    small = orders.loc[4]
    assert small.status == "filled" and small.fill_ratio == 1 and small.lifetime_s == pytest.approx(0.1)
    partly = orders.loc[1]  # 40 cancelled + 60 filled = gone without a delete message
    assert partly.status == "filled" and partly.fill_ratio == pytest.approx(0.6)
    assert orders.loc[2].status == "open" and np.isnan(orders.loc[2].lifetime_s)


def test_order_features():
    msgs = story()
    f = build_features(msgs, replay(msgs, depth=5))
    big = f.loc[3]
    assert big.size_ratio == pytest.approx(10.0)   # 1000 vs the one earlier sell of 100
    assert big.dist_ticks == pytest.approx(2.0)    # ask is 100.10, order at 100.12
    assert big.opp_exec_qty == 50                  # the buy that filled while it rested
    assert big.imb_jump_add > 0 and big.imb_jump_end < 0
    assert big.mid_move_bps == pytest.approx(0.0)
    assert f.loc[4].dist_ticks == 0                # it became the best bid
    assert f.loc[1].size_ratio == 1.0              # nothing earlier to compare with


def test_size_ratio_never_looks_ahead():
    msgs = story()
    later = msgs.copy()
    later.loc[later.order_id == 3, "size"] = 5000
    a = build_features(msgs, replay(msgs, depth=5))
    b = build_features(later, replay(later, depth=5))
    assert a.loc[[1, 2], "size_ratio"].equals(b.loc[[1, 2], "size_ratio"])


def test_window_features_count_fast_large_cancels():
    rows = [(float(i), ADD, i, 100, P + 10 * TICK, -1) for i in range(1, 30)]
    rows += [(40.0, ADD, 100, 1000, P + 11 * TICK, -1), (41.0, DELETE, 100, 1000, P + 11 * TICK, -1),
             (42.0, ADD, 101, 1000, P + 11 * TICK, -1), (43.0, DELETE, 101, 1000, P + 11 * TICK, -1)]
    msgs = make_messages(rows)
    f = build_features(msgs, replay(msgs, depth=5))
    assert f.loc[100, "n_large_cancels"] == 0   # itself is not counted
    assert f.loc[101, "n_large_cancels"] == 1
    assert f.loc[101, "cancel_to_trade"] == 2000  # 2000 cancelled, nothing traded


def test_level_jump_flags_a_size_spike_that_reverts():
    time = np.arange(0, 300, 0.5)
    rng = np.random.default_rng(0)
    book = np.zeros((len(time), 20), dtype=np.int64)
    book[:, 1::2] = 1000 + rng.integers(-20, 20, size=(len(time), 10))
    book[400:404, 3] += 50_000  # bid level 1 jumps for two seconds
    lv = level_jump_features(time, book, depth=5)
    hit = lv[(lv.level_jump_z >= 4) & lv.revert_s.notna()]
    assert len(hit) == 1
    assert hit.iloc[0].direction == 1 and hit.iloc[0].level == 1 and hit.iloc[0].revert_s == 2.0
