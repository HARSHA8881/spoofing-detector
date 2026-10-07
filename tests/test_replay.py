from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.load import ADD, CANCEL, DELETE, EMPTY_ASK, EMPTY_BID, EXEC, EXEC_HIDDEN
from src.load import find_stock_days, load_messages, load_orderbook
from src.replay import (OrderBook, book_from_snapshot, check_against_official, match_rate,
                        overlay_book, replay)
from tests.conftest import make_messages, synthetic_day


def test_add_builds_levels_and_best_prices():
    book = OrderBook()
    book.add(1, 1, 1000, 50)
    book.add(2, 1, 1000, 30)
    book.add(3, 1, 900, 10)
    book.add(4, -1, 1100, 70)
    assert book.best_bid() == 1000 and book.best_ask() == 1100
    assert book.top(1, 5) == [(1000, 80), (900, 10)]
    assert book.top(-1, 5) == [(1100, 70)]


def test_empty_book_uses_lobster_dummy_prices():
    book = OrderBook()
    assert book.best_bid() == EMPTY_BID and book.best_ask() == EMPTY_ASK


def test_partial_cancel_execution_and_delete():
    book = OrderBook()
    book.apply(ADD, 1, 100, 1000, 1)
    book.apply(CANCEL, 1, 30, 1000, 1)
    assert book.top(1, 1) == [(1000, 70)]
    book.apply(EXEC, 1, 20, 1000, 1)
    assert book.orders[1][2] == 50
    # a delete removes whatever is left, whatever size the message carries
    book.apply(DELETE, 1, 999, 1000, 1)
    assert book.top(1, 1) == [] and 1 not in book.orders and book.anomalies == 0


def test_full_execution_removes_order_and_level():
    book = OrderBook()
    book.apply(ADD, 1, 100, 1000, -1)
    book.apply(EXEC, 1, 100, 1000, -1)
    assert not book.orders and not book.asks


def test_hidden_execution_leaves_book_alone():
    book = OrderBook()
    book.apply(ADD, 1, 100, 1000, -1)
    assert book.apply(EXEC_HIDDEN, 0, 500, 1000, -1) == 0
    assert book.top(-1, 1) == [(1000, 100)]


def test_unknown_order_reduces_the_seeded_level():
    book = book_from_snapshot(np.array([1100, 200, 1000, 300]))
    book.apply(EXEC, 42, 120, 1000, 1)       # order 42 was resting before the open
    assert book.top(1, 1) == [(1000, 180)]
    book.apply(DELETE, 43, 180, 1000, 1)
    assert book.top(1, 1) == [] and book.anomalies == 0
    book.apply(DELETE, 44, 10, 500, 1)       # a level we know nothing about
    assert book.anomalies == 1


def test_replay_output_has_lobster_layout():
    msgs = make_messages([(1.0, ADD, 1, 10, 1000, 1), (2.0, ADD, 2, 20, 1100, -1),
                          (3.0, ADD, 3, 5, 1200, -1), (4.0, DELETE, 2, 20, 1100, -1)])
    out = replay(msgs, depth=2)
    assert out.shape == (4, 8)
    assert out[0].tolist() == [EMPTY_ASK, 0, 1000, 10, EMPTY_ASK, 0, EMPTY_BID, 0]
    assert out[2].tolist() == [1100, 20, 1000, 10, 1200, 5, EMPTY_BID, 0]
    assert out[3].tolist() == [1200, 5, 1000, 10, EMPTY_ASK, 0, EMPTY_BID, 0]


def test_replay_matches_brute_force_book():
    msgs = synthetic_day(n=3000, seed=3)
    out = replay(msgs, depth=3)
    live = {}
    for i, m in enumerate(msgs.itertuples(index=False)):
        if m.event_type == ADD:
            live[m.order_id] = (m.direction, m.price, m.size)
        else:
            del live[m.order_id]
        for direction, offset in ((-1, 0), (1, 2)):
            levels = {}
            for d, price, size in live.values():
                if d == direction:
                    levels[price] = levels.get(price, 0) + size
            best = sorted(levels.items(), reverse=direction == 1)[:3]
            got = [(p, s) for p, s in zip(out[i, offset::4], out[i, offset + 1::4]) if s > 0]
            assert got == best
    assert (out[:, 0] > out[:, 2]).all()  # never crossed


def test_engine_agrees_with_its_own_replay(toy_day):
    # the toy stream holds events at every depth, so compare against the whole book
    msgs, _ = toy_day
    full = replay(msgs, depth=200)
    assert (full[:, -1] == 0).all()  # 200 levels really is the whole book
    assert check_against_official(msgs, full) == 1.0
    assert match_rate(replay(msgs, depth=10), full, levels=10) == 1.0


def test_overlay_adds_injected_orders_on_top_of_the_official_book(toy_day):
    msgs, book = toy_day
    row = 500
    t = msgs.time[row]
    ask, bid = int(book[row, 0]), int(book[row, 2])
    extra = make_messages([(t, ADD, 10**9, 777, ask, -1), (msgs.time[row + 50], DELETE, 10**9, 777, ask, -1)])
    merged = msgs.assign(orig_row=np.arange(len(msgs)), injected=False)
    extra = extra.assign(orig_row=[row, row + 50], injected=True)
    merged = (pd.concat([merged, extra]).sort_values(["orig_row", "injected", "time"], kind="stable")
              .reset_index(drop=True))
    out = overlay_book(merged, book, depth=10)
    add_at = merged.index[merged.order_id == 10**9][0]
    assert out[add_at, 0] == ask and out[add_at, 1] == book[row, 1] + 777
    assert out[add_at, 2] == bid
    # outside the order's life the book is exactly the original one
    original = ~merged.injected.to_numpy()
    untouched = original & ((merged.orig_row < row) | (merged.orig_row > row + 50)).to_numpy()
    assert (out[untouched] == book[merged.orig_row[untouched]]).all()
    during = original & ((merged.orig_row > row) & (merged.orig_row <= row + 50)).to_numpy()
    assert (out[during, 1::4].sum(axis=1) >= book[merged.orig_row[during], 1::4].sum(axis=1)).all()


@pytest.mark.skipif(not find_stock_days(Path("data")), reason="LOBSTER sample files not downloaded")
def test_engine_reproduces_official_lobster_book():
    msg_path, book_path = next(iter(find_stock_days(Path("data")).values()))
    msgs = load_messages(msg_path).iloc[:50_000]
    official = load_orderbook(book_path)[:50_000]
    assert check_against_official(msgs, official) > 0.999
