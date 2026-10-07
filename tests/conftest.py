import numpy as np
import pandas as pd
import pytest

from src.load import ADD, DELETE, EXEC, MSG_COLS, TICK
from src.replay import OrderBook, replay


def make_messages(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=MSG_COLS).astype(
        {"event_type": "int8", "order_id": "int64", "size": "int64", "price": "int64", "direction": "int8"})


def synthetic_day(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """A random but valid full-information message stream (no pre-open orders)."""
    rng = np.random.default_rng(seed)
    book, rows, live = OrderBook(), [], []
    mid, t, next_id = 1_000_000, 34_200.0, 1
    for _ in range(n):
        t += rng.exponential(0.2)
        mid += int(rng.choice([-TICK, 0, 0, 0, TICK])) if rng.random() < 0.05 else 0
        action = rng.random()
        if action < 0.55 or len(live) < 40:
            direction = int(rng.choice([-1, 1]))
            price = mid - direction * TICK * int(rng.integers(1, 8))
            # never cross the opposite side
            if direction == 1 and book.asks and price >= book.best_ask():
                price = book.best_ask() - TICK
            if direction == -1 and book.bids and price <= book.best_bid():
                price = book.best_bid() + TICK
            size = int(rng.choice([50, 100, 100, 100, 200, 300]))
            row = (t, ADD, next_id, size, price, direction)
            live.append(next_id)
            next_id += 1
        elif action < 0.9:
            order_id = live.pop(int(rng.integers(len(live))))
            direction, price, size = book.orders[order_id]
            row = (t, DELETE, order_id, size, price, direction)
        else:  # a market order hits the best level on one side
            direction = int(rng.choice([-1, 1]))
            best = book.best_bid() if direction == 1 else book.best_ask()
            at_best = [o for o in live if book.orders[o][:2] == [direction, best]]
            if not at_best:
                continue
            order_id = at_best[0]
            size = book.orders[order_id][2]
            live.remove(order_id)
            row = (t, EXEC, order_id, size, best, direction)
        book.apply(*row[1:])
        rows.append(row)
    return make_messages(rows)


@pytest.fixture(scope="session")
def toy_day():
    msgs = synthetic_day()
    return msgs, replay(msgs, depth=10)
