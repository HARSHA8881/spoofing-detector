"""Rebuild the limit order book event by event from a LOBSTER message stream.

Data structures
---------------
* ``orders``: hash map order_id -> [direction, price, remaining]   O(1) lookup
* ``bids`` / ``asks``: sorted maps price -> total visible size      O(log L) update

With n messages and at most L live price levels the replay costs O(n log L)
time and O(live orders + L) memory. Writing the top ``depth`` levels after
each event adds O(depth), but only for the side the event touched.

Why a LOBSTER sample cannot be replayed from messages alone
-----------------------------------------------------------
A LOBSTER file requested with N levels only contains the events that touched
the top N levels. Orders resting at the open have no "add" message, and an
order that drifts deeper than level N can be deleted without the file ever
saying so. A pure message replay therefore keeps phantom liquidity. Two
functions deal with that honestly:

* ``check_against_official`` proves the engine on real data: for every message
  it compares the size change the engine predicts at that price with the change
  in the official orderbook file.
* ``overlay_book`` rebuilds the book after injection: the official book plus a
  replay of the injected orders only. Injected orders are passive and are
  removed by their own messages, so the two books simply add up.
"""

from __future__ import annotations

from itertools import islice

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

from src.load import ADD, CANCEL, DELETE, EMPTY_ASK, EMPTY_BID, EXEC


class OrderBook:
    def __init__(self):
        self.orders: dict[int, list[int]] = {}
        self.bids = SortedDict()  # keyed by -price so the best bid comes first
        self.asks = SortedDict()
        self.anomalies = 0  # reductions that could not be applied in full

    # -- level bookkeeping -------------------------------------------------
    def _side(self, direction: int):
        return (self.bids, -1) if direction == 1 else (self.asks, 1)

    def seed_level(self, direction: int, price: int, size: int) -> None:
        """Add anonymous resting liquidity (orders with no add message)."""
        if size <= 0:
            return
        levels, sign = self._side(direction)
        key = sign * price
        levels[key] = levels.get(key, 0) + size

    def _reduce_level(self, direction: int, price: int, size: int) -> None:
        levels, sign = self._side(direction)
        key = sign * price
        have = levels.get(key)
        if have is None:
            self.anomalies += 1
            return
        if size >= have:
            if size > have:
                self.anomalies += 1
            del levels[key]
        else:
            levels[key] = have - size

    # -- events ------------------------------------------------------------
    def add(self, order_id: int, direction: int, price: int, size: int) -> None:
        self.orders[order_id] = [direction, price, size]
        self.seed_level(direction, price, size)

    def reduce(self, order_id: int, direction: int, price: int, size: int,
               delete: bool = False) -> None:
        """Partial cancel, execution, or (delete=True) removal of what is left."""
        order = self.orders.get(order_id)
        if order is None:  # resting since before the open: only the level is known
            self._reduce_level(direction, price, size)
            return
        direction, price, remaining = order
        size = remaining if delete else min(size, remaining)
        self._reduce_level(direction, price, size)
        if size >= remaining:
            del self.orders[order_id]
        else:
            order[2] = remaining - size

    def apply(self, event_type: int, order_id: int, size: int, price: int,
              direction: int) -> int:
        """Apply one message. Returns the side touched (1, -1) or 0 for none."""
        if event_type == ADD:
            self.add(order_id, direction, price, size)
        elif event_type == CANCEL or event_type == EXEC:
            self.reduce(order_id, direction, price, size)
        elif event_type == DELETE:
            self.reduce(order_id, direction, price, size, delete=True)
        else:  # hidden executions, crosses and halts leave the visible book alone
            return 0
        return direction

    # -- queries -----------------------------------------------------------
    def best_bid(self) -> int:
        return -self.bids.peekitem(0)[0] if self.bids else EMPTY_BID

    def best_ask(self) -> int:
        return self.asks.peekitem(0)[0] if self.asks else EMPTY_ASK

    def top(self, direction: int, depth: int) -> list[tuple[int, int]]:
        """Best ``depth`` (price, size) levels of one side, best first."""
        levels, sign = self._side(direction)
        return [(sign * key, size) for key, size in islice(levels.items(), depth)]


def book_from_snapshot(row: np.ndarray) -> OrderBook:
    """Seed a book with anonymous liquidity from one LOBSTER orderbook row."""
    book = OrderBook()
    for ask_p, ask_s, bid_p, bid_s in np.asarray(row).reshape(-1, 4).tolist():
        if ask_p != EMPTY_ASK:
            book.seed_level(-1, ask_p, ask_s)
        if bid_p != EMPTY_BID:
            book.seed_level(1, bid_p, bid_s)
    return book


def replay(msgs: pd.DataFrame, book: OrderBook | None = None, depth: int = 10) -> np.ndarray:
    """Apply every message and record the top ``depth`` levels after each one.

    Returns an int64 array shaped like a LOBSTER orderbook file: one row per
    message, columns (ask price, ask size, bid price, bid size) per level.
    """
    book = OrderBook() if book is None else book
    n = len(msgs)
    out = np.empty((n, 4 * depth), dtype=np.int64)
    empty_ask = [EMPTY_ASK, 0] * depth
    empty_bid = [EMPTY_BID, 0] * depth

    def flat(direction: int) -> list[int]:
        row = [v for level in book.top(direction, depth) for v in level]
        pad = empty_bid if direction == 1 else empty_ask
        return row + pad[len(row):]

    ask_row, bid_row = flat(-1), flat(1)
    ask_cols = [c for lvl in range(depth) for c in (4 * lvl, 4 * lvl + 1)]
    bid_cols = [c for lvl in range(depth) for c in (4 * lvl + 2, 4 * lvl + 3)]
    asks_out = np.empty((n, 2 * depth), dtype=np.int64)
    bids_out = np.empty((n, 2 * depth), dtype=np.int64)

    events = zip(msgs.event_type.to_numpy().tolist(), msgs.order_id.to_numpy().tolist(),
                 msgs["size"].to_numpy().tolist(), msgs.price.to_numpy().tolist(),
                 msgs.direction.to_numpy().tolist())
    for i, (event_type, order_id, size, price, direction) in enumerate(events):
        touched = book.apply(event_type, order_id, size, price, direction)
        if touched == 1:
            bid_row = flat(1)
        elif touched == -1:
            ask_row = flat(-1)
        asks_out[i] = ask_row
        bids_out[i] = bid_row
    out[:, ask_cols] = asks_out
    out[:, bid_cols] = bids_out
    return out


def match_rate(replayed: np.ndarray, official: np.ndarray, levels: int) -> float:
    """Share of rows where the top ``levels`` levels equal the official book."""
    cols = 4 * levels
    return float((replayed[:, :cols] == official[:, :cols]).all(axis=1).mean())


def _size_at(book: np.ndarray, rows: np.ndarray, price: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Visible size at (direction, price) in the given orderbook rows, 0 if absent."""
    sub = book[rows]
    out = np.zeros(len(rows), dtype=np.int64)
    for offset, side in ((0, -1), (2, 1)):
        hit = (sub[:, offset::4] == price[:, None]) & (direction == side)[:, None]
        out += (sub[:, offset + 1::4] * hit).sum(axis=1)
    return out


def check_against_official(msgs: pd.DataFrame, official: np.ndarray) -> float:
    """Share of book-changing messages whose effect the engine predicts exactly.

    The engine tracks every order it saw being added; the predicted change in
    visible size at the message's price is compared with the official book
    (row i minus row i-1). Row 0 has no "before" row and is skipped.
    """
    book = OrderBook()
    types = msgs.event_type.to_numpy()
    predicted = np.zeros(len(msgs), dtype=np.int64)
    rows = zip(types.tolist(), msgs.order_id.to_numpy().tolist(), msgs["size"].to_numpy().tolist(),
               msgs.price.to_numpy().tolist(), msgs.direction.to_numpy().tolist())
    for i, (event_type, order_id, size, price, direction) in enumerate(rows):
        if event_type == ADD:
            predicted[i] = size
        elif event_type in (CANCEL, DELETE, EXEC):
            known = book.orders.get(order_id)
            predicted[i] = -(known[2] if known and event_type == DELETE else size)
        book.apply(event_type, order_id, size, price, direction)

    idx = np.flatnonzero(np.isin(types, (ADD, CANCEL, DELETE, EXEC)))
    idx = idx[idx > 0]
    price, direction = msgs.price.to_numpy()[idx], msgs.direction.to_numpy()[idx]
    actual = _size_at(official, idx, price, direction) - _size_at(official, idx - 1, price, direction)
    return float((actual == predicted[idx]).mean())


def overlay_book(msgs: pd.DataFrame, official: np.ndarray, depth: int = 10) -> np.ndarray:
    """Book after every message of an injected stream.

    ``msgs`` needs two columns written by the injector: ``orig_row`` (the row of
    the original file whose book is in force) and ``injected`` (bool).
    """
    out = official[msgs.orig_row.to_numpy(), : 4 * depth].copy()
    inj_rows = np.flatnonzero(msgs.injected.to_numpy())
    extra = OrderBook()
    cols = msgs[["event_type", "order_id", "size", "price", "direction"]].to_numpy()
    for k, r in enumerate(inj_rows):
        extra.apply(*cols[r].tolist())
        if not extra.orders:
            continue
        stop = inj_rows[k + 1] if k + 1 < len(inj_rows) else len(msgs)
        for direction, offset, empty in ((-1, 0, EMPTY_ASK), (1, 2, EMPTY_BID)):
            levels = extra.top(direction, depth)
            if not levels:
                continue
            for i in range(r, stop):
                merged = dict(zip(out[i, offset::4].tolist(), out[i, offset + 1::4].tolist()))
                merged.pop(empty, None)
                for price, size in levels:
                    merged[price] = merged.get(price, 0) + size
                best = sorted(merged.items(), reverse=direction == 1)[:depth]
                best += [(empty, 0)] * (depth - len(best))
                out[i, offset::4] = [p for p, _ in best]
                out[i, offset + 1::4] = [s for _, s in best]
    return out
