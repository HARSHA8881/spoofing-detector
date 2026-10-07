"""Turn a message stream and its row-aligned book into one feature row per order.

Everything is vectorised: O(n log n) for n messages (sorts and searchsorted),
O(n) memory. Orders with no "add" message (resting since before the open) are
dropped rather than guessed at.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.load import ADD, CANCEL, DELETE, EXEC, EXEC_HIDDEN, TICK

MEDIAN_WINDOW = 1000   # same-side adds used for the rolling median size
IMBALANCE_LEVELS = 5
WINDOW_S = 10.0
LARGE_RATIO = 5.0      # "large" and "fast" for n_large_cancels
FAST_S = 5.0

# Columns the models see. Scale-free on purpose, so a model trained on one
# stock can score another.
FEATURES = ["size_ratio", "lifetime_s", "fill_ratio", "dist_ticks", "mid_move_bps",
            "opp_exec_rel", "imb_jump_add", "imb_jump_end", "cancel_to_trade",
            "n_large_cancels"]


def order_lifecycles(msgs: pd.DataFrame) -> pd.DataFrame:
    """One row per visible order: entry, how long it lived, how it ended."""
    msgs = msgs.reset_index(drop=True)
    adds = msgs[msgs.event_type == ADD].drop_duplicates("order_id")
    orders = pd.DataFrame({
        "t_add": adds.time.to_numpy(), "size_add": adds["size"].to_numpy(),
        "price": adds.price.to_numpy(), "direction": adds.direction.to_numpy(),
        "add_row": adds.index.to_numpy(),
    }, index=pd.Index(adds.order_id.to_numpy(), name="order_id"))
    for col in ("role", "episode"):
        if col in msgs:
            orders[col] = adds[col].to_numpy()

    later = msgs[msgs.event_type.isin([CANCEL, DELETE, EXEC])]
    later = later[later.index > later.order_id.map(orders.add_row).fillna(np.inf)]
    filled = later[later.event_type == EXEC].groupby("order_id")["size"].sum()
    cancelled = later[later.event_type != EXEC].groupby("order_id")["size"].sum()
    deleted = later[later.event_type == DELETE].groupby("order_id").size()
    last_row = pd.Series(later.index, index=later.order_id).groupby(level=0).max()

    orders["filled_qty"] = filled.reindex(orders.index).fillna(0).astype("int64")
    orders["cancelled_qty"] = cancelled.reindex(orders.index).fillna(0).astype("int64")
    was_deleted = deleted.reindex(orders.index).notna()
    done = was_deleted | (orders.filled_qty + orders.cancelled_qty >= orders.size_add)
    orders["status"] = np.select([was_deleted, done], ["deleted", "filled"], "open")
    orders["end_row"] = last_row.reindex(orders.index).where(done)  # NaN = still open
    t_end = pd.Series(msgs.time.to_numpy()[orders.end_row.fillna(0).astype(int)],
                      index=orders.index).where(done)
    orders["t_end"] = t_end
    orders["lifetime_s"] = t_end - orders.t_add
    orders["fill_ratio"] = orders.filled_qty / orders.size_add
    return orders


def _imbalance(book: np.ndarray) -> np.ndarray:
    ask = book[:, 1: 4 * IMBALANCE_LEVELS: 4].sum(axis=1)
    bid = book[:, 3: 4 * IMBALANCE_LEVELS: 4].sum(axis=1)
    return (bid - ask) / np.maximum(bid + ask, 1)


def _cum(values: np.ndarray) -> np.ndarray:
    """Cumulative sum with a leading 0, so cum[b + 1] - cum[a + 1] covers rows (a, b]."""
    return np.concatenate([[0], np.cumsum(values)])


def build_features(msgs: pd.DataFrame, book: np.ndarray) -> pd.DataFrame:
    """Per-order features. ``book`` row i is the book right after message i."""
    msgs = msgs.reset_index(drop=True)
    orders = order_lifecycles(msgs)
    time, types = msgs.time.to_numpy(), msgs.event_type.to_numpy()
    size, side = msgs["size"].to_numpy(), msgs.direction.to_numpy()
    direction = orders.direction.to_numpy()
    add_row = orders.add_row.to_numpy()
    ended = orders.end_row.notna().to_numpy()
    end_row = orders.end_row.fillna(len(msgs) - 1).to_numpy().astype(int)

    # size relative to what is normal on that side right now (past orders only)
    median = orders.groupby("direction")["size_add"].transform(
        lambda s: s.shift(1).rolling(MEDIAN_WINDOW, min_periods=1).median())
    median = median.fillna(orders.size_add)
    orders["size_ratio"] = orders.size_add / median

    # where the order sat, using the book right after it arrived
    ask, bid = book[:, 0], book[:, 2]
    best = np.where(direction == 1, bid[add_row], ask[add_row])
    orders["dist_ticks"] = direction * (best - orders.price.to_numpy()) / TICK

    # price move over the order's life, positive = the way its pressure pushes
    mid = (ask + bid) / 2
    before_end = np.maximum(end_row - 1, add_row)
    orders["mid_move_bps"] = direction * (mid[before_end] - mid[add_row]) / mid[add_row] * 1e4

    # trades against the opposite side while the order rested
    is_exec = np.isin(types, (EXEC, EXEC_HIDDEN))
    exec_buy, exec_sell = _cum(size * (is_exec & (side == 1))), _cum(size * (is_exec & (side == -1)))
    opp = np.where(direction == 1, exec_sell[end_row + 1] - exec_sell[add_row + 1],
                   exec_buy[end_row + 1] - exec_buy[add_row + 1])
    orders["opp_exec_qty"] = opp
    orders["opp_exec_rel"] = opp / median.to_numpy()

    # book imbalance switching on at the add and off at the end
    imb = _imbalance(book)
    orders["imb_jump_add"] = direction * (imb[add_row] - imb[np.maximum(add_row - 1, 0)])
    orders["imb_jump_end"] = np.where(ended, direction * (imb[end_row] - imb[end_row - 1]), np.nan)

    # 10-second window ending when the order ended
    win_start = np.searchsorted(time, time[end_row] - WINDOW_S, side="left")
    cancels, execs = _cum(size * np.isin(types, (CANCEL, DELETE))), _cum(size * is_exec)
    cancelled = cancels[end_row + 1] - cancels[win_start]
    executed = execs[end_row + 1] - execs[win_start]
    orders["cancel_to_trade"] = np.where(ended, cancelled / np.maximum(executed, 1), np.nan)

    fast_large = ((orders.size_ratio >= LARGE_RATIO) & (orders.status == "deleted")
                  & (orders.filled_qty == 0) & (orders.lifetime_s < FAST_S)).to_numpy()
    ends = np.sort(orders.t_end.to_numpy()[fast_large])
    t_end = orders.t_end.to_numpy()
    count = np.searchsorted(ends, t_end, "right") - np.searchsorted(ends, t_end - WINDOW_S, "left")
    orders["n_large_cancels"] = np.where(ended, count - fast_large, np.nan)

    if "role" in orders:
        orders["label"] = (orders.role == "spoof").astype("int8")
    return orders


def level_jump_features(time: np.ndarray, book: np.ndarray, depth: int = 5,
                        step_s: float = 0.5, history: int = 240) -> pd.DataFrame:
    """Level-based signal for feeds without order ids (e.g. Kite 5-level depth).

    The book is sampled every ``step_s`` seconds. For each side and depth rank,
    ``level_jump_z`` is the size change since the last snapshot in units of that
    rank's recent standard deviation, and ``revert_s`` is how long the extra
    size stayed before at least 75% of it was gone (NaN = it stayed).
    """
    grid = np.arange(time[0], time[-1], step_s)
    rows = np.searchsorted(time, grid, side="right") - 1
    horizon = int(round(10 / step_s))
    out = []
    for direction, offset in ((-1, 1), (1, 3)):
        for lvl in range(depth):
            s = pd.Series(book[rows, 4 * lvl + offset].astype(float))
            jump = s.diff()
            sd = jump.shift(1).rolling(history, min_periods=30).std()
            z = (jump / sd.replace(0, np.nan)).to_numpy()
            gone_below = (s.shift(1) + 0.25 * jump).to_numpy()
            revert = np.full(len(s), np.nan)
            values = s.to_numpy()
            for i in np.flatnonzero(z >= 3):
                back = np.flatnonzero(values[i + 1: i + 1 + horizon] <= gone_below[i])
                if len(back):
                    revert[i] = (back[0] + 1) * step_s
            out.append(pd.DataFrame({"time": grid, "direction": direction, "level": lvl + 1,
                                     "size": values, "level_jump_z": z, "revert_s": revert}))
    return pd.concat(out, ignore_index=True)
