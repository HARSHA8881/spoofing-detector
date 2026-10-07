"""Turn a message stream and its row-aligned book into one feature row per order.

Everything is vectorised: O(n log n) for n messages (sorts and searchsorted),
O(n) memory. Orders with no "add" message (resting since before the open) are
dropped rather than guessed at.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.load import ADD, CANCEL, DELETE, EXEC, EXEC_HIDDEN, TICK
from src.replay import size_at

MEDIAN_WINDOW = 1000   # same-side adds used for the rolling median size
IMBALANCE_LEVELS = 5
WINDOW_S = 10.0
LARGE_RATIO = 5.0      # "large" and "fast" for n_large_cancels
FAST_S = 5.0
CANDIDATE_PCT = 0.97   # large (for this stock-day) unfilled cancels, grouped into episodes
CANDIDATE_LIFE_S = 30.0
LINK_S = 3.0           # same-side candidates closer than this belong to one episode
TOGETHER_S = 0.05
REPEAT_WINDOW_S = 300.0

# What the models see, by the question each group answers. All scale-free, so a
# model trained on one stock can score another.
FEATURE_GROUPS = {
    "size": ["size_ratio", "size_pct", "level_share_add", "depth_share_add",
             "imb_jump_add", "imb_jump_end"],
    "life and fill": ["lifetime_s", "lifetime_pct", "fill_ratio", "n_partial_cancels"],
    "placement": ["dist_ticks", "ahead_add_rel"],
    "cancel context": ["dist_change_ticks", "better_depth_end_rel", "t_since_trade_side",
                       "t_since_trade_level", "mid_move_bps"],
    "opposite trades": ["opp_exec_rel"],
    "activity window": ["cancel_to_trade", "n_large_cancels"],
    "episode": ["ep_n_orders", "ep_n_levels", "ep_cancel_together", "ep_repeats"],
    "stock context": ["ctx_spread_ticks", "ctx_queue_rel"],
}
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
# Features that need the whole day (percentile ranks, day averages, episode
# grouping). Everything else is known the moment the order ends.
NEEDS_FULL_DAY = ["size_pct", "lifetime_pct", "ep_n_orders", "ep_n_levels",
                  "ctx_spread_ticks", "ctx_queue_rel"]
REALTIME_FEATURES = [f for f in FEATURES if f not in NEEDS_FULL_DAY]


def order_lifecycles(msgs: pd.DataFrame) -> pd.DataFrame:
    """One row per visible order: entry, how long it lived, how it ended."""
    msgs = msgs.reset_index(drop=True)
    adds = msgs[msgs.event_type == ADD].drop_duplicates("order_id")
    orders = pd.DataFrame({
        "t_add": adds.time.to_numpy(), "size_add": adds["size"].to_numpy(),
        "price": adds.price.to_numpy(), "direction": adds.direction.to_numpy(),
        "add_row": adds.index.to_numpy(),
    }, index=pd.Index(adds.order_id.to_numpy(), name="order_id"))
    for col in ("role", "episode", "reason"):
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
    partial = later[later.event_type == CANCEL].groupby("order_id").size()
    orders["n_partial_cancels"] = partial.reindex(orders.index).fillna(0).astype("int64")
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

    _cancel_context(orders, msgs, book, median.to_numpy(), before_end, ended)
    _market_relative(orders, book, median.to_numpy())
    _episodes(orders)

    if "role" in orders:
        orders["label"] = (orders.role == "spoof").astype("int8")
    return orders


def _cancel_context(orders, msgs, book, median, before_end, ended) -> None:
    """What the market looked like just before the order ended."""
    direction, price = orders.direction.to_numpy(), orders.price.to_numpy()
    snap = book[before_end]

    # did the best price come toward the order (negative) or move away (positive)?
    best_end = np.where(direction == 1, snap[:, 2], snap[:, 0])
    dist_end = direction * (best_end - price) / TICK
    orders["dist_change_ticks"] = np.where(ended, dist_end - orders.dist_ticks, np.nan)

    # shares that would have to trade before the market reaches the order's price
    better_ask = (snap[:, 1::4] * (snap[:, 0::4] < price[:, None])).sum(axis=1)
    better_bid = (snap[:, 3::4] * (snap[:, 2::4] > price[:, None])).sum(axis=1)
    better = np.where(direction == 1, better_bid, better_ask)
    orders["better_depth_end_rel"] = np.where(ended, better / median, np.nan)

    # time since the last trade on the order's side, and at its own price
    execs = msgs[msgs.event_type.isin([EXEC, EXEC_HIDDEN])]
    ends = pd.DataFrame({"t_end": orders.t_end.to_numpy(), "direction": direction, "price": price,
                         "pos": np.arange(len(orders))}).dropna().sort_values("t_end")
    last = execs[["time", "direction", "price"]].rename(columns={"time": "t_trade"}).sort_values("t_trade")
    for name, keys in (("t_since_trade_side", ["direction"]), ("t_since_trade_level", ["direction", "price"])):
        hit = pd.merge_asof(ends, last[["t_trade", *keys]], left_on="t_end", right_on="t_trade",
                            by=keys, allow_exact_matches=False)
        since = np.full(len(orders), np.nan)
        since[hit.pos.to_numpy()] = (hit.t_end - hit.t_trade).to_numpy()
        orders[name] = since


def _market_relative(orders, book, median) -> None:
    """Size judged against the book around the order, and against the stock's day."""
    direction, price = orders.direction.to_numpy(), orders.price.to_numpy()
    add_row, size = orders.add_row.to_numpy(), orders.size_add.to_numpy()

    level = size_at(book, add_row, price, direction)  # queue at its price, itself included
    orders["level_share_add"] = np.where(level > 0, size / np.maximum(level, size), np.nan)
    orders["ahead_add_rel"] = np.where(level > 0, np.maximum(level - size, 0) / median, np.nan)
    snap = book[add_row]
    depth = np.where(direction == 1, snap[:, 3: 4 * IMBALANCE_LEVELS: 4].sum(axis=1),
                     snap[:, 1: 4 * IMBALANCE_LEVELS: 4].sum(axis=1))
    orders["depth_share_add"] = size / np.maximum(depth, size)

    # percentile ranks inside this stock-day: "big for MSFT" and "big for AMZN" line up
    orders["size_pct"] = orders.groupby("direction")["size_add"].rank(pct=True)
    orders["lifetime_pct"] = orders.lifetime_s.rank(pct=True)

    # what kind of stock this is, the same value on every row of the day
    spread = (book[:, 0] - book[:, 2]) / TICK
    orders["ctx_spread_ticks"] = float(np.mean(spread))
    orders["ctx_queue_rel"] = float(np.mean(book[:, 1] + book[:, 3]) / 2 / np.median(size))


def _episodes(orders) -> None:
    """Group large unfilled cancels on one side into candidate episodes.

    An analyst reviews an incident, not a row: layered orders and quick repeats
    belong together. ``alert_group`` is the episode id for candidates and a
    unique id for every other order.
    """
    cand = ((orders.status == "deleted") & (orders.filled_qty == 0)
            & (orders.size_pct >= CANDIDATE_PCT) & (orders.lifetime_s < CANDIDATE_LIFE_S)).to_numpy()
    n = len(orders)
    group = -np.arange(1, n + 1)  # non-candidates stand alone
    n_orders, n_levels, together, repeats = (np.zeros(n) for _ in range(4))
    t_add, t_end = orders.t_add.to_numpy(), orders.t_end.to_numpy()
    price, direction = orders.price.to_numpy(), orders.direction.to_numpy()
    next_id = 0
    for side in (-1, 1):
        idx = np.flatnonzero(cand & (direction == side))
        if not len(idx):
            continue
        idx = idx[np.argsort(t_add[idx], kind="stable")]
        prev_end = np.concatenate([[-np.inf], np.maximum.accumulate(t_end[idx])[:-1]])
        cluster = np.cumsum(t_add[idx] > prev_end + LINK_S) - 1 + next_id
        next_id = cluster[-1] + 1
        group[idx] = cluster
        frame = pd.DataFrame({"cluster": cluster, "price": price[idx]})
        n_orders[idx] = frame.groupby("cluster")["price"].transform("size").to_numpy()
        n_levels[idx] = frame.groupby("cluster")["price"].transform("nunique").to_numpy()
        ends = np.sort(t_end[idx])
        together[idx] = (np.searchsorted(ends, t_end[idx] + TOGETHER_S, "right")
                         - np.searchsorted(ends, t_end[idx] - TOGETHER_S, "left") - 1)
        repeats[idx] = (np.searchsorted(ends, t_add[idx], "left")
                        - np.searchsorted(ends, t_add[idx] - REPEAT_WINDOW_S, "left"))
    orders["alert_group"] = group
    orders["candidate"] = cand
    orders["ep_n_orders"], orders["ep_n_levels"] = n_orders, n_levels
    orders["ep_cancel_together"], orders["ep_repeats"] = together, repeats


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
