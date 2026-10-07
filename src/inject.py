"""Insert labelled synthetic episodes into a real LOBSTER message stream.

Two kinds of episode are injected:

* spoof: one or more large orders 1-3 levels from the best price, a small order
  on the opposite side that executes while they rest, then a cancel of the
  large orders before anything trades against them.
* hard negative: an equally large order that is cancelled shortly after the
  price has moved away from it, with no opposite-side order of its own. This is
  what a market maker repricing a quote looks like.

History does not react to injected orders, so a spoof cannot actually move the
price. To stop "the price moved" from becoming a label leak (hard negatives
always have a move), ``react_share`` of the spoof episodes are anchored at
moments where the market happened to move the way the spoof would have pushed.

Every injected order is passive and leaves the book through its own messages,
so the original messages stay valid. The book is rebuilt afterwards with
``replay.overlay_book``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.load import ADD, DELETE, EMPTY_ASK, EMPTY_BID, EXEC, TICK

EDGE_S = 300.0     # keep episodes away from the open and the close
MIN_LIFE_S = 0.1   # drop an order whose safe life would be shorter than this


@dataclass(frozen=True)
class InjectionConfig:
    n_episodes: tuple[int, int] = (20, 50)      # spoof episodes per day
    k_choices: tuple[float, ...] = (5, 10, 20)  # order size = k x median size
    hold_s: tuple[float, float] = (0.5, 5.0)
    levels_away: tuple[int, int] = (1, 3)
    max_layers: int = 3                         # spoof orders resting at once
    max_repeats: int = 3                        # back-to-back rounds per episode
    react_share: float = 0.5
    # why a spoof is pulled: after a fixed hold, right after its own fill, or
    # when the best price starts moving toward it
    cancel_reasons: tuple[str, ...] = ("timer", "after_fill", "approach")
    n_hard_neg: tuple[int, int] = (60, 120)
    seed: int = 0


class _Injector:
    def __init__(self, msgs: pd.DataFrame, book: np.ndarray, cfg: InjectionConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.book = book
        self.time = msgs.time.to_numpy()
        self.ask, self.bid = book[:, 0], book[:, 2]
        self.mid = (self.ask + self.bid) / 2
        self.move_min = max(TICK / 2, float(np.median(self.ask - self.bid)) / 2)
        adds = msgs[msgs.event_type == ADD]
        self.median_size = adds.groupby("direction")["size"].median().to_dict()
        # the spoofer's own fills are drawn from the sizes that really traded
        execs = msgs[msgs.event_type == EXEC]
        self.trade_sizes = {d: execs.loc[execs.direction == d, "size"].to_numpy() for d in (-1, 1)}
        self.next_id = int(msgs.order_id.max()) + 1
        self.rows: list[tuple] = []

    # -- helpers -----------------------------------------------------------
    def row_at(self, t: float) -> int:
        """Last original row at or before time t."""
        return int(np.searchsorted(self.time, t, side="right")) - 1

    def emit(self, t, event_type, order_id, size, price, direction, role, episode, reason):
        self.rows.append((t, event_type, order_id, size, price, direction, role, episode, reason,
                          self.row_at(t)))

    def big_size(self, direction: int) -> int:
        k = self.rng.choice(self.cfg.k_choices)
        raw = k * self.median_size[direction] * self.rng.uniform(0.8, 1.2)
        return max(100, int(round(raw / 100)) * 100)

    def level_price(self, row: int, direction: int, level: int) -> int:
        price = int(self.book[row, 4 * level + (0 if direction == -1 else 2)])
        if price in (EMPTY_ASK, EMPTY_BID):
            best = self.ask[row] if direction == -1 else self.bid[row]
            price = int(best - direction * level * TICK)
        return price

    def first_move(self, row: int, t_end: float, direction: int) -> int | None:
        """First row before t_end where mid has moved away from a `direction` order."""
        seg = self.mid[row + 1: self.row_at(t_end) + 1]
        hit = direction * (seg - self.mid[row]) >= self.move_min
        return row + 1 + int(hit.argmax()) if hit.any() else None

    def safe_cancel(self, t_add: float, t_cancel: float, price: int, direction: int) -> float:
        """Pull the order just before the best price on its side reaches it."""
        lo, hi = self.row_at(t_add) + 1, self.row_at(t_cancel) + 1
        best = self.ask[lo:hi] if direction == -1 else self.bid[lo:hi]
        reached = direction * (price - best) >= 0
        if reached.any():
            t_cancel = min(t_cancel, self.time[lo + int(reached.argmax())] - 1e-6)
        return t_cancel

    def anchor(self, need_move: bool) -> tuple[float, int, int | None]:
        """Pick a start time and side; with need_move, one followed by a price move."""
        lo, hi = self.time[0] + EDGE_S, self.time[-1] - EDGE_S
        for _ in range(5000):
            t0 = self.rng.uniform(lo, hi)
            direction = int(self.rng.choice([-1, 1]))
            if not need_move:
                return t0, direction, None
            window = self.rng.uniform(*self.cfg.hold_s)
            for d in (direction, -direction):
                move_row = self.first_move(self.row_at(t0), t0 + window, d)
                if move_row is not None:
                    return t0, d, move_row
        return t0, direction, None

    def levels(self, n: int) -> list[int]:
        lo, hi = self.cfg.levels_away
        choices = np.arange(lo, hi + 1)
        return self.rng.choice(choices, size=min(n, len(choices)), replace=False).tolist()

    def first_approach(self, t_start: float, t_end: float, direction: int) -> float | None:
        """First time the best price on the order's side moves toward the order."""
        row = self.row_at(t_start)
        best = self.ask if direction == -1 else self.bid
        seg = best[row + 1: self.row_at(t_end) + 1]
        hit = direction * (seg - best[row]) < 0
        return float(self.time[row + 1 + int(hit.argmax())]) if hit.any() else None

    def big_orders(self, t_start, t_cancel, direction, levels, role, episode, reason):
        """Add large orders and cancel them together. Returns (cancel time, reason)."""
        orders = []
        for i, level in enumerate(levels):
            t_add = t_start + i * self.rng.uniform(0.005, 0.05)
            price = self.level_price(self.row_at(t_add), direction, level)
            orders.append((t_add, price, self.big_size(direction)))
            safe = self.safe_cancel(t_add, t_cancel, price, direction)
            if safe < t_cancel and role == "spoof":
                reason = "approach"  # pulled because the market was about to reach it
            t_cancel = safe
        if t_cancel - orders[-1][0] < MIN_LIFE_S:
            return None
        for i, (t_add, price, size) in enumerate(orders):
            order_id, self.next_id = self.next_id, self.next_id + 1
            self.emit(t_add, ADD, order_id, size, price, direction, role, episode, reason)
            self.emit(t_cancel - 0.002 * i, DELETE, order_id, size, price, direction, role,
                      episode, reason)
        return t_cancel, reason

    # -- episodes ----------------------------------------------------------
    def spoof(self, episode: int) -> None:
        cfg, rng = self.cfg, self.rng
        t0, direction, _ = self.anchor(need_move=rng.random() < cfg.react_share)
        for _ in range(rng.integers(1, cfg.max_repeats + 1)):
            hold = rng.uniform(*cfg.hold_s)
            levels = self.levels(rng.integers(1, cfg.max_layers + 1))
            reason = str(rng.choice(cfg.cancel_reasons))
            t_cancel, t_fill = t0 + hold, None
            if reason == "after_fill":
                t_fill = t0 + rng.uniform(0.3, 0.8) * hold
                t_cancel = t_fill + rng.uniform(0.02, 0.3)
            elif reason == "approach":
                t_near = self.first_approach(t0 + 0.1, t0 + hold, direction)
                if t_near is None:
                    reason = "timer"  # the market never came closer, so the hold just ran out
                else:
                    t_cancel = t_near + rng.uniform(0.01, 0.1)
            placed = self.big_orders(t0, t_cancel, direction, levels, "spoof", episode, reason)
            if placed is not None:
                t_end, reason = placed
                if t_fill is None or t_fill > t_end - 0.02:
                    t_fill = t0 + rng.uniform(0.3, 0.8) * (t_end - t0)
                self.genuine_trade(t_fill, -direction, episode, reason)
            t0 += hold + rng.uniform(0.5, 3.0)

    def genuine_trade(self, t: float, direction: int, episode: int, reason: str) -> None:
        """The order the spoofer wants filled: small, at or inside the best price."""
        row = self.row_at(t)
        best = self.bid[row] if direction == 1 else self.ask[row]
        inside = self.ask[row] - self.bid[row] > TICK
        price = int(best + direction * TICK) if inside else int(best)
        size = int(self.rng.choice(self.trade_sizes[direction]))
        order_id, self.next_id = self.next_id, self.next_id + 1
        self.emit(t, ADD, order_id, size, price, direction, "genuine", episode, reason)
        self.emit(t + 1e-4, EXEC, order_id, size, price, direction, "genuine", episode, reason)

    def hard_negative(self, episode: int) -> None:
        t0, direction, move_row = self.anchor(need_move=True)
        if move_row is None:
            return
        t_cancel = self.time[move_row] + self.rng.uniform(0.05, 0.5)
        self.big_orders(t0, t_cancel, direction, self.levels(1), "hard_neg", episode, "moved_away")


def inject(msgs: pd.DataFrame, book: np.ndarray, cfg: InjectionConfig) -> pd.DataFrame:
    """Return the message stream with injected episodes merged in, time-ordered.

    Extra columns: ``role`` (orig / spoof / genuine / hard_neg), ``label`` (1 on
    every row of a spoof episode), ``episode`` (-1 on original rows), ``reason``
    (why an injected order was cancelled), ``injected`` and ``orig_row`` (the
    original row whose book is in force).
    """
    inj = _Injector(msgs, book, cfg)
    n_spoof = int(inj.rng.integers(cfg.n_episodes[0], cfg.n_episodes[1] + 1))
    n_hard = int(inj.rng.integers(cfg.n_hard_neg[0], cfg.n_hard_neg[1] + 1))
    for episode in range(n_spoof):
        inj.spoof(episode)
    for episode in range(n_spoof, n_spoof + n_hard):
        inj.hard_negative(episode)

    orig = msgs.assign(role="orig", episode=-1, reason="", orig_row=np.arange(len(msgs)), injected=False)
    new = pd.DataFrame(inj.rows, columns=[*msgs.columns, "role", "episode", "reason", "orig_row"])
    new["injected"] = True
    out = pd.concat([orig, new.astype(orig.dtypes.to_dict())], ignore_index=True)
    out = out.sort_values(["orig_row", "injected", "time"], kind="stable", ignore_index=True)
    out["label"] = out.role.isin(["spoof", "genuine"]).astype("int8")
    return out
