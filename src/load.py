"""Read LOBSTER message and orderbook files.

Prices stay as integers (dollars x 10,000) everywhere inside the pipeline:
integer prices are exact dictionary keys for the book, floats are not.
Convert with ``to_dollars`` only for display.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]
PRICE_SCALE = 10_000
TICK = 100  # one cent, in LOBSTER price units

# LOBSTER event types
ADD, CANCEL, DELETE, EXEC, EXEC_HIDDEN, CROSS, HALT = 1, 2, 3, 4, 5, 6, 7

# LOBSTER fills unoccupied levels with these dummy prices
EMPTY_ASK = 9_999_999_999
EMPTY_BID = -9_999_999_999


def to_dollars(price):
    return price / PRICE_SCALE


def load_messages(path: str | Path) -> pd.DataFrame:
    """Load a LOBSTER message file. Row i lines up with orderbook row i."""
    msgs = pd.read_csv(path, header=None, names=MSG_COLS, usecols=range(6))
    return msgs.astype({"event_type": "int8", "order_id": "int64", "size": "int64",
                        "price": "int64", "direction": "int8"})


def load_orderbook(path: str | Path, depth: int | None = None) -> np.ndarray:
    """Load a LOBSTER orderbook file as an int64 array of shape (n, 4 * levels).

    Column layout per level: ask price, ask size, bid price, bid size.
    """
    book = pd.read_csv(path, header=None, dtype="int64").to_numpy()
    if depth is not None:
        book = book[:, : 4 * depth]
    return book


def find_stock_days(data_dir: str | Path) -> dict[str, tuple[Path, Path]]:
    """Map ticker -> (message file, orderbook file) for every sample in data_dir."""
    days = {}
    for msg_path in sorted(Path(data_dir).glob("*_message_*.csv")):
        book_path = Path(str(msg_path).replace("_message_", "_orderbook_"))
        if book_path.exists():
            days[msg_path.name.split("_")[0]] = (msg_path, book_path)
    return days


def book_columns(depth: int) -> list[str]:
    cols = []
    for lvl in range(1, depth + 1):
        cols += [f"ask_p{lvl}", f"ask_s{lvl}", f"bid_p{lvl}", f"bid_s{lvl}"]
    return cols
