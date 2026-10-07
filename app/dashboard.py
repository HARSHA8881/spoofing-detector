"""Streamlit alert viewer. Reads what `python -m src.run` wrote to artifacts/.

    uv run streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load import EMPTY_ASK, EMPTY_BID, PRICE_SCALE, TICK  # noqa: E402

ART = ROOT / "artifacts"
DAY = pd.Timestamp("2012-06-21")  # the LOBSTER sample date
EVENT_NAMES = {1: "added", 2: "partly cancelled", 3: "cancelled", 4: "executed"}
DETECTORS = {"lgbm": "LightGBM (supervised)", "rules": "Rule baseline", "iforest": "Isolation Forest"}
ROLE_NAMES = {"spoof": "Injected spoof", "hard_neg": "Hard negative", "orig": "Original order"}
ROLE_COLORS = {"spoof": "#2a78d6", "hard_neg": "#eb6834", "orig": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUES = [[0.0, "#cde2fb"], [0.5, "#5598e7"], [1.0, "#0d366b"]]
PUSH_UP, PUSH_DOWN = "#eb6834", "#2a78d6"

st.set_page_config(page_title="Spoofing surveillance", layout="wide")


def clock(seconds):
    return DAY + pd.to_timedelta(seconds, unit="s")


@st.cache_data
def load_orders(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(ART / "features" / f"{ticker}.parquet").set_index("order_id")


@st.cache_data
def load_alerts() -> pd.DataFrame:
    return pd.read_parquet(ART / "alerts.parquet")


@st.cache_data
def load_messages(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(ART / "messages" / f"{ticker}.parquet")


@st.cache_data
def load_book(ticker: str) -> np.ndarray:
    return pd.read_parquet(ART / "book" / f"{ticker}.parquet").to_numpy()


@st.cache_data
def mid_per_second(ticker: str) -> pd.DataFrame:
    time, book = load_messages(ticker).time.to_numpy(), load_book(ticker)
    grid = np.arange(np.ceil(time[0]), time[-1], 5.0)
    rows = np.searchsorted(time, grid, side="right") - 1
    return pd.DataFrame({"time": grid, "mid": (book[rows, 0] + book[rows, 2]) / 2 / PRICE_SCALE})


def style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(color=INK, size=13), hovermode="closest",
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#52514e")))
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7", tickfont=dict(color=MUTED), tickprefix="$")
    return fig


def timeline(ticker: str, alerts: pd.DataFrame, show_truth: bool, selected) -> go.Figure:
    mid = mid_per_second(ticker)
    fig = go.Figure(go.Scatter(x=clock(mid.time), y=mid.mid, mode="lines", name="Mid price",
                               line=dict(color=MUTED, width=1.5), hoverinfo="skip"))
    groups = alerts.groupby("role") if show_truth else [("alert", alerts)]
    for role, g in groups:
        fig.add_trace(go.Scatter(
            x=clock(g.t_add), y=g.price / PRICE_SCALE, mode="markers",
            name=ROLE_NAMES.get(role, "Alert"), customdata=np.stack([g.order_id, g["rank"]], axis=1),
            marker=dict(size=9, color=ROLE_COLORS.get(role, ROLE_COLORS["spoof"]),
                        line=dict(color=SURFACE, width=2)),
            hovertemplate="alert #%{customdata[1]}<br>%{x|%H:%M:%S}<br>order at $%{y:.2f}<extra></extra>"))
    if selected is not None:
        fig.add_trace(go.Scatter(x=[clock(selected.t_add)], y=[selected.price / PRICE_SCALE], mode="markers",
                                 name="Selected", hoverinfo="skip",
                                 marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=INK, width=2))))
    return style(fig, 340)


def book_heatmap(ticker: str, order: pd.Series, pad_s: float) -> go.Figure:
    msgs, book = load_messages(ticker), load_book(ticker)
    time = msgs.time.to_numpy()
    t_end = order.t_end if pd.notna(order.t_end) else order.t_add + pad_s
    grid = np.arange(order.t_add - pad_s, t_end + pad_s, max((t_end - order.t_add + 2 * pad_s) / 400, 0.01))
    rows = np.clip(np.searchsorted(time, grid, side="right") - 1, 0, len(time) - 1)
    snap = book[rows]
    prices, sizes = snap[:, 0::2], snap[:, 1::2].astype(float)
    real = (prices != EMPTY_ASK) & (prices != EMPTY_BID)
    lo = min(prices[real].min(), order.price) - TICK
    hi = max(prices[real].max(), order.price) + TICK
    step = TICK * max(1, int(np.ceil((hi - lo) / TICK / 150)))  # at most ~150 price rows
    levels = np.arange(lo, hi + step, step)
    z = np.zeros((len(levels), len(grid)))
    cols = np.repeat(np.arange(len(grid)), prices.shape[1]).reshape(prices.shape)
    np.add.at(z, (((prices[real] - lo) // step).astype(int), cols[real]), sizes[real])
    z[z == 0] = np.nan

    fig = go.Figure(go.Heatmap(
        x=clock(grid), y=levels / PRICE_SCALE, z=z, colorscale=BLUES, zmin=0,
        zmax=float(np.nanpercentile(z, 99)), colorbar=dict(title="Shares resting", thickness=12, len=0.8),
        hovertemplate="%{x|%H:%M:%S.%L}<br>$%{y:.2f}<br>%{z:,.0f} shares<extra></extra>"))
    mid = (snap[:, 0] + snap[:, 2]) / 2 / PRICE_SCALE
    fig.add_trace(go.Scatter(x=clock(grid), y=mid, mode="lines", name="Mid price",
                             line=dict(color=INK, width=2), hoverinfo="skip"))
    life = msgs[msgs.order_id == order.name]
    fig.add_trace(go.Scatter(
        x=clock(np.array([order.t_add, t_end])), y=[order.price / PRICE_SCALE] * 2, mode="lines+markers",
        name="This order", line=dict(color=PUSH_UP, width=2),
        marker=dict(size=11, color=PUSH_UP, line=dict(color=SURFACE, width=2)),
        text=["added", EVENT_NAMES.get(int(life.event_type.iloc[-1]), "still open")],
        hovertemplate="%{text} %{x|%H:%M:%S.%L}<extra></extra>"))
    other = msgs[(msgs.time >= grid[0]) & (msgs.time <= grid[-1]) & msgs.event_type.isin([4, 5])
                 & (msgs.direction == -order.direction)]
    if len(other):
        fig.add_trace(go.Scatter(
            x=clock(other.time), y=other.price / PRICE_SCALE, mode="markers", name="Trades on the opposite side",
            marker=dict(size=8, symbol="diamond", color="#1baf7a", line=dict(color=SURFACE, width=1)),
            customdata=other["size"], hovertemplate="%{customdata:,} shares traded at $%{y:.2f}<extra></extra>"))
    return style(fig, 420)


def life_story(ticker: str, order: pd.Series) -> tuple[str, pd.DataFrame]:
    msgs = load_messages(ticker)
    life = msgs[msgs.order_id == order.name]
    side = "buy" if order.direction == 1 else "sell"
    how = {"deleted": "cancelled", "filled": "fully executed", "open": "still resting at the close"}[order.status]
    lived = f" after {order.lifetime_s:.2f} seconds" if pd.notna(order.lifetime_s) else ""
    text = (f"A **{side}** order for **{order.size_add:,.0f} shares** at **${order.price / PRICE_SCALE:.2f}** "
            f"({order.size_ratio:.1f}x the usual size, {order.dist_ticks:.0f} ticks from the best price) was added at "
            f"{clock(order.t_add):%H:%M:%S.%f}. It was {how}{lived} with {order.fill_ratio:.0%} filled. "
            f"While it rested, {order.opp_exec_qty:,.0f} shares traded on the opposite side and the mid price moved "
            f"{order.mid_move_bps:+.2f} bps in the direction its pressure would push.")
    table = pd.DataFrame({"Time": clock(life.time).dt.strftime("%H:%M:%S.%f"),
                          "Event": life.event_type.map(EVENT_NAMES), "Shares": life["size"],
                          "Price": (life.price / PRICE_SCALE).map("${:.2f}".format)})
    return text, table


def shap_chart(detail: dict) -> go.Figure:
    s = pd.Series(detail).sort_values(key=abs)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker=dict(color=[PUSH_UP if v > 0 else PUSH_DOWN for v in s.values]),
                           hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
    fig = style(fig, 300)
    fig.update_yaxes(tickprefix="", tickfont=dict(color=INK))
    fig.update_xaxes(title="Push on the score (log-odds): right = more suspicious", zeroline=True,
                     zerolinecolor="#c3c2b7")
    return fig


# -- page ---------------------------------------------------------------------
if not (ART / "alerts.parquet").exists():
    st.error("No results yet. Run `uv run python -m src.run` first.")
    st.stop()

all_alerts = load_alerts()
with st.sidebar:
    st.header("Spoofing surveillance")
    ticker = st.selectbox("Stock-day", sorted(all_alerts.ticker.unique()), format_func=lambda t: f"{t} · 2012-06-21")
    detector = st.selectbox("Detector", list(DETECTORS), format_func=DETECTORS.get)
    k = st.slider("Alerts to review", 10, 200, 50, step=10)
    show_truth = st.toggle("Show injected ground truth", value=True,
                           help="Only possible because the spoofing here is synthetic. Real data has no labels.")
    st.caption("Alerts are patterns for a human to review, not proof of intent. "
               "Public data has no trader IDs.")

orders = load_orders(ticker)
alerts = all_alerts[(all_alerts.ticker == ticker) & (all_alerts.detector == detector) & (all_alerts["rank"] <= k)]
alerts = alerts.join(orders[["t_add", "price", "direction", "size_add", "size_ratio", "lifetime_s", "role"]],
                     on="order_id")

st.title(f"{ticker}: top {k} alerts from {DETECTORS[detector]}")
scored = orders[orders.role != "genuine"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders scored", f"{len(scored):,}")
c2.metric("Injected spoof orders", f"{int(scored.label.sum()):,}")
if show_truth:
    c3.metric(f"Precision@{k}", f"{(alerts.role == 'spoof').mean():.0%}",
              help="Share of these alerts that are injected spoof orders")
    c4.metric("Hard negatives flagged", f"{int((alerts.role == 'hard_neg').sum())} of {int((scored.role == 'hard_neg').sum())}",
              help="Legitimate-looking large cancels (price moved away first) that made the alert list")

key = f"selected_{ticker}_{detector}"
if st.session_state.get(key) not in set(alerts.order_id):
    st.session_state[key] = int(alerts.order_id.iloc[0])

st.subheader("Alerts on the day's price")
st.caption("Each dot is one alerted order at its own price. Click a dot, or a row below, to open it.")
selected_row = alerts.set_index("order_id").loc[st.session_state[key]]
picked = st.plotly_chart(timeline(ticker, alerts, show_truth, selected_row), on_select="rerun",
                         selection_mode="points", key=f"timeline_{key}")
points = [p for p in picked.selection.points if "customdata" in p]
if points and int(points[0]["customdata"][0]) != st.session_state[key]:
    st.session_state[key] = int(points[0]["customdata"][0])
    st.rerun()

table = pd.DataFrame({
    "Rank": alerts["rank"], "Time": clock(alerts.t_add).dt.strftime("%H:%M:%S"),
    "Side": alerts.direction.map({1: "buy", -1: "sell"}), "Shares": alerts.size_add,
    "Size vs usual": alerts.size_ratio.round(1), "Lived (s)": alerts.lifetime_s.round(2),
    "Score": alerts.score.round(3)})
if show_truth:
    table["Truth"] = alerts.role.map(ROLE_NAMES)
chosen = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row",
                      height=250, key=f"table_{key}_{k}")
if chosen.selection.rows:
    row_id = int(alerts.order_id.iloc[chosen.selection.rows[0]])
    if row_id != st.session_state[key]:
        st.session_state[key] = row_id
        st.rerun()

alert = alerts.set_index("order_id").loc[st.session_state[key]]
order = orders.loc[st.session_state[key]]
st.divider()
st.subheader(f"Alert #{int(alert['rank'])}: order {st.session_state[key]}")
left, right = st.columns([3, 2])
with left:
    st.markdown("**Order book around the order**")
    pad = st.radio("Context", [2, 5, 15, 60], index=1, horizontal=True, format_func=lambda s: f"±{s}s",
                   label_visibility="collapsed")
    st.plotly_chart(book_heatmap(ticker, order, pad), key=f"book_{key}")
    st.caption("Darker = more shares resting at that price. The orange line is this order from add to end.")
with right:
    st.markdown("**Life story**")
    text, events = life_story(ticker, order)
    st.markdown(text)
    st.dataframe(events, hide_index=True)
    st.markdown("**Why it was flagged**")
    st.write(alert.why)
    detail = json.loads(alert.detail)
    if detector == "lgbm":
        st.plotly_chart(shap_chart(detail), key=f"shap_{key}")
    else:
        st.dataframe(pd.DataFrame({"Feature": list(detail), "Value": list(detail.values())}), hide_index=True)
    if show_truth:
        st.info(f"Ground truth: {ROLE_NAMES[order.role].lower()}.")
