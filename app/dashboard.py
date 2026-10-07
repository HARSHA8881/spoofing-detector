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

from src import feedback, report  # noqa: E402
from src.load import EMPTY_ASK, EMPTY_BID, PRICE_SCALE, TICK  # noqa: E402

ART = ROOT / "artifacts"
LABELS = ART / "review_labels.csv"
DAY = pd.Timestamp("2012-06-21")  # the LOBSTER sample date
EVENT_NAMES = {1: "added", 2: "partly cancelled", 3: "cancelled", 4: "executed"}
DETECTORS = {"lgbm": "LightGBM classifier", "lgbm_rank": "LightGBM ranker", "rules": "Rule scorecard",
             "iforest": "Isolation Forest", "reviewed": "LightGBM + your reviews"}
ROLE_NAMES = {"spoof": "Injected spoof", "hard_neg": "Hard negative", "orig": "Original order"}
ROLE_COLORS = {"spoof": "#2a78d6", "hard_neg": "#eb6834", "orig": "#1baf7a"}
METRIC_NAMES = {"pr_auc": "PR-AUC", "precision@20": "Precision@20", "precision@50": "Precision@50",
                "precision@100": "Precision@100", "ep_precision@20": "Episode precision@20",
                "ep_precision@50": "Episode precision@50", "ep_recall@50": "Episode recall@50",
                "recall@100": "Recall@100", "hard_neg_fpr@50": "Hard-negative FPR@50"}
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUES = [[0.0, "#cde2fb"], [0.5, "#5598e7"], [1.0, "#0d366b"]]

st.set_page_config(page_title="Spoofing surveillance", page_icon="🔎", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 3.4rem; max-width: 1500px;}
  [data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e1e0d9;}
  .hero {background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%); color: #fff;
         padding: 22px 28px; border-radius: 16px; margin-bottom: 18px;}
  .hero h1 {font-size: 1.75rem; margin: 0 0 4px 0; padding: 0; color: #fff; font-weight: 650;}
  .hero p {margin: 0; color: #cde2fb; font-size: 0.98rem;}
  .kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 8px;}
  .kpi {background: #fff; border: 1px solid #e1e0d9; border-radius: 14px; padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(11,11,11,.04);}
  .kpi .label {color: #52514e; font-size: .8rem; letter-spacing: .02em;}
  .kpi .value {color: #0b0b0b; font-size: 1.7rem; font-weight: 650; line-height: 1.25;}
  .kpi .note {color: #898781; font-size: .78rem;}
  .card {background: #fff; border: 1px solid #e1e0d9; border-radius: 14px; padding: 16px 20px; margin-bottom: 14px;}
  .card h4 {margin: 0 0 6px 0; padding: 0; font-size: 1rem;}
  .pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
         border: 1px solid #c3c2b7; color: #0b0b0b; background: #f0efec; margin-right: 6px;}
  .dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;}
  .story {color: #0b0b0b; line-height: 1.55; font-size: .95rem;}
  .muted {color: #52514e; font-size: .88rem;}
  [data-testid="stPlotlyChart"] {background: #fff; border: 1px solid #e1e0d9; border-radius: 14px; padding: 6px;}
  button[data-baseweb="tab"] {font-size: 1rem;}
</style>
""", unsafe_allow_html=True)


def clock(seconds):
    return DAY + pd.to_timedelta(seconds, unit="s")


def kpis(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                    f'<div class="note">{note}</div></div>' for label, value, note in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def role_pill(role: str) -> str:
    return (f'<span class="pill"><span class="dot" style="background:{ROLE_COLORS[role]}"></span>'
            f'{ROLE_NAMES[role]}</span>')


# -- data ---------------------------------------------------------------------
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
def load_train() -> pd.DataFrame:
    return pd.read_parquet(ART / "features" / "train.parquet").set_index("order_id")


@st.cache_data
def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ART / name, **kwargs)


@st.cache_data
def mid_series(ticker: str) -> pd.DataFrame:
    time, book = load_messages(ticker).time.to_numpy(), load_book(ticker)
    grid = np.arange(np.ceil(time[0]), time[-1], 5.0)
    rows = np.searchsorted(time, grid, side="right") - 1
    return pd.DataFrame({"time": grid, "mid": (book[rows, 0] + book[rows, 2]) / 2 / PRICE_SCALE})


def build_alerts(ticker: str, detector: str, k: int, by_episode: bool) -> pd.DataFrame:
    """Top alerts with the order's own columns joined on. One row per episode if asked."""
    orders = load_orders(ticker)
    if detector == "reviewed":
        det, scores = st.session_state[f"reviewed_{ticker}"]
        scored = orders[orders.role != "genuine"]
        top = scored.loc[scores[scored.index].nlargest(400).index]
        alerts = pd.DataFrame({"order_id": top.index, "score": scores[top.index].to_numpy(),
                               "why": det.explain(top),
                               "detail": [json.dumps(d) for d in det.shap_values(top).round(4).to_dict("records")]})
    else:
        every = load_alerts()
        alerts = every[(every.ticker == ticker) & (every.detector == detector)].sort_values("rank")
    alerts = alerts.join(orders[["t_add", "price", "direction", "size_add", "size_pct", "lifetime_s", "role",
                                 "alert_group", "ep_n_orders"]], on="order_id")
    if by_episode:
        alerts = alerts.drop_duplicates("alert_group")
    alerts = alerts.head(k).reset_index(drop=True)
    alerts["rank"] = np.arange(1, len(alerts) + 1)
    return alerts


# -- charts -------------------------------------------------------------------
def style(fig: go.Figure, height: int, dollars: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=12, r=12, t=14, b=10), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(color=INK, size=13), hovermode="closest",
                      hoverlabel=dict(bgcolor="#fff", font_color=INK, bordercolor=AXIS),
                      legend=dict(orientation="h", y=1.1, x=0, font=dict(color=SECONDARY)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False,
                     tickprefix="$" if dollars else "")
    return fig


def timeline(ticker: str, alerts: pd.DataFrame, show_truth: bool, selected) -> go.Figure:
    mid = mid_series(ticker)
    fig = go.Figure(go.Scatter(x=clock(mid.time), y=mid.mid, mode="lines", name="Mid price",
                               line=dict(color=MUTED, width=1.5), hoverinfo="skip"))
    groups = [(r, alerts[alerts.role == r]) for r in ROLE_NAMES] if show_truth else [("alert", alerts)]
    for role, g in groups:
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=clock(g.t_add), y=g.price / PRICE_SCALE, mode="markers",
            name=ROLE_NAMES.get(role, "Alert"), customdata=np.stack([g.order_id, g["rank"]], axis=1),
            marker=dict(size=10, color=ROLE_COLORS.get(role, BLUE), line=dict(color=SURFACE, width=2)),
            hovertemplate="alert #%{customdata[1]}<br>%{x|%H:%M:%S}<br>order at $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[clock(selected.t_add)], y=[selected.price / PRICE_SCALE], mode="markers",
                             name="Selected", hoverinfo="skip",
                             marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color=INK, width=2))))
    return style(fig, 330)


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
        name="This order", line=dict(color=ORANGE, width=2),
        marker=dict(size=11, color=ORANGE, line=dict(color=SURFACE, width=2)),
        text=["added", EVENT_NAMES.get(int(life.event_type.iloc[-1]), "still open")],
        hovertemplate="%{text} %{x|%H:%M:%S.%L}<extra></extra>"))
    other = msgs[(msgs.time >= grid[0]) & (msgs.time <= grid[-1]) & msgs.event_type.isin([4, 5])
                 & (msgs.direction == -order.direction)]
    if len(other):
        fig.add_trace(go.Scatter(
            x=clock(other.time), y=other.price / PRICE_SCALE, mode="markers", name="Trades on the opposite side",
            marker=dict(size=8, symbol="diamond", color="#1baf7a", line=dict(color=SURFACE, width=1)),
            customdata=other["size"], hovertemplate="%{customdata:,} shares traded at $%{y:.2f}<extra></extra>"))
    return style(fig, 430)


def push_chart(pushes: pd.Series, title: str) -> go.Figure:
    """Horizontal bars for signed contributions: orange pushes up, blue pushes down."""
    s = pushes.sort_values(key=abs).tail(10)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", width=0.6,
                           marker=dict(color=[ORANGE if v > 0 else BLUE for v in s.values], cornerradius=4),
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig = style(fig, 60 + 26 * len(s), dollars=False)
    fig.update_yaxes(tickfont=dict(color=INK), showgrid=False)
    fig.update_xaxes(title=dict(text=title, font=dict(color=SECONDARY, size=12)), zeroline=True,
                     zerolinecolor=AXIS)
    return fig


def metric_bars(values: pd.Series, low: pd.Series | None, high: pd.Series | None) -> go.Figure:
    error = None
    if low is not None:
        error = dict(type="data", symmetric=False, array=(high - values).clip(lower=0),
                     arrayminus=(values - low).clip(lower=0), color=INK, thickness=1.5, width=6)
    names = [DETECTORS[d] for d in values.index]
    fig = go.Figure(go.Bar(x=names, y=values.values, width=0.5, marker=dict(color=BLUE, cornerradius=4),
                           error_y=error, hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    top = values if high is None else np.maximum(values, high)  # labels sit above the whiskers
    fig.add_trace(go.Scatter(x=names, y=top + 0.02, mode="text", text=[f"{v:.2f}" for v in values.values],
                             textposition="top center", textfont=dict(color=INK, size=14), hoverinfo="skip"))
    fig.update_layout(showlegend=False)
    fig = style(fig, 330, dollars=False)
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_xaxes(showgrid=False, tickfont=dict(color=INK))
    return fig


# -- tabs ---------------------------------------------------------------------
def review_tab(ticker: str, detector: str, k: int, by_episode: bool, show_truth: bool) -> None:
    orders = load_orders(ticker)
    scored = orders[orders.role != "genuine"]
    alerts = build_alerts(ticker, detector, k, by_episode)
    unit = "episodes" if by_episode else "orders"
    items = [("Orders scored", f"{len(scored):,}", f"{ticker} · one trading day"),
             ("Alerts in view", f"{len(alerts)}", f"top {unit} from {DETECTORS[detector]}")]
    if show_truth:
        hits = (alerts.role == "spoof").mean()
        hard = int((alerts.role == "hard_neg").sum())
        items += [(f"Precision@{k}", f"{hits:.0%}", f"of these alerts are injected spoofs"),
                  ("Hard negatives flagged", f"{hard}", f"of {int((scored.role == 'hard_neg').sum())} legitimate large cancels"),
                  ("Injected spoof orders", f"{int(scored.label.sum())}", f"1 in {len(scored) // max(int(scored.label.sum()), 1):,} orders")]
    kpis(items)

    key = f"selected_{ticker}_{detector}_{by_episode}"
    if st.session_state.get(key) not in set(alerts.order_id):
        st.session_state[key] = int(alerts.order_id.iloc[0])
    selected = alerts.set_index("order_id").loc[st.session_state[key]]

    st.markdown("#### Alerts on the day's price")
    st.caption("Each dot is one alerted order at its own price. Click a dot, or a row in the table, to open it.")
    picked = st.plotly_chart(timeline(ticker, alerts, show_truth, selected), on_select="rerun",
                             selection_mode="points", key=f"timeline_{key}_{k}")
    points = [p for p in picked.selection.points if "customdata" in p]
    if points and int(points[0]["customdata"][0]) != st.session_state[key]:
        st.session_state[key] = int(points[0]["customdata"][0])
        st.rerun()

    labels = feedback.load_labels(LABELS)
    verdicts = labels[labels.ticker == ticker].set_index("order_id").verdict
    table = pd.DataFrame({
        "Rank": alerts["rank"], "Time": clock(alerts.t_add).dt.strftime("%H:%M:%S"),
        "Side": alerts.direction.map({1: "buy", -1: "sell"}), "Shares": alerts.size_add,
        "Larger than": alerts.size_pct, "Lived (s)": alerts.lifetime_s.round(2),
        "Orders in episode": alerts.ep_n_orders.clip(lower=1).astype(int), "Score": alerts.score.round(3),
        "Your review": alerts.order_id.map(verdicts).fillna("")})
    if show_truth:
        table["Truth"] = alerts.role.map(ROLE_NAMES)
    chosen = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row", height=260,
        key=f"table_{key}_{k}", column_config={
            "Score": st.column_config.NumberColumn(format="%.3f"),
            "Larger than": st.column_config.ProgressColumn("Larger than (% of day's orders)", min_value=0,
                                                           max_value=1, format="percent")})
    if chosen.selection.rows:
        row_id = int(alerts.order_id.iloc[chosen.selection.rows[0]])
        if row_id != st.session_state[key]:
            st.session_state[key] = row_id
            st.rerun()

    order_id = st.session_state[key]
    alert, order = alerts.set_index("order_id").loc[order_id], orders.loc[order_id]
    st.divider()
    head = f"#### Alert #{int(alert['rank'])} · order {order_id}"
    st.markdown(head)
    if show_truth:
        reason = report.REASON_NAMES.get(order.reason, "")
        st.markdown(role_pill(order.role) + (f'<span class="muted">{reason}</span>' if reason else ""),
                    unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Order book around the order**")
        pad = st.radio("Context", [2, 5, 15, 60], index=1, horizontal=True, format_func=lambda s: f"±{s}s",
                       label_visibility="collapsed", key=f"pad_{key}")
        st.plotly_chart(book_heatmap(ticker, order, pad), key=f"book_{key}")
        st.caption("Darker = more shares resting at that price. The orange line is this order from add to end.")
    with right:
        st.markdown(f'<div class="card"><h4>Life story</h4><div class="story">{report.life_story(order)}</div></div>',
                    unsafe_allow_html=True)
        life = load_messages(ticker)
        life = life[life.order_id == order_id]
        st.dataframe(pd.DataFrame({"Time": clock(life.time).dt.strftime("%H:%M:%S.%f").str[:-3],
                                   "Event": life.event_type.map(EVENT_NAMES), "Shares": life["size"],
                                   "Price": (life.price / PRICE_SCALE).map("${:.2f}".format)}), hide_index=True)
        st.markdown("**Why it was flagged**")
        detail = pd.Series(json.loads(alert.detail))
        if detector in ("lgbm", "lgbm_rank", "reviewed"):
            st.plotly_chart(push_chart(detail, "SHAP push on the score · right = more suspicious"),
                            key=f"shap_{key}")
        else:
            st.markdown(f'<div class="muted">{alert.why}</div>', unsafe_allow_html=True)

        st.markdown("**Your review**")
        b1, b2, b3 = st.columns([1, 1.4, 1.6])
        if b1.button("Spoof", key=f"yes_{key}", use_container_width=True):
            feedback.save_label(ticker, order_id, "spoof", LABELS)
            st.rerun()
        if b2.button("Not spoof", key=f"no_{key}", use_container_width=True):
            feedback.save_label(ticker, order_id, "not spoof", LABELS)
            st.rerun()
        b3.markdown(f'<div class="muted" style="padding-top:8px">{verdicts.get(order_id, "not reviewed yet")}</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Feedback loop")
    n = len(verdicts)
    c1, c2 = st.columns([3, 1])
    c1.markdown(f'<div class="muted">You have reviewed <b>{n}</b> alert{"s" if n != 1 else ""} on {ticker} '
                f'({int((verdicts == "spoof").sum())} spoof, {int((verdicts == "not spoof").sum())} not spoof). '
                "Retraining adds them to the training set with extra weight and rescores the day; the result "
                'appears as the detector "LightGBM + your reviews".</div>', unsafe_allow_html=True)
    if c2.button("Retrain with my reviews", disabled=n == 0, use_container_width=True, type="primary"):
        with st.spinner("Retraining LightGBM"):
            st.session_state[f"reviewed_{ticker}"] = feedback.retrain(
                load_train(), scored, labels[labels.ticker == ticker])
        st.session_state["goto_reviewed"] = True
        st.rerun()


def comparison_tab() -> None:
    summary = load_csv("results.csv", index_col=0)
    ci = load_csv("results_ci.csv")
    seeds = load_csv("results_seeds.csv", header=[0, 1], index_col=0)
    names = list(summary.index)
    st.markdown("#### Detectors on unseen stock-days and unseen injection settings")
    st.caption("Test days AMZN and MSFT; trained on AAPL, GOOG and INTC with different injection settings. "
               "Bars are injection seed 0; whiskers are 95% bootstrap intervals where available.")
    metric = st.radio("Metric", list(METRIC_NAMES), horizontal=True, format_func=METRIC_NAMES.get,
                      label_visibility="collapsed")
    band = ci[ci.metric == metric].set_index("detector").reindex(names)
    has_ci = band.low.notna().all()
    st.plotly_chart(metric_bars(summary[metric], band.low if has_ci else None, band.high if has_ci else None),
                    key="metric_bars")

    st.markdown(f"#### Mean ± standard deviation over {len(load_csv('results_per_seed.csv').seed.unique())} injection seeds")
    table = pd.DataFrame({METRIC_NAMES[m]: [f"{seeds.loc[d, (m, 'mean')]:.2f} ± {seeds.loc[d, (m, 'std')]:.2f}"
                                           for d in names] for m in METRIC_NAMES},
                         index=[DETECTORS[d] for d in names])
    st.dataframe(table)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Which signals matter")
        st.caption("Change in LightGBM PR-AUC when one feature group is removed (seed 0). "
                   "Left = the model got worse without it.")
        abl = load_csv("results_ablation.csv").iloc[1:].set_index("variant").pr_auc_change
        st.plotly_chart(push_chart(abl, "Change in PR-AUC"), key="ablation")
    with right:
        st.markdown("#### Does it only catch the spoofs I wrote one way?")
        st.caption("Share of injected orders that reach LightGBM's top 100 of the day, by why the order "
                   "was cancelled (all seeds pooled).")
        reason = load_csv("results_by_reason.csv").set_index("reason")
        names_r = {"timer": "Spoof: fixed hold", "after_fill": "Spoof: after its own fill",
                   "approach": "Spoof: price came toward it", "moved_away": "Hard negative: price moved away"}
        reason = reason.loc[[r for r in names_r if r in reason.index]]
        fig = go.Figure(go.Bar(
            x=reason.in_top_100, y=[names_r[r] for r in reason.index], orientation="h", width=0.55,
            marker=dict(color=[ORANGE if r == "moved_away" else BLUE for r in reason.index], cornerradius=4),
            text=[f"{v:.0%} of {n:,}" for v, n in zip(reason.in_top_100, reason.n_orders)],
            textposition="outside", textfont=dict(color=INK), cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig = style(fig, 250, dollars=False)
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_yaxes(showgrid=False, tickfont=dict(color=INK), autorange="reversed")
        st.plotly_chart(fig, key="by_reason")

    st.markdown("#### Follow-up runs (LightGBM, seed 0)")
    follow = load_csv("results_followup.csv", index_col=0)[list(METRIC_NAMES)].rename(columns=METRIC_NAMES)
    st.dataframe(follow.style.format("{:.2f}"))


def errors_tab() -> None:
    cases = json.loads((ART / "error_analysis.json").read_text())
    st.markdown("#### Where LightGBM is wrong")
    st.caption("Its five highest-ranked alerts that are not injected spoofs, and the five injected spoofs it "
               "ranked lowest. Each comes with the order's life story and the features that pushed the score.")
    for kind, title in (("false positive", "Top-ranked alerts that are not spoofs"),
                        ("missed spoof", "Injected spoofs it ranked lowest")):
        st.markdown(f"##### {title}")
        for i, c in enumerate(c for c in cases if c["kind"] == kind):
            with st.expander(f"{c['ticker']} · order {c['order_id']} · rank {c['rank']:,} of {c['of']:,}",
                             expanded=i == 0):
                truth = c["truth"] + (f", {c['reason']}" if c["reason"] else "")
                left, right = st.columns([3, 2], gap="large")
                left.markdown(f'<span class="pill">Truth: {truth}</span>', unsafe_allow_html=True)
                left.markdown(f'<div class="story" style="margin-top:10px">{c["story"]}</div>', unsafe_allow_html=True)
                pushes = pd.Series({f"{s['feature']} = {s['value']:.3g}": s["push"] for s in c["shap"]})
                right.plotly_chart(push_chart(pushes, "SHAP push · right = more suspicious"),
                                   key=f"err_{kind}_{i}")


def about_tab() -> None:
    st.markdown("""
<div class="card"><h4>What this is</h4><div class="story">A research tool. It reads NASDAQ order-level data
(LOBSTER samples, 21 June 2012), plants labelled synthetic spoofing episodes in it, and measures whether detectors
find them on stock-days and injection settings they never saw. Every score is computed after the order has
ended.</div></div>
<div class="card"><h4>What an alert means</h4><div class="story">A pattern worth a human look: a large order that
rested briefly, was cancelled unfilled, and had trading on the opposite side. It is not proof of intent. Public
data has no trader IDs, so the fake order cannot be linked to the trade that profited from it.</div></div>
<div class="card"><h4>Limits to keep in mind</h4><div class="story">
• The labels are synthetic. Scores show the pipeline finds this pattern, not that it finds real spoofing.<br>
• Injected orders cannot move the real price, because history does not react.<br>
• The free samples cover five stocks on one date, so "unseen days" means unseen stocks on that date.<br>
• US equities, not Indian F&amp;O. The live Kite path is not built.</div></div>
""", unsafe_allow_html=True)


# -- page ---------------------------------------------------------------------
if not (ART / "alerts.parquet").exists():
    st.error("No results yet. Run `uv run python -m src.run` first.")
    st.stop()

tickers = sorted(load_alerts().ticker.unique())
with st.sidebar:
    st.markdown("### 🔎 Spoofing surveillance")
    ticker = st.selectbox("Stock-day", tickers, format_func=lambda t: f"{t} · 2012-06-21")
    options = [d for d in DETECTORS if d != "reviewed" or f"reviewed_{ticker}" in st.session_state]
    if st.session_state.pop("goto_reviewed", False):
        st.session_state["detector"] = "reviewed"
    if st.session_state.get("detector") not in options:
        st.session_state["detector"] = options[0]
    detector = st.selectbox("Detector", options, format_func=DETECTORS.get, key="detector")
    k = st.slider("Alerts to review", 10, 200, 50, step=10)
    by_episode = st.toggle("Group alerts into episodes", value=False,
                           help="Layered orders and quick repeats on one side are shown as one alert, "
                                "ranked by their most suspicious order.")
    show_truth = st.toggle("Show injected ground truth", value=True,
                           help="Only possible because the spoofing here is synthetic. Real data has no labels.")
    st.caption("Alerts are patterns for a human to review, not proof of intent. Public data has no trader IDs.")

st.markdown(f"""<div class="hero"><h1>{ticker} · order book surveillance</h1>
<p>Top {k} {"episodes" if by_episode else "orders"} flagged by the {DETECTORS[detector]} on 21 June 2012,
scored after the fact on a stock the model never trained on.</p></div>""", unsafe_allow_html=True)

review, compare, errors, about = st.tabs(["Review alerts", "Detector comparison", "Where it fails", "About"])
with review:
    review_tab(ticker, detector, k, by_episode, show_truth)
with compare:
    comparison_tab()
with errors:
    errors_tab()
with about:
    about_tab()
