"""
Streamlit app: Nifty (or other index) candlesticks + planet ecliptic longitudes.

Top panel  : OHLC candlesticks from Yahoo Finance.
Bottom panel: Planet ecliptic longitudes from Swiss Ephemeris.
             – Direct motion  : solid line in planet colour.
             – Retrograde     : dotted line in a muted/warm variant of the colour.
             – R / D stations : labelled scatter markers at ingress/egress.
             – Conjunctions   : vertical dotted lines across both panels.
Shared x-axis — zoom / pan / hover are linked across both panels.

Run:
    streamlit run nifty_planet_app.py
"""
from __future__ import annotations

import colorsys
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import swisseph as swe
import yfinance as yf
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Nifty × Planet Longitudes",
    layout="wide",
    initial_sidebar_state="expanded",
)

MUMBAI_LAT, MUMBAI_LON, MUMBAI_ALT = 19.0760, 72.8777, 14
IST = pytz.timezone("Asia/Kolkata")

PLANETS: dict[str, tuple[int, str]] = {
    "Sun":     (swe.SUN,        "#ff7f0e"),
    "Moon":    (swe.MOON,       "#1e90ff"),
    "Mercury": (swe.MERCURY,    "#2ca02c"),
    "Venus":   (swe.VENUS,      "#ff69b4"),
    "Mars":    (swe.MARS,       "#d62728"),
    "Jupiter": (swe.JUPITER,    "#ffd700"),
    "Saturn":  (swe.SATURN,     "#1f3a93"),
    "Rahu":    (swe.MEAN_NODE,  "#888888"),
    "Ketu":    (-1,             "#8b4513"),   # derived
}

INSTRUMENTS: dict[str, str] = {
    "NIFTY 50":         "^NSEI",
    "BANK NIFTY":       "^NSEBANK",
    "SENSEX":           "^BSESN",
    "NIFTY IT":         "^CNXIT",
    "NIFTY MIDCAP 100": "NIFTY_MIDCAP_100.NS",
}

RESOLUTIONS = {
    "Daily (fastest)": "1D",
    "12 hours":        "12h",
    "6 hours":         "6h",
    "4 hours":         "4h",
    "1 hour (slow)":   "1h",
}

# ------------------------------------------------------------------
# Colour helpers
# ------------------------------------------------------------------
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

def retro_color(hex_color: str) -> str:
    """
    Returns a warm, desaturated variant of the given hex colour for retrograde
    segments: hue is shifted slightly toward orange-red, saturation is halved,
    and lightness is raised — visually distinct but clearly related.
    """
    r, g, b = (x / 255 for x in hex_to_rgb(hex_color))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Shift hue ~15° toward orange-red, desaturate, lighten
    h_new = (h + 0.04) % 1.0
    s_new = max(0.0, s * 0.45)
    l_new = min(1.0, l * 1.45 + 0.15)
    rn, gn, bn = colorsys.hls_to_rgb(h_new, l_new, s_new)
    return "#{:02x}{:02x}{:02x}".format(int(rn * 255), int(gn * 255), int(bn * 255))

# ------------------------------------------------------------------
# Data fetchers (cached)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(symbol: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end + timedelta(days=1),
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def compute_planet_longitudes(
    start: date,
    end: date,
    freq: str,
    topocentric: bool,
    true_node: bool,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        datetime, Sun, Moon, ..., Ketu          (ecliptic longitude °)
        Sun_spd, Moon_spd, ..., Ketu_spd        (°/day; negative = retrograde)
    """
    if topocentric:
        swe.set_topo(MUMBAI_LON, MUMBAI_LAT, MUMBAI_ALT)
        flags = swe.FLG_SWIEPH | swe.FLG_TOPOCTR | swe.FLG_SPEED
    else:
        flags = swe.FLG_SWIEPH | swe.FLG_SPEED

    start_dt = IST.localize(datetime.combine(start, datetime.min.time()))
    end_dt   = IST.localize(datetime.combine(end,   datetime.min.time()))
    dates = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz=IST)

    rahu_id = swe.TRUE_NODE if true_node else swe.MEAN_NODE

    rows = []
    for dt_ist in dates:
        dt_utc = dt_ist.astimezone(pytz.UTC)
        jd = swe.julday(
            dt_utc.year, dt_utc.month, dt_utc.day,
            dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600,
        )
        row = {"datetime": dt_ist}
        rahu_lon, rahu_spd = None, None
        for name, (pid, _) in PLANETS.items():
            if name == "Ketu":
                # Ketu is always exactly 180° opposite Rahu
                row[name]          = (rahu_lon + 180.0) % 360.0 if rahu_lon is not None else None
                # Ketu speed mirrors Rahu (same magnitude, same direction — both retrograde)
                row[f"{name}_spd"] = rahu_spd if rahu_spd is not None else None
            elif name == "Rahu":
                xx, _ = swe.calc_ut(jd, rahu_id, flags)
                rahu_lon = xx[0]
                rahu_spd = xx[3]
                row[name]          = rahu_lon
                row[f"{name}_spd"] = rahu_spd
            else:
                xx, _ = swe.calc_ut(jd, pid, flags)
                row[name]          = xx[0]
                row[f"{name}_spd"] = xx[3]
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Motion analysis helpers
# ------------------------------------------------------------------
def split_motion_segments(
    df: pd.DataFrame, name: str
) -> list[dict]:
    """
    Splits a planet series into consecutive direct / retrograde segments.
    Each segment overlaps by one point so the drawn line stays continuous.
    Returns list of dicts: {is_retro, x, y}
    """
    spd_col = f"{name}_spd"
    if spd_col not in df.columns or df.empty:
        return [{"is_retro": False, "x": df["datetime"].tolist(), "y": df[name].tolist()}]

    is_retro = (df[spd_col] < 0).tolist()
    xs = df["datetime"].tolist()
    ys = df[name].tolist()

    segments: list[dict] = []
    cur_retro = is_retro[0]
    seg_x, seg_y = [xs[0]], [ys[0]]

    for i in range(1, len(xs)):
        if is_retro[i] != cur_retro:
            seg_x.append(xs[i])   # one-point overlap for seamless join
            seg_y.append(ys[i])
            segments.append({"is_retro": cur_retro, "x": seg_x, "y": seg_y})
            cur_retro = is_retro[i]
            seg_x, seg_y = [xs[i]], [ys[i]]
        else:
            seg_x.append(xs[i])
            seg_y.append(ys[i])

    segments.append({"is_retro": cur_retro, "x": seg_x, "y": seg_y})
    return segments


def find_stations(
    df: pd.DataFrame, name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (retro_stations, direct_stations) — rows where the planet
    changes from direct→retrograde (R) or retrograde→direct (D).
    """
    spd_col = f"{name}_spd"
    if spd_col not in df.columns or len(df) < 2:
        empty = df.iloc[0:0]
        return empty, empty
    spd = df[spd_col]
    r_mask = (spd < 0) & (spd.shift(1) >= 0)   # just turned retrograde
    d_mask = (spd >= 0) & (spd.shift(1) < 0)   # just turned direct
    return df[r_mask].copy(), df[d_mask].copy()


def find_conjunctions(
    df: pd.DataFrame,
    planets: list[str],
    orb: float = 6.0,
) -> list[dict]:
    """
    For every pair of selected planets, finds the moment of closest approach
    within each conjunction window (angular separation ≤ orb).
    Returns list of dicts: {datetime, p1, p2, sep}.
    """
    results: list[dict] = []
    valid = [p for p in planets if p in df.columns]
    for i, p1 in enumerate(valid):
        for p2 in valid[i + 1:]:
            diff = ((df[p1] - df[p2] + 180) % 360) - 180
            abs_diff = diff.abs()
            in_conj = abs_diff <= orb

            if not in_conj.any():
                continue

            # Group consecutive True runs
            group_ids = (in_conj != in_conj.shift()).cumsum()
            for _, grp in df[in_conj].groupby(group_ids[in_conj]):
                peak_idx = abs_diff.loc[grp.index].idxmin()
                results.append({
                    "datetime": df.loc[peak_idx, "datetime"],
                    "p1": p1,
                    "p2": p2,
                    "sep": round(abs_diff.loc[peak_idx], 2),
                })
    return results


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Controls")

instrument_name = st.sidebar.selectbox("Instrument", list(INSTRUMENTS.keys()))
symbol = INSTRUMENTS[instrument_name]

today = date.today()
default_start = today - timedelta(days=180)

c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Start", default_start, max_value=today)
end_date   = c2.date_input("End",   today,         max_value=today)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

selected_planets = st.sidebar.multiselect(
    "Planets",
    options=list(PLANETS.keys()),
    default=["Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"],
)

res_label = st.sidebar.select_slider(
    "Planet sampling",
    options=list(RESOLUTIONS.keys()),
    value="6 hours",
)
resolution = RESOLUTIONS[res_label]

forecast_days = st.sidebar.slider(
    "Forecast planet positions (days ahead)",
    min_value=0, max_value=180, value=60, step=5,
    help="Extend planet longitude curves this many days beyond the price end date. "
         "Swiss Ephemeris computes exact future positions — no extrapolation.",
)
forecast_end = end_date + timedelta(days=forecast_days)

chart_type = st.sidebar.radio("Price chart", ["Candlestick", "Close line"], horizontal=True)

with st.sidebar.expander("Advanced", expanded=False):
    use_topo      = st.checkbox("Topocentric (Mumbai)", value=False,
                                help="Account for observer parallax at Mumbai.")
    use_true_node = st.checkbox("True Node for Rahu/Ketu", value=False,
                                help="Uses oscillating true node instead of the mean node.")
    show_zero     = st.checkbox("Show longitude = 0° line", value=True)
    hide_weekends = st.checkbox(
        "Hide weekends on price axis",
        value=False,
        help="Collapses Sat/Sun on the price chart. "
             "When OFF (recommended), planet curves stay mathematically continuous.",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Motion overlays")

show_retro_lines    = st.sidebar.checkbox("Retrograde / Direct lines", value=True,
                                          help="Dotted line during retrograde, solid during direct motion.")
show_station_labels = st.sidebar.checkbox("R / D station markers", value=True,
                                          help="Mark exact Retrograde (R) and Direct (D) ingress points.")
show_conjunctions   = st.sidebar.checkbox("Conjunction lines", value=True,
                                          help="Vertical dotted lines when two selected planets are within orb.")
conj_orb = st.sidebar.slider(
    "Conjunction orb (°)", min_value=1, max_value=15, value=6, step=1,
    help="Angular separation threshold to detect a conjunction.",
    disabled=not show_conjunctions,
)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
with st.spinner(f"Fetching {instrument_name} from Yahoo…"):
    price_df = fetch_prices(symbol, start_date, end_date)

with st.spinner("Computing planetary positions…"):
    planet_df = compute_planet_longitudes(
        start_date, forecast_end, resolution, use_topo, use_true_node,
    )

if hide_weekends:
    dow = planet_df["datetime"].dt.dayofweek
    planet_df = planet_df[dow < 5].reset_index(drop=True)

if price_df.empty:
    st.error(f"No price data returned for {symbol}. Try a different date range.")
    st.stop()

# ------------------------------------------------------------------
# Pre-compute motion events (before figure)
# ------------------------------------------------------------------
conjunction_events: list[dict] = []
if show_conjunctions and len(selected_planets) >= 2:
    conjunction_events = find_conjunctions(planet_df, selected_planets, orb=float(conj_orb))

# ------------------------------------------------------------------
# Header strip
# ------------------------------------------------------------------
last_close  = float(price_df["Close"].iloc[-1])
first_close = float(price_df["Close"].iloc[0])
change_pct  = (last_close / first_close - 1) * 100

h1, h2, h3, h4 = st.columns(4)
h1.metric(instrument_name, f"{last_close:,.2f}", f"{change_pct:+.2f}% in range")
h2.metric("Bars", f"{len(price_df):,}")
h3.metric("Planet samples", f"{len(planet_df):,}")
if forecast_days > 0:
    h4.metric("Forecast to", forecast_end.strftime("%d %b %Y"))
else:
    h4.metric("Location", "Mumbai (IST)")

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.62, 0.38],
    subplot_titles=(f"{instrument_name} — {symbol}", "Planet ecliptic longitude (°)"),
)

# ── Price panel ─────────────────────────────────────────────────────
if chart_type == "Candlestick":
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df["Open"],   high=price_df["High"],
            low=price_df["Low"],     close=price_df["Close"],
            name=instrument_name,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )
else:
    fig.add_trace(
        go.Scatter(
            x=price_df.index, y=price_df["Close"],
            name="Close",
            line=dict(color="#1f77b4", width=1.6),
            showlegend=False,
        ),
        row=1, col=1,
    )

# ── Forecast zone shading + "Today" marker (both panels) ────────────
today_dt = datetime.combine(today, datetime.min.time())
today_str = today_dt.strftime("%Y-%m-%d")

if forecast_days > 0:
    forecast_end_str = forecast_end.strftime("%Y-%m-%d")
    # Shaded forecast region on both panels
    for row_idx in (1, 2):
        fig.add_vrect(
            x0=today_str, x1=forecast_end_str,
            fillcolor="rgba(255,215,80,0.07)",
            line_width=0,
            row=row_idx, col=1,
        )

# "Today" vertical dashed line on both panels
# NOTE: Plotly 6.7.0 has a bug where annotation= on add_vline crashes with
# date-string x-axes. We split into add_vline + add_annotation separately.
for row_idx in (1, 2):
    fig.add_vline(
        x=today_str,
        line=dict(color="rgba(255,200,40,0.75)", dash="dash", width=1.6),
        row=row_idx, col=1,
    )

# "◀ Today" label — anchored to the price panel top (paper y ≈ 0.99)
fig.add_annotation(
    x=today_str,
    xref="x",          # shared x-axis ref
    yref="paper",
    y=0.99,
    text="◀ Today",
    font=dict(size=10, color="rgba(230,180,30,1.0)"),
    xanchor="right",
    yanchor="top",
    showarrow=False,
)

# ── Conjunction vertical lines (both panels) ─────────────────────────
CONJ_LINE_COLOR  = "rgba(160,120,220,0.55)"   # soft purple
CONJ_LABEL_COLOR = "rgba(130,90,200,0.90)"

for ev in conjunction_events:
    dt_str = str(ev["datetime"])
    label  = f"{ev['p1']}∥{ev['p2']} ({ev['sep']:.1f}°)"

    # Draw line on both panels — no annotation= to avoid Plotly 6.7 date-string crash
    for row_idx in (1, 2):
        fig.add_vline(
            x=dt_str,
            line=dict(color=CONJ_LINE_COLOR, dash="dot", width=1.4),
            row=row_idx, col=1,
        )

    # Label in the planet panel only (paper y ≈ 0–0.42 for row 2)
    fig.add_annotation(
        x=dt_str,
        xref="x",
        yref="paper",
        y=0.38,
        text=label,
        font=dict(size=9, color=CONJ_LABEL_COLOR),
        textangle=-90,
        xanchor="center",
        yanchor="top",
        showarrow=False,
    )

# ── Planet longitude panel ───────────────────────────────────────────
PLOT_ORDER = ["Sun", "Moon", "Mercury", "Venus", "Mars",
              "Jupiter", "Saturn", "Rahu", "Ketu"]

# Collect station points for a combined legend-friendly scatter
station_r_x, station_r_y, station_r_text, station_r_color = [], [], [], []
station_d_x, station_d_y, station_d_text, station_d_color = [], [], [], []

for name in PLOT_ORDER:
    if name not in selected_planets:
        continue

    _, base_color = PLANETS[name]
    retro_col     = retro_color(base_color)
    first_seg     = True   # for legend grouping

    # ── Direct / retrograde line segments ──
    if show_retro_lines:
        segments = split_motion_segments(planet_df, name)
    else:
        # Single solid trace (original behaviour)
        segments = [{"is_retro": False,
                     "x": planet_df["datetime"].tolist(),
                     "y": planet_df[name].tolist()}]

    for seg in segments:
        is_retro = seg["is_retro"]
        color    = retro_col if is_retro else base_color
        dash     = "dot"    if is_retro else "solid"
        width    = 1.6      if is_retro else 1.3
        suffix   = " ℞"     if is_retro else ""

        fig.add_trace(
            go.Scatter(
                x=seg["x"],
                y=seg["y"],
                mode="lines",
                name=name + suffix,
                showlegend=first_seg,
                legendgroup=name,
                line=dict(color=color, dash=dash, width=width),
                hovertemplate=(
                    f"<b>{name}{'  ℞ (retrograde)' if is_retro else ''}</b><br>"
                    f"%{{x|%d %b %Y %H:%M}}<br>lon %{{y:.3f}}°<extra></extra>"
                ),
            ),
            row=2, col=1,
        )
        first_seg = False

    # ── Station markers (R / D) ──
    if show_station_labels:
        r_df, d_df = find_stations(planet_df, name)
        for _, row_data in r_df.iterrows():
            station_r_x.append(row_data["datetime"])
            station_r_y.append(row_data[name])
            station_r_text.append(f"{name} R")
            station_r_color.append(retro_col)
        for _, row_data in d_df.iterrows():
            station_d_x.append(row_data["datetime"])
            station_d_y.append(row_data[name])
            station_d_text.append(f"{name} D")
            station_d_color.append(base_color)

# Add all R stations as one scatter trace
if show_station_labels and station_r_x:
    fig.add_trace(
        go.Scatter(
            x=station_r_x,
            y=station_r_y,
            mode="markers+text",
            name="Station ℞",
            legendgroup="stations",
            marker=dict(
                symbol="circle",
                size=9,
                color=station_r_color,
                line=dict(width=1.5, color="rgba(180,60,60,0.9)"),
            ),
            text=["℞"] * len(station_r_x),
            textposition="top center",
            textfont=dict(size=9, color="rgba(180,60,60,0.95)"),
            customdata=station_r_text,
            hovertemplate="<b>%{customdata}</b><br>%{x|%d %b %Y %H:%M}<br>lon %{y:.2f}°<extra></extra>",
            showlegend=True,
        ),
        row=2, col=1,
    )

# Add all D stations as one scatter trace
if show_station_labels and station_d_x:
    fig.add_trace(
        go.Scatter(
            x=station_d_x,
            y=station_d_y,
            mode="markers+text",
            name="Station D",
            legendgroup="stations",
            marker=dict(
                symbol="diamond",
                size=9,
                color=station_d_color,
                line=dict(width=1.5, color="rgba(30,120,30,0.9)"),
            ),
            text=["D"] * len(station_d_x),
            textposition="bottom center",
            textfont=dict(size=9, color="rgba(30,120,30,0.95)"),
            customdata=station_d_text,
            hovertemplate="<b>%{customdata}</b><br>%{x|%d %b %Y %H:%M}<br>lon %{y:.2f}°<extra></extra>",
            showlegend=True,
        ),
        row=2, col=1,
    )

if show_zero:
    fig.add_hline(
        y=0, line=dict(color="gray", dash="dot", width=0.8),
        row=2, col=1,
    )

# ── Layout ────────────────────────────────────────────────────────────
fig.update_layout(
    height=820,
    hovermode="x unified",
    dragmode="zoom",
    legend=dict(
        orientation="h",
        y=1.05, x=1, xanchor="right",
        font=dict(size=11),
    ),
    margin=dict(t=60, b=40, l=60, r=40),
    xaxis_rangeslider_visible=False,
    template="plotly_white",
)

fig.update_yaxes(title_text="Price",        row=1, col=1)
fig.update_yaxes(title_text="Longitude (°)", row=2, col=1)
fig.update_xaxes(title_text="Date",          row=2, col=1)

if hide_weekends:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=2, col=1)

st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------
# Conjunction table
# ------------------------------------------------------------------
if show_conjunctions and conjunction_events:
    with st.expander(f"Conjunction events  ({len(conjunction_events)} found, orb ≤ {conj_orb}°)", expanded=False):
        conj_rows = [
            {
                "Date/Time (IST)": ev["datetime"].strftime("%d %b %Y  %H:%M"),
                "Planet 1": ev["p1"],
                "Planet 2": ev["p2"],
                "Separation (°)": ev["sep"],
            }
            for ev in sorted(conjunction_events, key=lambda e: e["datetime"])
        ]
        st.dataframe(pd.DataFrame(conj_rows), width="stretch", hide_index=True)

# ------------------------------------------------------------------
# Raw data tables
# ------------------------------------------------------------------
with st.expander("Raw data"):
    t1, t2 = st.tabs(["Prices", "Planet longitudes + speed"])
    t1.dataframe(price_df, width="stretch")
    t2.dataframe(planet_df, width="stretch")

# ------------------------------------------------------------------
# Legend key
# ------------------------------------------------------------------
st.caption(
    "Prices: Yahoo Finance · Planets: Swiss Ephemeris (geocentric by default) · "
    "Ketu = Rahu + 180° · "
    "Solid line = direct motion · Dotted line = retrograde (℞) · "
    "◆ D = direct station · ● ℞ = retrograde station · "
    "Purple dotted verticals = conjunctions within orb · "
    "Yellow dashed line = Today · Shaded region = forecast zone."
)
