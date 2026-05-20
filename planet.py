"""
Streamlit app: Nifty (or other index) candlesticks + planet ecliptic latitudes.

Top panel  : OHLC candlesticks from Yahoo Finance.
Bottom panel: Planet ecliptic latitudes from Swiss Ephemeris.
Shared x-axis — zoom / pan / hover are linked across both panels.

Run:
    streamlit run nifty_planet_app.py
"""
from __future__ import annotations

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
    page_title="Nifty × Planet Latitudes",
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
    # yfinance sometimes returns a MultiIndex on columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def compute_planet_latitudes(
    start: date,
    end: date,
    freq: str,
    topocentric: bool,
    true_node: bool,
) -> pd.DataFrame:
    if topocentric:
        swe.set_topo(MUMBAI_LON, MUMBAI_LAT, MUMBAI_ALT)
        flags = swe.FLG_SWIEPH | swe.FLG_TOPOCTR
    else:
        flags = swe.FLG_SWIEPH

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
        for name, (pid, _) in PLANETS.items():
            if name == "Ketu":
                row[name] = 0.0                 # on the ecliptic
            elif name == "Rahu":
                xx, _ = swe.calc_ut(jd, rahu_id, flags)
                row[name] = xx[1]
            else:
                xx, _ = swe.calc_ut(jd, pid, flags)
                row[name] = xx[1]
        rows.append(row)

    return pd.DataFrame(rows)


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
    min_value=0,
    max_value=365,
    value=90,
    step=5,
    help="Extend planet latitude curves this many days beyond today. "
         "No candlestick data will appear in the future region.",
)
planet_end_date = end_date + timedelta(days=forecast_days)

chart_type = st.sidebar.radio("Price chart", ["Candlestick", "Close line"], horizontal=True)

with st.sidebar.expander("Advanced", expanded=False):
    use_topo        = st.checkbox("Topocentric (Mumbai)", value=False,
                                  help="Account for observer parallax at Mumbai. "
                                       "Mainly visible as tiny daily ripples on the Moon.")
    use_true_node   = st.checkbox("True Node for Rahu/Ketu", value=False,
                                  help="Uses oscillating true node instead of the mean node.")
    show_zero       = st.checkbox("Show latitude = 0 line", value=True)
    hide_weekends   = st.checkbox(
        "Hide weekends on price axis",
        value=False,
        help="Collapses Sat/Sun on the price chart. "
             "When OFF (recommended), planet curves stay mathematically continuous. "
             "When ON, planet samples are filtered to weekdays so both panels align.",
    )

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
with st.spinner(f"Fetching {instrument_name} from Yahoo…"):
    price_df = fetch_prices(symbol, start_date, end_date)

with st.spinner("Computing planetary positions…"):
    planet_df = compute_planet_latitudes(
        start_date, planet_end_date, resolution, use_topo, use_true_node,
    )

# If the user opted to collapse weekends on the price axis, drop weekend
# samples from the planet frame too. Otherwise Plotly squeezes them into the
# collapsed region and the line zig-zags across the chart.
if hide_weekends:
    dow = planet_df["datetime"].dt.dayofweek
    planet_df = planet_df[dow < 5].reset_index(drop=True)

if price_df.empty:
    st.error(f"No price data returned for {symbol}. Try a different date range.")
    st.stop()

# ------------------------------------------------------------------
# Header strip
# ------------------------------------------------------------------
last_close = float(price_df["Close"].iloc[-1])
first_close = float(price_df["Close"].iloc[0])
change_pct = (last_close / first_close - 1) * 100

h1, h2, h3, h4 = st.columns(4)
h1.metric(instrument_name, f"{last_close:,.2f}", f"{change_pct:+.2f}% in range")
h2.metric("Bars", f"{len(price_df):,}")
h3.metric("Planet samples", f"{len(planet_df):,}")
h4.metric("Forecast to", planet_end_date.strftime("%d %b %Y") if forecast_days > 0 else "Today")

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.62, 0.38],
    subplot_titles=(f"{instrument_name} — {symbol}", "Planet ecliptic latitude (°)"),
)

# --- Price panel -----------------------------------------------------
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

# --- Planet panel ----------------------------------------------------
plot_order = ["Sun", "Moon", "Mercury", "Venus", "Mars",
              "Jupiter", "Saturn", "Rahu", "Ketu"]

for name in plot_order:
    if name not in selected_planets:
        continue
    _, color = PLANETS[name]
    fig.add_trace(
        go.Scatter(
            x=planet_df["datetime"],
            y=planet_df[name],
            name=name,
            line=dict(color=color, width=1.3),
            legendgroup="planets",
            hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %Y %H:%M}}<br>lat %{{y:.3f}}°<extra></extra>",
        ),
        row=2, col=1,
    )

if show_zero:
    fig.add_hline(
        y=0, line=dict(color="gray", dash="dot", width=0.8),
        row=2, col=1,
    )

# --- Future forecast shading -----------------------------------------
if forecast_days > 0:
    today_str = today.isoformat()
    future_end_str = planet_end_date.isoformat()

    for row_num in (1, 2):
        # Shaded band for the future region
        fig.add_vrect(
            x0=today_str, x1=future_end_str,
            fillcolor="rgba(255,255,100,0.06)",
            line_width=0,
            row=row_num, col=1,
        )
    # Dashed vertical "today" line on both panels
    fig.add_vline(
        x=today_str,
        line=dict(color="rgba(255,255,0,0.55)", dash="dash", width=1.2),
    )
    # Annotation label
    fig.add_annotation(
        x=today_str, y=1.01,
        xref="x2", yref="y2 domain",
        text="◀ Today",
        showarrow=False,
        font=dict(size=11, color="rgba(255,255,0,0.7)"),
        xanchor="right",
    )

# --- Layout ----------------------------------------------------------
fig.update_layout(
    height=780,
    hovermode="x unified",
    dragmode="zoom",
    legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
    margin=dict(t=60, b=40, l=60, r=40),
    xaxis_rangeslider_visible=False,
    template="plotly_white",
)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Latitude (°)", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)

# Skip weekends/holidays on the price x-axis ONLY when the user opts in.
# Default (off) keeps planet curves mathematically continuous, which is the
# standard convention for astro-price overlays. When on, we ALSO drop weekend
# samples from the planet series so Plotly doesn't compress them into the
# collapsed range and produce vertical jitter.
if hide_weekends:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=1, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Data tables
# ------------------------------------------------------------------
with st.expander("Raw data"):
    t1, t2 = st.tabs(["Prices", "Planet latitudes"])
    t1.dataframe(price_df, use_container_width=True)
    t2.dataframe(planet_df, use_container_width=True)

st.caption(
    "Prices: Yahoo Finance · Planets: Swiss Ephemeris (geocentric by default) · "
    "Rahu/Ketu are lunar nodes — latitude is 0 by definition. · "
    "Yellow shaded region = future forecast (no price data available)."
)
