"""
ASTRO-LEVELS TERMINAL
=====================================================================
NSE F&O previous-period support/resistance (Yahoo Finance) fused with a
live planetary-degree engine (Swiss Ephemeris / Moshier — no data files).

WHAT IT DOES
  1.  Pulls daily OHLC for every symbol in your F&O list and computes:
        - Previous MONTH high / low  (+ the date each printed)
        - Previous WEEK  high / low  (+ dates)
        - Previous DAY   high / low  (+ date)
  2.  Computes, for every planet/node, the GEOCENTRIC and HELIOCENTRIC
        ecliptic longitude (Tropical or Sidereal/Lahiri), speed & retro.
  3.  Highlights a planet's degree when it lands (within an orb) on:
        - a GANN level   : 0, 22.5, 45, 67.5 ... 337.5   (22.5deg grid)
        - a SQUARE degree: 0, 1, 4, 9, 16 ... 324         (n^2)
        - a CUBE degree  : 0, 1, 8, 27, 64 ... 343         (n^3)
  4.  Highlights when GEO longitude ~= HELIO longitude (they coincide).
  5.  Optional GANN Square-of-9 bridge: converts each stock's H/L PRICE
        into a 0-360deg wheel value and flags it when it resonates with a
        planet's geocentric degree (price-meets-planet S/R confluence).

RUN
    pip install streamlit yfinance pandas pyswisseph
    streamlit run astro_levels_terminal.py

NOTES
  - Ephemeris uses the built-in Moshier model (FLG_MOSEPH): accurate to a
    few arc-seconds, zero external data files. Switch to FLG_SWIEPH if you
    have the Swiss Ephemeris files installed and want sub-arc-sec precision.
  - Heliocentric longitude is only meaningful for true planets, so it is
    shown as "--" for Sun, Moon, Rahu and Ketu.
  - A handful of very recently renamed / demerged tickers (e.g. TMPV, LTM,
    VMM, GVT&D) may not resolve on Yahoo yet; they are listed under
    "unresolved symbols" so you can patch SYMBOL_OVERRIDES below.
=====================================================================
"""

from __future__ import annotations
import io
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import swisseph as swe
import yfinance as yf

# --------------------------------------------------------------------------- #
#  CONSTANTS
# --------------------------------------------------------------------------- #
IST = timezone(timedelta(hours=5, minutes=30))

# Default F&O universe (cleaned from fno_list.csv). The sidebar uploader
# overrides this if you supply your own single-column CSV.
DEFAULT_SYMBOLS = [
    "LODHA","HCLTECH","HINDPETRO","BPCL","GVT&D","HEROMOTOCO","TCS","IOC","AUBANK",
    "INDIGO","ASHOKLEY","PIDILITIND","SHRIRAMFIN","MPHASIS","ALKEM","RBLBANK","ABB",
    "POWERINDIA","DALBHARAT","NAUKRI","COFORGE","FORCEMOT","UPL","MANKIND","360ONE",
    "LTF","BANKBARODA","POLICYBZR","ADANIENSOL","KOTAKBANK","OBEROIRLTY","WIPRO",
    "ASIANPAINT","BANKINDIA","M&M","MARUTI","PNB","PIIND","BHARATFORG","INFY",
    "CHOLAFIN","GODREJPROP","KPITTECH","SAMMAANCAP","ICICIGI","SONACOMS","INDUSINDBK",
    "INDIANB","TMPV","AMBUJACEM","TRENT","ABCAPITAL","CIPLA","KAYNES","NESTLEIND",
    "TATAELXSI","ADANIPOWER","HDFCBANK","RECLTD","PFC","CANBK","EICHERMOT","ICICIPRULI",
    "ADANIGREEN","DLF","TATACONSUM","IREDA","LT","ICICIBANK","UNITDSPR","SUZLON",
    "BIOCON","SIEMENS","HDFCLIFE","LICHSGFIN","ADANIENT","BAJAJFINSV","BAJFINANCE",
    "LTM","HDFCAMC","SBIN","KALYANKJIL","PGEL","INOXWIND","TATAPOWER","ULTRACEMCO",
    "GMRAIRPORT","IDEA","JIOFIN","NYKAA","PATANJALI","YESBANK","BSE","GODFRYPHLP",
    "FEDERALBNK","RADICO","BHEL","CROMPTON","PRESTIGE","VMM","PNBHOUSING","WAAREEENER",
    "NBCC","RVNL","BANDHANBNK","CONCOR","BHARTIARTL","SBICARD","UNOMINDA","SUPREMEIND",
    "VBL","KEI","EXIDEIND","UNIONBANK","HAL","DIXON","AUROPHARMA","JUBLFOOD","PHOENIXLTD",
    "OFSS","CAMS","DABUR","ONGC","RELIANCE","TVSMOTOR","CDSL","DMART","HAVELLS","IEX",
    "LUPIN","MOTHERSON","INDUSTOWER","NAM-INDIA","HYUNDAI","MAZDOCK","BAJAJ-AUTO",
    "DRREDDY","TITAN","ANGELONE","BOSCHLTD","PAYTM","COLPAL","NHPC","TIINDIA","NMDC",
    "TECHM","SRF","PETRONET","COALINDIA","AXISBANK","NUVAMA","SWIGGY","HINDUNILVR",
    "INDHOTEL","GRASIM","PERSISTENT","BDL","DIVISLAB","GAIL","JSWENERGY","LAURUSLABS",
    "MOTILALOFS","IDFCFIRSTB","POLYCAB","COCHINSHIP","KFINTECH","SHREECEM","VOLTAS",
    "ZYDUSLIFE","GODREJCP","APOLLOHOSP","BEL","BLUESTARCO","GLENMARK","MARICO",
    "ADANIPORTS","BAJAJHLDNG","BRITANNIA","CGPOWER","CUMMINSIND","DELHIVERY","ITC",
    "JINDALSTEL","SBILIFE","SOLARINDS","AMBER","POWERGRID","TATASTEEL","NTPC",
    "PREMIERENE","SUNPHARMA","FORTIS","ASTRAL","MUTHOOTFIN","SAIL","OIL","MAXHEALTH",
    "PAGEIND","JSWSTEEL","TORNTPHARM","MFSL","MCX","LICI","ETERNAL","HINDALCO","VEDL",
    "MANAPPURAM","IRFC","NATIONALUM","HINDZINC",
]

# Manual Yahoo ticker fixes for any symbol whose .NS form is wrong.
# Add entries here as you discover them, e.g. "TMPV": "TATAMOTORS.NS"
SYMBOL_OVERRIDES: dict[str, str] = {}

PLANETS = [
    ("Sun",     swe.SUN,     False),
    ("Moon",    swe.MOON,    False),
    ("Mercury", swe.MERCURY, True),
    ("Venus",   swe.VENUS,   True),
    ("Mars",    swe.MARS,    True),
    ("Jupiter", swe.JUPITER, True),
    ("Saturn",  swe.SATURN,  True),
    ("Uranus",  swe.URANUS,  True),
    ("Neptune", swe.NEPTUNE, True),
    ("Pluto",   swe.PLUTO,   True),
    # Rahu / Ketu handled separately (node id chosen by sidebar toggle)
]

SIGNS = ["Ari","Tau","Gem","Can","Leo","Vir","Lib","Sco","Sag","Cap","Aqu","Pis"]

# Canonical special-degree sets (identical to degrees.csv, generated so the
# app is self-contained and never depends on the upload path).
GANN_LEVELS   = [round(22.5 * k, 4) for k in range(16)]          # 0 .. 337.5
SQUARE_LEVELS = [n * n for n in range(19) if n * n <= 360]        # 0,1,4,...,324
CUBE_LEVELS   = [n ** 3 for n in range(8) if n ** 3 <= 360]       # 0,1,8,...,343

# Highlight colours (dark-theme friendly rgba)
C_GANN   = "rgba(56,189,248,0.22)"    # cyan
C_SQUARE = "rgba(251,191,36,0.22)"    # amber
C_CUBE   = "rgba(244,114,182,0.22)"   # magenta
C_CONJ   = "rgba(74,222,128,0.25)"    # green (geo == helio)
C_PLANET = "rgba(167,139,250,0.28)"   # violet (price-deg meets planet)


# --------------------------------------------------------------------------- #
#  PURE LOGIC  (importable / testable without a Streamlit runtime)
# --------------------------------------------------------------------------- #
def julday_from_ist(dt_ist: datetime) -> float:
    ut = dt_ist.astimezone(timezone.utc)
    return swe.julday(ut.year, ut.month, ut.day,
                      ut.hour + ut.minute / 60 + ut.second / 3600)


def circ_dist(a: float, b: float) -> float:
    """Smallest angular separation between two longitudes (0..180)."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def nearest_level(lon: float, levels: list[float], orb: float):
    """Return (level, distance) of the closest level within orb, else None."""
    best, bd = None, 1e9
    for lv in levels:
        d = circ_dist(lon, lv)
        if d < bd:
            best, bd = lv, d
    if best is not None and bd <= orb:
        return best, bd
    return None


def fmt_sign(lon: float) -> str:
    si = int(lon // 30) % 12
    deg = lon - si * 30
    d = int(deg)
    m = int(round((deg - d) * 60))
    if m == 60:
        d, m = d + 1, 0
    return f"{SIGNS[si]} {d:02d}\u00b0{m:02d}'"


def price_to_degree(price: float) -> float:
    """GANN Square-of-9 angle: ((sqrt(P)-1)*180) mod 360. price=1 -> 0deg."""
    if price is None or price <= 0:
        return float("nan")
    return ((np.sqrt(price) - 1.0) * 180.0) % 360.0


def compute_planets(dt_ist: datetime, sidereal: bool, true_node: bool) -> pd.DataFrame:
    """Return a DataFrame of planetary longitudes (geo & helio), speed, retro."""
    jd = julday_from_ist(dt_ist)
    base = swe.FLG_MOSEPH | swe.FLG_SPEED
    if sidereal:
        swe.set_sid_mode(swe.SIDM_LAHIRI)
        base |= swe.FLG_SIDEREAL

    rows = []
    for name, pid, has_helio in PLANETS:
        geo = swe.calc_ut(jd, pid, base)[0]
        glon, gspeed = geo[0] % 360.0, geo[3]
        hlon = np.nan
        if has_helio:
            hel = swe.calc_ut(jd, pid, base | swe.FLG_HELCTR)[0]
            hlon = hel[0] % 360.0
        rows.append(dict(Planet=name, Geo=glon, Helio=hlon,
                         Speed=gspeed, Retro=gspeed < 0))

    # Lunar nodes
    node_id = swe.TRUE_NODE if true_node else swe.MEAN_NODE
    nd = swe.calc_ut(jd, node_id, base)[0]
    rahu, nspeed = nd[0] % 360.0, nd[3]
    rows.append(dict(Planet="Rahu", Geo=rahu, Helio=np.nan,
                     Speed=nspeed, Retro=nspeed < 0))
    rows.append(dict(Planet="Ketu", Geo=(rahu + 180.0) % 360.0, Helio=np.nan,
                     Speed=nspeed, Retro=nspeed < 0))
    return pd.DataFrame(rows)


def period_levels(d: pd.DataFrame, today) -> dict:
    """Prev month/week/day H/L (+dates) from a daily OHLC frame."""
    d = d.dropna(subset=["High", "Low"]).copy()
    if d.empty:
        return {}
    d.index = pd.to_datetime(d.index).tz_localize(None)
    idx = d.index.normalize()
    out: dict[str, tuple] = {}

    # previous trading day (last bar strictly before today)
    pday = d[idx.date < today]
    if len(pday):
        r = pday.iloc[-1]
        dt = pday.index[-1].date()
        out["PDH"] = (float(r["High"]), dt)
        out["PDL"] = (float(r["Low"]), dt)

    # previous calendar week (Mon-anchored)
    monday = today - timedelta(days=today.weekday())
    pw = d[(idx.date >= monday - timedelta(days=7)) & (idx.date <= monday - timedelta(days=1))]
    if len(pw):
        out["PWH"] = (float(pw["High"].max()), pw["High"].idxmax().date())
        out["PWL"] = (float(pw["Low"].min()),  pw["Low"].idxmin().date())

    # previous calendar month
    pm = today.replace(day=1) - timedelta(days=1)
    pmm = d[(d.index.year == pm.year) & (d.index.month == pm.month)]
    if len(pmm):
        out["PMH"] = (float(pmm["High"].max()), pmm["High"].idxmax().date())
        out["PML"] = (float(pmm["Low"].min()),  pmm["Low"].idxmin().date())

    # last traded price (most recent close)
    out["LTP"] = (float(d["Close"].iloc[-1]), d.index[-1].date())
    return out


def yahoo_ticker(sym: str) -> str:
    sym = sym.strip().upper()
    return SYMBOL_OVERRIDES.get(sym, f"{sym}.NS")


# --------------------------------------------------------------------------- #
#  STREAMLIT-ONLY  (data loading w/ cache + UI)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple, period: str) -> dict:
    """Chunked batch download. Returns {'data': {ticker: df}, 'failed': [...]}"""
    data, failed = {}, []
    step = 40
    for i in range(0, len(tickers), step):
        part = list(tickers[i:i + step])
        try:
            raw = yf.download(part, period=period, interval="1d",
                              group_by="ticker", auto_adjust=False,
                              progress=False, threads=True)
        except Exception:
            raw = None
        for t in part:
            try:
                if raw is None:
                    failed.append(t); continue
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                if d is None or d.dropna(subset=["High", "Low"]).empty:
                    failed.append(t); continue
                data[t] = d
            except Exception:
                failed.append(t)
    return {"data": data, "failed": failed}


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');
        html, body, [class*="css"], .stApp { background:#0a0e14; color:#c9d1d9; }
        .stApp { font-family:'DM Mono', ui-monospace, monospace; }
        h1,h2,h3,h4 { font-family:'Syne', sans-serif !important; letter-spacing:.5px;
                      color:#e6edf3 !important; }
        h1 { font-weight:800; }
        .stDataFrame, .stTable { font-family:'DM Mono', monospace !important; }
        section[data-testid="stSidebar"] { background:#070a0f; border-right:1px solid #1c2330; }
        .pmeta { color:#6e7681; font-size:0.80rem; }
        .legend span { display:inline-block; padding:2px 9px; margin:2px 6px 2px 0;
                       border-radius:4px; font-size:0.74rem; }
        table.astro { border-collapse:collapse; width:100%; font-size:0.82rem; }
        table.astro th { background:#11161f; color:#8b949e; text-align:right;
                         padding:6px 10px; border-bottom:1px solid #222b38;
                         font-weight:500; position:sticky; top:0; }
        table.astro td { padding:5px 10px; border-bottom:1px solid #161b22;
                         text-align:right; white-space:nowrap; }
        table.astro td.lft { text-align:left; color:#e6edf3; font-weight:500; }
        .retro { color:#f85149; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hit_color(lon, want_gann, want_sq, want_cube, orb):
    """Return (css_bg, label) for the highest-priority special-level hit."""
    if np.isnan(lon):
        return "", ""
    if want_gann:
        h = nearest_level(lon, GANN_LEVELS, orb)
        if h:
            return C_GANN, f"G{h[0]:g}\u00b0({h[1]:.2f})"
    if want_sq:
        h = nearest_level(lon, SQUARE_LEVELS, orb)
        if h:
            return C_SQUARE, f"Sq{h[0]:g}({h[1]:.2f})"
    if want_cube:
        h = nearest_level(lon, CUBE_LEVELS, orb)
        if h:
            return C_CUBE, f"Cb{h[0]:g}({h[1]:.2f})"
    return "", ""


def render_planet_panel(pf: pd.DataFrame, want_gann, want_sq, want_cube,
                        orb, conj_orb):
    """Hand-built HTML table for full font + colour control."""
    rows_html = []
    for _, r in pf.iterrows():
        geo, hel = r["Geo"], r["Helio"]
        g_bg, g_lbl = hit_color(geo, want_gann, want_sq, want_cube, orb)
        h_bg, h_lbl = hit_color(hel, want_gann, want_sq, want_cube, orb)

        # geo == helio confluence
        conj_bg = ""
        conj_txt = "--"
        if not np.isnan(hel):
            dd = circ_dist(geo, hel)
            conj_txt = f"{dd:.2f}"
            if dd <= conj_orb:
                conj_bg = C_CONJ

        retro = "<span class='retro'>R</span>" if r["Retro"] else ""
        helio_cell = "--" if np.isnan(hel) else f"{hel:.3f}"
        tags = " · ".join(t for t in (
            (("Geo→" + g_lbl) if g_lbl else ""),
            (("Helio→" + h_lbl) if h_lbl else ""),
        ) if t)

        rows_html.append(
            f"<tr>"
            f"<td class='lft'>{r['Planet']} {retro}</td>"
            f"<td style='background:{g_bg}'>{geo:.3f}</td>"
            f"<td class='lft'>{fmt_sign(geo)}</td>"
            f"<td style='background:{h_bg}'>{helio_cell}</td>"
            f"<td style='background:{conj_bg}'>{conj_txt}</td>"
            f"<td>{r['Speed']:+.4f}</td>"
            f"<td class='lft' style='color:#6e7681'>{tags}</td>"
            f"</tr>"
        )

    table = (
        "<table class='astro'><thead><tr>"
        "<th style='text-align:left'>Planet</th><th>Geo \u00b0</th>"
        "<th style='text-align:left'>Geo sign</th><th>Helio \u00b0</th>"
        "<th>\u0394 Geo\u2013Helio</th><th>Speed \u00b0/day</th>"
        "<th style='text-align:left'>Level hits</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
    )
    st.markdown(table, unsafe_allow_html=True)


def build_stock_table(price_pack, today, show_deg, pf,
                      want_gann, want_sq, want_cube, orb, planet_orb):
    """Return (display_df, style_masks) for the stock S/R table."""
    data = price_pack["data"]
    geo_degs = pf["Geo"].tolist()
    geo_names = pf["Planet"].tolist()

    disp_rows, color_rows, order = [], [], []
    LEVELS = [("PMH", "PM High"), ("PML", "PM Low"),
              ("PWH", "PW High"), ("PWL", "PW Low"),
              ("PDH", "PD High"), ("PDL", "PD Low")]

    for tkr, d in data.items():
        sym = tkr[:-3] if tkr.endswith(".NS") else tkr
        lv = period_levels(d, today)
        if not lv:
            continue
        row, crow = {"Symbol": sym}, {}
        ltp = lv.get("LTP", (np.nan, None))[0]
        row["LTP"] = f"{ltp:,.2f}" if not np.isnan(ltp) else "--"

        for key, label in LEVELS:
            if key in lv:
                val, dt = lv[key]
                row[label] = f"{val:,.2f} \u00b7 {dt:%d-%b}"
                if show_deg:
                    deg = price_to_degree(val)
                    # special-level marker
                    _, sp_lbl = hit_color(deg, want_gann, want_sq, want_cube, orb)
                    mark = "\u25b2" if sp_lbl else ""
                    # planet resonance
                    pmatch, pd_ = "", 1e9
                    for nm, gd in zip(geo_names, geo_degs):
                        dd = circ_dist(deg, gd)
                        if dd < pd_:
                            pmatch, pd_ = nm, dd
                    hit_planet = pd_ <= planet_orb
                    row[f"{label} \u00b0"] = (
                        f"{deg:6.2f}{mark}" +
                        (f" {pmatch[:3]}" if hit_planet else "")
                    )
                    if hit_planet:
                        crow[f"{label} \u00b0"] = C_PLANET
                    elif sp_lbl:
                        crow[f"{label} \u00b0"] = (
                            C_GANN if sp_lbl.startswith("G")
                            else C_SQUARE if sp_lbl.startswith("Sq")
                            else C_CUBE
                        )
            else:
                row[label] = "--"
                if show_deg:
                    row[f"{label} \u00b0"] = "--"
        disp_rows.append(row)
        color_rows.append(crow)
        order.append(sym)

    if not disp_rows:
        return pd.DataFrame(), {}

    df = pd.DataFrame(disp_rows).set_index("Symbol")
    cmask = pd.DataFrame(color_rows, index=order).reindex(
        columns=df.columns).fillna("")
    return df, cmask


def style_stock(df: pd.DataFrame, cmask: pd.DataFrame):
    def _css(_):
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for c in df.columns:
            if c in cmask.columns:
                out[c] = cmask[c].map(lambda v: f"background-color:{v}" if v else "")
        return out
    return df.style.apply(_css, axis=None)


# --------------------------------------------------------------------------- #
#  APP
# --------------------------------------------------------------------------- #
def run_app():
    st.set_page_config(page_title="Astro-Levels Terminal",
                       layout="wide", page_icon="\u2641")
    inject_css()

    st.title("\u2641  ASTRO-LEVELS TERMINAL")
    st.markdown(
        "<div class='pmeta'>NSE F&O previous-period S/R \u00d7 live planetary "
        "longitudes (geo + helio) \u00d7 GANN / square / cube degree confluence"
        "</div>", unsafe_allow_html=True)

    # ----- sidebar ----------------------------------------------------------
    with st.sidebar:
        st.header("Ephemeris")
        zod = st.radio("Zodiac", ["Tropical", "Sidereal (Lahiri)"], index=0)
        sidereal = zod.startswith("Sidereal")
        true_node = st.toggle("True node (else Mean)", value=False)

        c1, c2 = st.columns(2)
        now_ist = datetime.now(IST)
        d_in = c1.date_input("Date (IST)", now_ist.date())
        t_in = c2.time_input("Time (IST)", now_ist.time())
        dt_ist = datetime.combine(d_in, t_in, tzinfo=IST)

        st.header("Degree highlights")
        sel = st.multiselect("Special-level sets",
                             ["GANN 22.5\u00b0", "Square (n\u00b2)", "Cube (n\u00b3)"],
                             default=["GANN 22.5\u00b0", "Square (n\u00b2)", "Cube (n\u00b3)"])
        want_gann = "GANN 22.5\u00b0" in sel
        want_sq = "Square (n\u00b2)" in sel
        want_cube = "Cube (n\u00b3)" in sel
        orb = st.slider("Level orb (\u00b0)", 0.1, 5.0, 1.0, 0.1)
        conj_orb = st.slider("Geo\u2248Helio orb (\u00b0)", 0.1, 5.0, 1.0, 0.1)

        st.header("Stocks")
        up = st.file_uploader("Symbol list CSV (1 column)", type=["csv"])
        period = st.selectbox("History window",
                              ["6mo", "1y", "2y"], index=1)
        show_deg = st.toggle("GANN price\u2192degree bridge", value=False,
                             help="Convert each H/L price to a 0-360\u00b0 wheel "
                                  "value and flag planet resonance.")
        planet_orb = st.slider("Price\u2192deg planet orb (\u00b0)",
                               0.1, 5.0, 1.0, 0.1, disabled=not show_deg)
        limit = st.number_input("Limit symbols (0 = all)", 0, 300, 0, 10)
        go = st.button("Fetch / refresh prices", type="primary",
                       width='stretch')

    # ----- symbols ----------------------------------------------------------
    if up is not None:
        raw = pd.read_csv(up, header=None).iloc[:, 0].astype(str)
        syms = [s.strip().upper() for s in raw
                if s.strip() and s.strip().upper() not in ("SYMBOL", "NAN")]
    else:
        syms = list(DEFAULT_SYMBOLS)
    if limit and limit > 0:
        syms = syms[:limit]
    tickers = tuple(yahoo_ticker(s) for s in syms)

    # ----- planetary panel --------------------------------------------------
    pf = compute_planets(dt_ist, sidereal, true_node)

    st.subheader("Planetary degrees")
    st.markdown(
        "<div class='legend'>"
        f"<span style='background:{C_GANN}'>GANN 22.5\u00b0</span>"
        f"<span style='background:{C_SQUARE}'>Square n\u00b2</span>"
        f"<span style='background:{C_CUBE}'>Cube n\u00b3</span>"
        f"<span style='background:{C_CONJ}'>Geo\u2248Helio</span>"
        f"<span class='pmeta'>{dt_ist:%Y-%m-%d %H:%M} IST &nbsp; "
        f"{'Sidereal/Lahiri' if sidereal else 'Tropical'}</span>"
        "</div>", unsafe_allow_html=True)
    render_planet_panel(pf, want_gann, want_sq, want_cube, orb, conj_orb)

    st.divider()

    # ----- stock table ------------------------------------------------------
    st.subheader("F&O previous-period levels")
    today = datetime.now(IST).date()

    if "pack" not in st.session_state:
        st.session_state.pack = None
    if go or st.session_state.pack is None:
        with st.spinner(f"Downloading {len(tickers)} symbols from Yahoo \u2026"):
            st.session_state.pack = load_prices(tickers, period)
    pack = st.session_state.pack

    n_ok = len(pack["data"])
    st.markdown(
        f"<div class='pmeta'>{n_ok}/{len(tickers)} symbols resolved &nbsp;|&nbsp; "
        f"reference day {today:%a %d-%b-%Y} &nbsp;|&nbsp; values = price \u00b7 print-date"
        "</div>", unsafe_allow_html=True)

    if show_deg:
        st.markdown(
            "<div class='legend'>"
            f"<span style='background:{C_PLANET}'>price\u00b0 \u2248 planet</span>"
            f"<span style='background:{C_GANN}'>\u25b2 special level</span>"
            "</div>", unsafe_allow_html=True)

    df, cmask = build_stock_table(pack, today, show_deg, pf,
                                  want_gann, want_sq, want_cube, orb, planet_orb)
    if df.empty:
        st.warning("No usable price data \u2014 try Fetch / refresh.")
    else:
        styler = style_stock(df, cmask)
        st.dataframe(styler, width='stretch', height=620)

        csv = df.to_csv().encode()
        st.download_button("Download table (CSV)", csv,
                           file_name=f"fno_levels_{today:%Y%m%d}.csv",
                           mime="text/csv")

    if pack["failed"]:
        with st.expander(f"Unresolved symbols ({len(pack['failed'])})"):
            st.write(", ".join(t[:-3] if t.endswith('.NS') else t
                               for t in pack["failed"]))
            st.caption("Patch SYMBOL_OVERRIDES at the top of the script "
                       "(e.g. recently demerged / renamed tickers).")

    st.caption("Moshier ephemeris (no data files). Heliocentric shown only for "
               "true planets. Not investment advice.")


if __name__ == "__main__":
    run_app()
