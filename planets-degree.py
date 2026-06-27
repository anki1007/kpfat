"""
ASTRO-LEVELS TERMINAL  (pure-Python / skyfield build)
=====================================================================
Identical features to the Swiss Ephemeris build, but with NO native
extension to compile — so it deploys on any Python (incl. 3.14) with a
plain `git push`, and never hits the pyswisseph `_ZSt7nothrow` link error.

EPHEMERIS PRECISION (validated against Swiss Ephemeris / Moshier):
    geocentric planets   <= 0.35 arcsec   (0.0001 deg)
    heliocentric planets <= 0.34 arcsec   (0.0001 deg)
    mean lunar node      <= 17   arcsec   (0.0048 deg)
    Lahiri ayanamsa      <  1    arcsec
All far below the 1 deg GANN/square/cube orbs.

LIMITATION vs the swisseph build: lunar nodes are MEAN node only
(de421 carries no node data; true node needs Swiss Ephemeris).

RUN
    pip install streamlit yfinance pandas numpy skyfield
    streamlit run planets-degree.py

EPHEMERIS FILE (de421.bsp, ~17 MB, covers 1899-2053):
    On first run skyfield downloads it automatically to a temp dir.
    For zero cold-start latency on Streamlit Cloud, optionally commit
    de421.bsp next to this script (it will be used directly):
        curl -L -o de421.bsp \\
          https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp
=====================================================================
"""

from __future__ import annotations
import os
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from skyfield.api import Loader, load_file

# --------------------------------------------------------------------------- #
#  CONSTANTS
# --------------------------------------------------------------------------- #
IST = timezone(timedelta(hours=5, minutes=30))

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
SYMBOL_OVERRIDES: dict[str, str] = {}

# (display name, skyfield body key, heliocentric-applicable)
BODIES = [
    ("Sun",     "sun",                False),
    ("Moon",    "moon",               False),
    ("Mercury", "mercury",            True),
    ("Venus",   "venus",              True),
    ("Mars",    "mars",               True),
    ("Jupiter", "jupiter barycenter", True),
    ("Saturn",  "saturn barycenter",  True),
    ("Uranus",  "uranus barycenter",  True),
    ("Neptune", "neptune barycenter", True),
    ("Pluto",   "pluto barycenter",   True),
]
SIGNS = ["Ari","Tau","Gem","Can","Leo","Vir","Lib","Sco","Sag","Cap","Aqu","Pis"]

GANN_LEVELS   = [round(22.5 * k, 4) for k in range(16)]
SQUARE_LEVELS = [n * n for n in range(19) if n * n <= 360]
CUBE_LEVELS   = [n ** 3 for n in range(8) if n ** 3 <= 360]

C_GANN   = "rgba(56,189,248,0.22)"
C_SQUARE = "rgba(251,191,36,0.22)"
C_CUBE   = "rgba(244,114,182,0.22)"
C_CONJ   = "rgba(74,222,128,0.25)"
C_PLANET = "rgba(167,139,250,0.28)"


# --------------------------------------------------------------------------- #
#  EPHEMERIS  (skyfield)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading ephemeris \u2026")
def load_ephemeris():
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(tempfile.gettempdir(), "skyfield-data")
    os.makedirs(cache, exist_ok=True)
    ld = Loader(cache)
    ts = ld.timescale()                       # built-in leap seconds, no download
    local = os.path.join(here, "de421.bsp")   # committed file wins (no download)
    eph = load_file(local) if os.path.exists(local) else ld("de421.bsp")
    return ts, eph


def lahiri_ayanamsa(tt_jd: float) -> float:
    """Lahiri ayanamsa (deg); matches Swiss Ephemeris SIDM_LAHIRI to <1 arcsec."""
    T = (tt_jd - 2451545.0) / 36525.0
    return 23.857084 + 1.396971 * T + 0.0003086 * T * T


def mean_node_deg(tt_jd: float) -> float:
    """Mean longitude of ascending lunar node (deg), equinox of date — Meeus."""
    T = (tt_jd - 2451545.0) / 36525.0
    return (125.0445479 - 1934.1362891 * T + 0.0020754 * T * T
            + T ** 3 / 467441.0 - T ** 4 / 60616000.0) % 360.0


# --------------------------------------------------------------------------- #
#  PURE LOGIC
# --------------------------------------------------------------------------- #
def circ_dist(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def nearest_level(lon: float, levels: list[float], orb: float):
    best, bd = None, 1e9
    for lv in levels:
        d = circ_dist(lon, lv)
        if d < bd:
            best, bd = lv, d
    return (best, bd) if best is not None and bd <= orb else None


def fmt_sign(lon: float) -> str:
    si = int(lon // 30) % 12
    deg = lon - si * 30
    d = int(deg)
    m = int(round((deg - d) * 60))
    if m == 60:
        d, m = d + 1, 0
    return f"{SIGNS[si]} {d:02d}\u00b0{m:02d}'"


def price_to_degree(price: float) -> float:
    if price is None or price <= 0:
        return float("nan")
    return ((np.sqrt(price) - 1.0) * 180.0) % 360.0


def compute_planets(dt_ist: datetime, sidereal: bool, ts, eph) -> pd.DataFrame:
    """Geo + helio ecliptic longitudes (equinox of date) via skyfield."""
    from skyfield.api import wgs84  # noqa: F401  (kept for parity / future use)
    ut = dt_ist.astimezone(timezone.utc)
    t = ts.from_datetime(ut)
    earth, sun = eph["earth"], eph["sun"]
    ayan = lahiri_ayanamsa(t.tt) if sidereal else 0.0

    def geo_trop(key, tt):
        tx = ts.tt_jd(tt)
        return earth.at(tx).observe(eph[key]).apparent().ecliptic_latlon(epoch=tx)[1].degrees

    def helio_trop(key):
        return sun.at(t).observe(eph[key]).ecliptic_latlon(epoch=t)[1].degrees

    def shift(x):
        return (x - ayan) % 360.0

    rows = []
    for name, key, has_helio in BODIES:
        glon_trop = geo_trop(key, t.tt)
        # daily speed via 1-day central difference (signed, wrap-safe)
        speed = ((geo_trop(key, t.tt + 0.5) - geo_trop(key, t.tt - 0.5) + 540) % 360) - 180
        hlon = shift(helio_trop(key)) if has_helio else np.nan
        rows.append(dict(Planet=name, Geo=shift(glon_trop), Helio=hlon,
                         Speed=speed, Retro=speed < 0))

    # Mean node (Rahu) + Ketu
    rahu_trop = mean_node_deg(t.tt)
    nspeed = ((mean_node_deg(t.tt + 0.5) - mean_node_deg(t.tt - 0.5) + 540) % 360) - 180
    rahu = shift(rahu_trop)
    rows.append(dict(Planet="Rahu", Geo=rahu, Helio=np.nan, Speed=nspeed, Retro=nspeed < 0))
    rows.append(dict(Planet="Ketu", Geo=(rahu + 180.0) % 360.0, Helio=np.nan,
                     Speed=nspeed, Retro=nspeed < 0))
    return pd.DataFrame(rows)


def period_levels(d: pd.DataFrame, today) -> dict:
    d = d.dropna(subset=["High", "Low"]).copy()
    if d.empty:
        return {}
    d.index = pd.to_datetime(d.index).tz_localize(None)
    idx = d.index.normalize()
    out: dict[str, tuple] = {}

    pday = d[idx.date < today]
    if len(pday):
        r = pday.iloc[-1]; dt = pday.index[-1].date()
        out["PDH"] = (float(r["High"]), dt)
        out["PDL"] = (float(r["Low"]), dt)

    monday = today - timedelta(days=today.weekday())
    pw = d[(idx.date >= monday - timedelta(days=7)) & (idx.date <= monday - timedelta(days=1))]
    if len(pw):
        out["PWH"] = (float(pw["High"].max()), pw["High"].idxmax().date())
        out["PWL"] = (float(pw["Low"].min()),  pw["Low"].idxmin().date())

    pm = today.replace(day=1) - timedelta(days=1)
    pmm = d[(d.index.year == pm.year) & (d.index.month == pm.month)]
    if len(pmm):
        out["PMH"] = (float(pmm["High"].max()), pmm["High"].idxmax().date())
        out["PML"] = (float(pmm["Low"].min()),  pmm["Low"].idxmin().date())

    out["LTP"] = (float(d["Close"].iloc[-1]), d.index[-1].date())
    return out


def yahoo_ticker(sym: str) -> str:
    sym = sym.strip().upper()
    return SYMBOL_OVERRIDES.get(sym, f"{sym}.NS")


# --------------------------------------------------------------------------- #
#  STREAMLIT-ONLY
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tickers: tuple, period: str) -> dict:
    data, failed = {}, []
    step = 40
    for i in range(0, len(tickers), step):
        part = list(tickers[i:i + step])
        try:
            raw = yf.download(part, period=period, interval="1d", group_by="ticker",
                              auto_adjust=False, progress=False, threads=True)
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
        h1,h2,h3,h4 { font-family:'Syne', sans-serif !important; letter-spacing:.5px; color:#e6edf3 !important; }
        h1 { font-weight:800; }
        .stDataFrame, .stTable { font-family:'DM Mono', monospace !important; }
        section[data-testid="stSidebar"] { background:#070a0f; border-right:1px solid #1c2330; }
        .pmeta { color:#6e7681; font-size:0.80rem; }
        .legend span { display:inline-block; padding:2px 9px; margin:2px 6px 2px 0; border-radius:4px; font-size:0.74rem; }
        table.astro { border-collapse:collapse; width:100%; font-size:0.82rem; }
        table.astro th { background:#11161f; color:#8b949e; text-align:right; padding:6px 10px; border-bottom:1px solid #222b38; font-weight:500; position:sticky; top:0; }
        table.astro td { padding:5px 10px; border-bottom:1px solid #161b22; text-align:right; white-space:nowrap; }
        table.astro td.lft { text-align:left; color:#e6edf3; font-weight:500; }
        .retro { color:#f85149; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hit_color(lon, want_gann, want_sq, want_cube, orb):
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


def render_planet_panel(pf, want_gann, want_sq, want_cube, orb, conj_orb):
    rows_html = []
    for _, r in pf.iterrows():
        geo, hel = r["Geo"], r["Helio"]
        g_bg, g_lbl = hit_color(geo, want_gann, want_sq, want_cube, orb)
        h_bg, h_lbl = hit_color(hel, want_gann, want_sq, want_cube, orb)
        conj_bg, conj_txt = "", "--"
        if not np.isnan(hel):
            dd = circ_dist(geo, hel)
            conj_txt = f"{dd:.2f}"
            if dd <= conj_orb:
                conj_bg = C_CONJ
        retro = "<span class='retro'>R</span>" if r["Retro"] else ""
        helio_cell = "--" if np.isnan(hel) else f"{hel:.3f}"
        tags = " \u00b7 ".join(t for t in (
            (("Geo\u2192" + g_lbl) if g_lbl else ""),
            (("Helio\u2192" + h_lbl) if h_lbl else ""),
        ) if t)
        rows_html.append(
            f"<tr><td class='lft'>{r['Planet']} {retro}</td>"
            f"<td style='background:{g_bg}'>{geo:.3f}</td>"
            f"<td class='lft'>{fmt_sign(geo)}</td>"
            f"<td style='background:{h_bg}'>{helio_cell}</td>"
            f"<td style='background:{conj_bg}'>{conj_txt}</td>"
            f"<td>{r['Speed']:+.4f}</td>"
            f"<td class='lft' style='color:#6e7681'>{tags}</td></tr>"
        )
    st.markdown(
        "<table class='astro'><thead><tr>"
        "<th style='text-align:left'>Planet</th><th>Geo \u00b0</th>"
        "<th style='text-align:left'>Geo sign</th><th>Helio \u00b0</th>"
        "<th>\u0394 Geo\u2013Helio</th><th>Speed \u00b0/day</th>"
        "<th style='text-align:left'>Level hits</th></tr></thead><tbody>"
        + "".join(rows_html) + "</tbody></table>",
        unsafe_allow_html=True)


def build_stock_table(price_pack, today, show_deg, pf,
                      want_gann, want_sq, want_cube, orb, planet_orb):
    data = price_pack["data"]
    geo_degs = pf["Geo"].tolist()
    geo_names = pf["Planet"].tolist()
    LEVELS = [("PMH", "PM High"), ("PML", "PM Low"),
              ("PWH", "PW High"), ("PWL", "PW Low"),
              ("PDH", "PD High"), ("PDL", "PD Low")]
    disp_rows, color_rows, order = [], [], []
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
                    _, sp_lbl = hit_color(deg, want_gann, want_sq, want_cube, orb)
                    mark = "\u25b2" if sp_lbl else ""
                    pmatch, pd_ = "", 1e9
                    for nm, gd in zip(geo_names, geo_degs):
                        dd = circ_dist(deg, gd)
                        if dd < pd_:
                            pmatch, pd_ = nm, dd
                    hit_planet = pd_ <= planet_orb
                    row[f"{label} \u00b0"] = f"{deg:6.2f}{mark}" + (f" {pmatch[:3]}" if hit_planet else "")
                    if hit_planet:
                        crow[f"{label} \u00b0"] = C_PLANET
                    elif sp_lbl:
                        crow[f"{label} \u00b0"] = (C_GANN if sp_lbl.startswith("G")
                                                   else C_SQUARE if sp_lbl.startswith("Sq") else C_CUBE)
            else:
                row[label] = "--"
                if show_deg:
                    row[f"{label} \u00b0"] = "--"
        disp_rows.append(row); color_rows.append(crow); order.append(sym)
    if not disp_rows:
        return pd.DataFrame(), {}
    df = pd.DataFrame(disp_rows).set_index("Symbol")
    cmask = pd.DataFrame(color_rows, index=order).reindex(columns=df.columns).fillna("")
    return df, cmask


def style_stock(df, cmask):
    def _css(_):
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for c in df.columns:
            if c in cmask.columns:
                out[c] = cmask[c].map(lambda v: f"background-color:{v}" if v else "")
        return out
    return df.style.apply(_css, axis=None)


def run_app():
    st.set_page_config(page_title="Astro-Levels Terminal", layout="wide", page_icon="\u2641")
    inject_css()
    st.title("\u2641  ASTRO-LEVELS TERMINAL")
    st.markdown("<div class='pmeta'>NSE F&O previous-period S/R \u00d7 live planetary "
                "longitudes (geo + helio) \u00d7 GANN / square / cube confluence "
                "\u00b7 skyfield engine</div>", unsafe_allow_html=True)

    try:
        ts, eph = load_ephemeris()
    except Exception as e:
        st.error("Could not load the de421 ephemeris. On Streamlit Cloud the first "
                 "run downloads it automatically; if egress is blocked, commit "
                 "`de421.bsp` next to this script. Details: " + str(e))
        st.stop()

    with st.sidebar:
        st.header("Ephemeris")
        zod = st.radio("Zodiac", ["Tropical", "Sidereal (Lahiri)"], index=0)
        sidereal = zod.startswith("Sidereal")
        st.caption("Node = mean node (true node needs the Swiss Ephemeris build).")

        # Persist the chosen instant across reruns. The widgets are KEYED and
        # seeded once from session_state; seeding them with a live
        # datetime.now() default (as before) made Streamlit treat the picker as
        # a new widget every rerun and snap it back to the current time, so
        # forward dates/times could never "stick".
        if "calc_date" not in st.session_state:
            _n = datetime.now(IST)
            st.session_state.calc_date = _n.date()
            st.session_state.calc_time = _n.time().replace(second=0, microsecond=0)

        def _reset_now():
            _n = datetime.now(IST)
            st.session_state.calc_date = _n.date()
            st.session_state.calc_time = _n.time().replace(second=0, microsecond=0)

        c1, c2 = st.columns(2)
        d_in = c1.date_input("Date (IST)", key="calc_date")
        t_in = c2.time_input("Time (IST)", key="calc_time",
                             step=timedelta(minutes=5))
        st.button("\u27f3 Reset to now", on_click=_reset_now, width='stretch')
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
        period = st.selectbox("History window", ["6mo", "1y", "2y"], index=1)
        show_deg = st.toggle("GANN price\u2192degree bridge", value=False)
        planet_orb = st.slider("Price\u2192deg planet orb (\u00b0)", 0.1, 5.0, 1.0, 0.1,
                               disabled=not show_deg)
        limit = st.number_input("Limit symbols (0 = all)", 0, 300, 0, 10)
        go = st.button("Fetch / refresh prices", type="primary", width='stretch')

    if up is not None:
        raw = pd.read_csv(up, header=None).iloc[:, 0].astype(str)
        syms = [s.strip().upper() for s in raw
                if s.strip() and s.strip().upper() not in ("SYMBOL", "NAN")]
    else:
        syms = list(DEFAULT_SYMBOLS)
    if limit and limit > 0:
        syms = syms[:limit]
    tickers = tuple(yahoo_ticker(s) for s in syms)

    pf = compute_planets(dt_ist, sidereal, ts, eph)

    st.subheader("Planetary degrees")
    st.markdown(
        "<div class='legend'>"
        f"<span style='background:{C_GANN}'>GANN 22.5\u00b0</span>"
        f"<span style='background:{C_SQUARE}'>Square n\u00b2</span>"
        f"<span style='background:{C_CUBE}'>Cube n\u00b3</span>"
        f"<span style='background:{C_CONJ}'>Geo\u2248Helio</span>"
        f"<span class='pmeta'>{dt_ist:%Y-%m-%d %H:%M} IST &nbsp; "
        f"{'Sidereal/Lahiri' if sidereal else 'Tropical'}</span></div>",
        unsafe_allow_html=True)
    render_planet_panel(pf, want_gann, want_sq, want_cube, orb, conj_orb)

    st.divider()
    st.subheader("F&O previous-period levels")
    today = datetime.now(IST).date()

    if "pack" not in st.session_state:
        st.session_state.pack = None
    if go or st.session_state.pack is None:
        with st.spinner(f"Downloading {len(tickers)} symbols from Yahoo \u2026"):
            st.session_state.pack = load_prices(tickers, period)
    pack = st.session_state.pack

    st.markdown(
        f"<div class='pmeta'>{len(pack['data'])}/{len(tickers)} symbols resolved &nbsp;|&nbsp; "
        f"reference day {today:%a %d-%b-%Y} &nbsp;|&nbsp; values = price \u00b7 print-date</div>",
        unsafe_allow_html=True)
    if show_deg:
        st.markdown(
            "<div class='legend'>"
            f"<span style='background:{C_PLANET}'>price\u00b0 \u2248 planet</span>"
            f"<span style='background:{C_GANN}'>\u25b2 special level</span></div>",
            unsafe_allow_html=True)

    df, cmask = build_stock_table(pack, today, show_deg, pf,
                                  want_gann, want_sq, want_cube, orb, planet_orb)
    if df.empty:
        st.warning("No usable price data \u2014 try Fetch / refresh.")
    else:
        st.dataframe(style_stock(df, cmask), width='stretch', height=620)
        st.download_button("Download table (CSV)", df.to_csv().encode(),
                           file_name=f"fno_levels_{today:%Y%m%d}.csv", mime="text/csv")

    if pack["failed"]:
        with st.expander(f"Unresolved symbols ({len(pack['failed'])})"):
            st.write(", ".join(t[:-3] if t.endswith('.NS') else t for t in pack["failed"]))
            st.caption("Patch SYMBOL_OVERRIDES at the top of the script.")

    st.caption("skyfield / de421 ephemeris (geo & helio matched to Swiss Ephemeris "
               "to <0.005\u00b0). Mean node only. Not investment advice.")


if __name__ == "__main__":
    run_app()
