import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(
    page_title="Flight Fare Intelligence",
    page_icon="✈️",
    layout="wide",
)

# ============================ THEME ============================
PAL = [
    "#E8C77A","#B8962E","#CBD5DC","#5C7A8C","#6FBF9F",
    "#2F7F6B","#7FB8D6","#2F6F9E","#D9825B","#B5573A",
]

THEME = dict(
    plot_template="plotly_dark",
    bg="#12151A",
    card="#1B2028",
    border="rgba(124,200,240,.20)",
    tag="rgba(124,200,240,.07)",
    text="#E8EEF5",
    muted="#A5B0BF",
    accent="#7CC8F0",
    good="#6FE0B2",
    warn="#F2C66D",
    bad="#FF9292",
    grid="rgba(232,238,245,0.07)",
    line="rgba(232,238,245,0.40)",
)

CLASS_COLORS = {
    "Economy": "#3DB8A8",
    "Premium Economy": "#5B8DEF",
    "Business": "#F2C14E",
    "First": "#F0714F",
}

COST_SCALE = ["#E9C46A", "#5BC0BE", "#3A7CA5", "#2A5C8F"]
HEAT_SCALE = ["#E9C46A", "#5BC0BE", "#3A7CA5", "#1C2541"]   # FIXED - route heatmap

CITY_CODES = {
    "Ahmedabad": "Ahm",
    "Bangalore": "Ban",
    "Chennai": "Che",
    "Delhi": "Del",
    "Hyderabad": "Hyd",
    "Kolkata": "Kol",
    "Mumbai": "Mum",
}   # short names for small screens

CITY_INFO = "<br>".join(
    f"<b>{short}</b> = {city}" for city, short in CITY_CODES.items()
)     # text of the (i) button

STOP_COLORS = {
    "Non-stop": "#8FD66B",
    "1 stop": "#B79CFF",
    "2+ stops": "#E5E7EB",
}   # scatter colours: green / violet / light grey


# ============================ CONSTANTS  ============================ 
DATA_FILES = ["Cleaned_dataset.csv", "data/Cleaned_dataset.csv"]
MODEL_FILE = "model/flight_fare_model.joblib"

CAT_FEATURES = ["Airline", "Class", "Source", "Destination", "Departure", "Arrival", "Journey_day"]
NUM_FEATURES = ["Stops", "Duration_in_hours", "Days_left"]
FEATURES = CAT_FEATURES + NUM_FEATURES

CLASS_ORDER = ["Economy", "Premium Economy", "Business", "First"]
TIME_ORDER = ["Before 6 AM", "6 AM - 12 PM", "12 PM - 6 PM", "After 6 PM"]
TIME_COLORS = dict(
    zip(
        ["Before 6 AM", "6 AM - 12 PM", "12 PM - 6 PM", "After 6 PM"],
        [PAL[6], PAL[0], PAL[8], PAL[4]],
    )
)
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STOP_LABELS = {0: "Non-stop", 1: "1 stop", 2: "2+ stops"}
MIN_ROUTE_FLIGHTS = 30


#  ============================ CSS - Colours come from THEME through CSS variables ============================ 
ROOT = ":root{" + "".join(f"--{name}:{value};" for name, value in THEME.items()) + "}"

CSS = "<style>" + ROOT + """
header[data-testid="stHeader"]{background:transparent!important}
.block-container{padding-top:4rem;padding-bottom:2.5rem!important;max-width:1400px}
[data-testid="stMainBlockContainer"]{padding-bottom:2.5rem!important}
.app-title{font-size:clamp(1.5rem,3.5vw,2.2rem);font-weight:800;line-height:1.3;color:var(--text);margin:0 0 .3rem}
.app-title span{color:var(--accent)}
.app-sub{color:var(--muted);font-size:.85rem;margin-bottom:1rem}

/* ===== MAIN KPI CARDS ===== */
.kpi-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:1rem 0 1.8rem}

.kpi-card{
 --c:var(--accent);
 position:relative;min-width:0;padding:20px 18px 17px;
 border-radius:9px;background:#191E26;
 border:1px solid #303945;
 border-top:3px solid var(--c);
 overflow:hidden;
 box-shadow:0 4px 14px -10px var(--c);
 transition:border-color .2s ease,transform .2s ease
}

.kpi-card.blue{--c:#7CC8F0}
.kpi-card.green{--c:#6FE0B2}
.kpi-card.amber{--c:#F2C66D}
.kpi-card.red{--c:#FF9292}

.kpi-label{
 color:#A5B0BF;font-size:.72rem;font-weight:700;
 text-transform:uppercase;letter-spacing:.08em;line-height:1.5
}

.kpi-value{
 color:var(--c);font-size:clamp(1.45rem,2.2vw,2rem);
 font-weight:800;margin:10px 0 7px;
 overflow-wrap:anywhere;line-height:1.2;
 font-variant-numeric:tabular-nums
}

.kpi-sub{color:#A5B0BF;font-size:.76rem;line-height:1.5}

.kpi-card:hover,
.kpi-row.hero .kpi-card:hover{
  border-color:var(--c);
  box-shadow:0 0 0 1px var(--c);
  transform:translateY(-2px);
}

/* ===== HERO KPIs: FIRST SCREEN ===== */
.kpi-row.hero{gap:16px;margin:1.2rem 0 2rem}
.kpi-row.hero .kpi-card{
 padding:23px 20px 20px;min-height:140px;
 background:#1C232D;border-color:#35404D;
 border-top:3px solid var(--c);border-radius:9px
}
.kpi-row.hero .kpi-value{
 font-size:clamp(1.55rem,2.5vw,2.2rem);margin:12px 0 8px
}
.kpi-row.hero .kpi-label{font-size:.73rem}
.kpi-row.hero .kpi-sub{font-size:.78rem}

/* ===== TABLET ===== */
@media(max-width:900px){
 .kpi-row,.kpi-row.hero{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
 .kpi-row.hero .kpi-card{min-height:125px;padding:18px 16px}
}

/* ===== MOBILE ===== */
@media(max-width:560px){
 .block-container{padding-left:1rem;padding-right:1rem}
 .kpi-row,.kpi-row.hero{gap:10px;margin:1rem 0 1.5rem}
 .kpi-card,.kpi-row.hero .kpi-card{padding:14px 12px;min-height:112px}
 .kpi-label,.kpi-row.hero .kpi-label{font-size:.61rem}
 .kpi-value,.kpi-row.hero .kpi-value{font-size:1.22rem;margin:8px 0 6px}
 .kpi-sub,.kpi-row.hero .kpi-sub{font-size:.65rem}
}

.info{position:relative;display:inline-block;margin-left:.5rem;width:1.2rem;height:1.2rem;line-height:1.1rem;
 text-align:center;font-size:.8rem;font-weight:700;border:1px solid var(--accent);border-radius:50%;
 color:var(--accent);cursor:help;vertical-align:middle;outline:none}
.info .tip{display:none;position:absolute;left:0;top:1.7rem;z-index:50;width:min(240px,80vw);padding:.6rem .8rem;
 border-radius:8px;background:var(--card);border:1px solid var(--border);box-shadow:0 8px 24px rgba(0,0,0,.45);
 color:var(--text);font-size:.8rem;font-weight:400;line-height:1.7;text-align:left}
.info:hover .tip,.info:focus .tip{display:block}

.chart-note{background:var(--tag);border:1px solid var(--border);border-radius:10px;
 padding:.7rem 1rem;margin:.2rem 0 .6rem;font-size:.9rem;line-height:1.65;color:var(--text)}
.chart-note .row{margin:.3rem 0}
.chart-note .lbl{display:block;font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
 color:var(--accent);margin-bottom:.1rem}
.chart-note b{color:var(--good);font-weight:800}
.chart-note bad{color:var(--bad);font-weight:800}

.lgd-row{display:flex;flex-wrap:wrap;justify-content:center;gap:6px 26px;padding:0 1.2rem;margin:.2rem 0 .4rem}
.lgd{display:flex;align-items:center;gap:.45rem;font-size:.78rem;color:var(--muted)}
.lgd i{display:inline-block;width:11px;height:11px;border-radius:3px;flex:none}
@media(max-width:560px){
 .lgd-row{gap:6px 0;padding:0 .6rem}
 .lgd{flex:0 0 49%;justify-content:center}      /* 2 per row, each centred -> a centred 2x2 */
}

span[data-baseweb="tag"]{background:rgba(224,180,92,.12)!important;border:1px solid rgba(224,180,92,.45);
 border-radius:8px}
span[data-baseweb="tag"] span{color:#E8E4D8!important}
span[data-baseweb="tag"] svg{fill:#E0B45C}

/* ===== RECOMMENDATIONS ===== */
.reco-head{font-size:1.05rem;font-weight:800;color:var(--text);margin:1.6rem 0 .6rem}
.reco-sub{color:var(--muted);font-size:.74rem;margin:-.3rem 0 .7rem}

.reco-row{
  display:grid;grid-template-columns:repeat(var(--n,4),minmax(0,1fr));gap:14px;
 margin-bottom:2.5rem            /* <- recommendations ke baad space */
}
.reco-card{--c:var(--accent);min-width:0;padding:16px 16px 14px;border-radius:9px;
 background:#191E26;border:1px solid #303945;border-left:3px solid var(--c)}
.reco-card.blue{--c:#7CC8F0}.reco-card.green{--c:#6FE0B2}
.reco-card.amber{--c:#F2C66D}.reco-card.red{--c:#FF9292}
.reco-title{color:var(--c);font-weight:800;font-size:.9rem;margin-bottom:6px}
.reco-text{color:var(--muted);font-size:.8rem;line-height:1.55}

/* tablet + mobile: 2 x 2 */
@media(max-width:900px){
 .reco-row{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
}
@media(max-width:560px){
 .reco-row{margin-bottom:2rem}
 .reco-card{padding:12px 11px 11px}
 .reco-title{font-size:.78rem}
 .reco-text{font-size:.7rem;line-height:1.5}
}

</style>"""


#  ============================ SMALL HELPERS ============================ 
def kpi_row(cards, hero=False):
    html = "".join(
        f'<div class="kpi-card {colour}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'
        for label, value, sub, colour in cards
    )
    st.markdown(
        f'<div class="kpi-row{" hero" if hero else ""}">{html}</div>',
        unsafe_allow_html=True,
    )


def insight(content=None):
    """Display a short takeaway directly below a chart."""
    if content:
        st.markdown(
            f'<div class="chart-note"><div class="row"><span class="lbl">🔎 Insight</span>{content}</div></div>',
            unsafe_allow_html=True,
        )


def heading(text, info=None):
    """Chart title. With `info`, a small (i) button shows the text on hover (laptop) or tap (phone)."""
    bubble = f'<span class="info" tabindex="0">i<span class="tip">{info}</span></span>' if info else ""
    st.markdown(f"#### {text} {bubble}", unsafe_allow_html=True)


def draw(fig, x_title=None, y_title=None, height=380, legend=True, margin=None):
    """Apply the shared chart style and show the chart.
    margin = optional dict that overrides the defaults, e.g. margin=dict(l=4, r=70) for charts with long labels."""
    box = dict(t=20, b=45, l=24, r=30)
    box.update(margin or {})

    fig.update_layout(
        template=THEME["plot_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=THEME["text"]),
        height=height,
        showlegend=False,
        margin=box,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True, hoverformat=",.0f")
    fig.update_yaxes(gridcolor=THEME["grid"], zeroline=False, automargin=True, hoverformat=",.0f")

    if x_title is not None:
        fig.update_xaxes(title_text=x_title or None, title_standoff=14)    # gap between tick labels and the axis title

    if y_title is not None:
        fig.update_yaxes(title_text=y_title or None, title_standoff=14)

    if legend:      # our own legend above the chart: 2 per row on a phone, one row on a laptop, never covers the chart
        items = ""
        for trace in fig.data:
            colour = trace.marker.color if isinstance(trace.marker.color, str) else trace.line.color
            items += f'<span class="lgd"><i style="background:{colour}"></i>{trace.name}</span>'
        st.markdown(f'<div class="lgd-row">{items}</div>', unsafe_allow_html=True)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def premium(frame):
    """Last-minute premium: fare within 3 days of departure vs 22+ days ahead (None if unknown)."""
    near = frame.loc[frame["Days_left"] <= 3, "Fare"].mean()
    far = frame.loc[frame["Days_left"] >= 22, "Fare"].mean()

    if pd.notna(near) and pd.notna(far) and far > 0:
        return near / far - 1

    return None


def make_rows(n=1, **values):
    """DataFrame with the exact columns the model expects (n identical rows)."""
    return pd.DataFrame([{col: values[col] for col in FEATURES}] * n, columns=FEATURES)


def default_duration(df, airline, cls, src, dst, stops):
    """Typical duration for a flight; falls back to broader groups when there is no exact match."""
    attempts = [
        {"Airline": airline, "Class": cls, "Source": src, "Destination": dst, "Stops": stops},
        {"Class": cls, "Source": src, "Destination": dst, "Stops": stops},
        {"Class": cls, "Source": src, "Destination": dst},
        {"Class": cls},
    ]

    for conditions in attempts:
        mask = np.ones(len(df), dtype=bool)

        for col, value in conditions.items():
            mask &= (df[col] == value).to_numpy()

        if mask.any():
            return float(df.loc[mask, "Duration_in_hours"].median())

    return float(df["Duration_in_hours"].median())


# ============================ DATA AND MODEL (cached) ============================ 

@st.cache_resource(show_spinner="Loading data...")     # keeps ONE copy in memory (cache_data copies it on every click)
def load_data():
    path = next((p for p in DATA_FILES if os.path.exists(p)), None)

    if path is None:
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # strip spaces by cleaning the few unique values instead of rewriting all 450k strings (saves memory)
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].map({value: str(value).strip() for value in df[col].unique()})

    numeric_cols = ["Fare", "Duration_in_hours", "Days_left"]

    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):              # only clean text such as "1,234"
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=["Date_of_journey", "Flight_code"], errors="ignore")   # not used; dropped AFTER removing duplicates

    df["Stops"] = df["Total_stops"].str.extract(r"(\d+)")[0].astype(float).fillna(0).astype(int)
    df["Route"] = (df["Source"] + " → " + df["Destination"]).astype("category").astype(object)   # shares one string per route

    return df


@st.cache_resource(show_spinner="Preparing the prediction model (first start only)...")
def load_model(_df):
    # 1) try the model saved by the notebook
    load_error = None
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            model.predict(_df[FEATURES].head(5))
            return model, "saved model (flight_fare_model.joblib)"
        except Exception as e:
            load_error = repr(e)                                           # e.g. scikit-learn version mismatch

    # 2) otherwise train the same kind of model here
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    booster = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.2,
        max_leaf_nodes=127,
        min_samples_leaf=20,
        l2_regularization=0.1,
        categorical_features=[True] * len(CAT_FEATURES) + [False] * len(NUM_FEATURES),
        random_state=42,
    )

    model = Pipeline([
        ("prep", prep),
        ("model", TransformedTargetRegressor(regressor=booster, func=np.log1p, inverse_func=np.expm1)),
    ])

    sample = _df.sample(min(len(_df), 200_000), random_state=42)
    model.fit(sample[FEATURES], sample["Fare"])

    return model, f"trained automatically (saved model failed: {load_error})" if load_error else "model trained automatically by the app"

# ============================ PAGE PARTS: header and sidebar ============================ 
def render_header():
    st.markdown(
        '<div class="app-title">✈️ Flight Fare <span>Dashboard & Predictor</span></div>'
        '<div class="app-sub">Domestic India flights · explore what drives fares and predict a fare</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(df, model_source):
    """Sidebar filters. Returns the filtered data, the same data ignoring the Class filter, and chosen classes."""
    st.sidebar.header("🎛️ Flight Filters")

    classes = [c for c in CLASS_ORDER if c in df["Class"].unique()]
    airlines = sorted(df["Airline"].unique())
    cities_from = sorted(df["Source"].unique())
    cities_to = sorted(df["Destination"].unique())
    stop_names = list(STOP_LABELS.values())

    sel_class = st.sidebar.multiselect("Class", classes, default=classes)
    sel_airline = st.sidebar.multiselect("Airline", airlines, default=airlines)
    sel_from = st.sidebar.multiselect("From", cities_from, default=cities_from)
    sel_to = st.sidebar.multiselect("To", cities_to, default=cities_to)
    sel_stops = st.sidebar.multiselect("Stops", stop_names, default=stop_names)

    stop_numbers = [n for n, label in STOP_LABELS.items() if label in sel_stops]
    others = (
        df["Airline"].isin(sel_airline)
        & df["Source"].isin(sel_from)
        & df["Destination"].isin(sel_to)
        & df["Stops"].isin(stop_numbers)
    )

    st.sidebar.divider()
    st.sidebar.caption(f"Flights in data: {len(df):,} | Airlines: {df['Airline'].nunique()}")
    st.sidebar.caption(f"Prediction model: {model_source}")

    # copy only the columns the charts use (copying all 12 columns of 445k rows on every click ran out of memory)
    keep = [
        "Airline", "Class", "Source", "Destination", "Route", "Stops", "Fare", "Days_left",
        "Departure", "Arrival", "Journey_day", "Duration_in_hours",
    ]

    return (
        df.loc[others & df["Class"].isin(sel_class), keep],
        df.loc[others, ["Airline", "Class"]],
        sel_class,
    )

def reco_head(title):
    st.markdown(f'<div class="reco-head">{title}</div>', unsafe_allow_html=True)


def reco_cards(cards, basis):
    html = "".join(
        f'<div class="reco-card {colour}"><div class="reco-title">{icon} {name}</div>'
        f'<div class="reco-text">{text}</div></div>'
        for icon, name, text, colour in cards
    )
    st.markdown(
        f'<div class="reco-sub">{basis}</div><div class="reco-row" style="--n:{len(cards)}">{html}</div>',
        unsafe_allow_html=True,
    )


def reco_by_class(recos, kind, key):
    """Heading, class selector, then cards. Independent of the sidebar filters."""
    reco_head("🎯 Smart Booking Tips")
    available = [c for c in CLASS_ORDER if recos[kind].get(c)]
    cls = st.radio("Tips for", available, horizontal=True,
                   key=f"reco_{key}", label_visibility="collapsed")
    reco_cards(recos[kind][cls], f"Based on all {cls} flights · stays the same with any sidebar filter")


@st.cache_data(show_spinner=False)
def build_recos(_df):
    df = _df
    out = {"overview": [], "airlines": {}, "timing": {}}

    # ---------- Overview (all classes) ----------
    prem = premium(df)
    cls_mean = df.groupby("Class")["Fare"].mean()
    eco_mean = cls_mean.get("Economy")
    top_cls = cls_mean.idxmax()

    if prem is not None and prem > 0:
        out["overview"].append(("📅", "Book early",
            f"Flights within 3 days cost {prem * 100:+.0f}% vs 22+ days ahead. Plan at least 3 weeks before.", "green"))
    if eco_mean and top_cls != "Economy":
        out["overview"].append(("💺", "Choose class wisely",
            f"{top_cls} costs {cls_mean.max() / eco_mean:.1f}x Economy on average. Upgrade only when you need it.", "amber"))
    out["overview"].append(("🎯", "Budget with the median",
        f"Typical fare is ₹{df['Fare'].median():,.0f}; the average (₹{df['Fare'].mean():,.0f}) is pulled up by a few expensive flights.", "blue"))
    out["overview"].append(("⚠️", "Compare before you book",
        "Fares vary a lot even within one class. Check the Airlines tab and the Predictor.", "red"))

    # ---------- per class ----------
    for cls in CLASS_ORDER:
        sub = df[df["Class"] == cls]
        if sub.empty:
            continue

        # ===== Airlines & Routes =====
        a = []
        by_air = sub.groupby("Airline")["Fare"].mean().sort_values()
        spread = pd.Series(dtype=float)

        if len(by_air) > 1:
            q = sub.groupby("Airline")["Fare"].quantile([0.25, 0.5, 0.75]).unstack()
            spread = ((q[0.75] - q[0.25]) / q[0.5]).dropna()
            a.append(("💰", "Cheapest airline",
                f"{by_air.index[0]} has the lowest {cls} average (₹{by_air.iloc[0]:,.0f}).", "green"))
        else:
            a.append(("💰", "Limited choice",
                f"Only {by_air.index[0]} sells {cls} (avg ₹{by_air.iloc[0]:,.0f}), so compare routes and timing instead.", "green"))

        r = sub.groupby("Route")["Fare"].agg(["mean", "size"])
        r = r[r["size"] >= MIN_ROUTE_FLIGHTS]

        if not r.empty:
            a.append(("🚫", "Costly route",
                f"{r['mean'].idxmax()} is the costliest {cls} route (₹{r['mean'].max():,.0f}). Compare alternatives.", "red"))
        if len(spread) > 1:
            a.append(("📊", "Predictable prices",
                f"{spread.idxmin()} has the most stable fares; {spread.idxmax()} the least.", "blue"))
        if not r.empty:
            a.append(("🛫", "Busiest route",
                f"{r['size'].idxmax()} has the most flights ({int(r['size'].max()):,}), so more choice and competition.", "amber"))

        out["airlines"][cls] = a

        # ===== Timing, Stops & Duration =====
        t = []

        slot = sub.groupby("Departure")["Fare"].mean().dropna()
        if len(slot) > 1:
            t.append(("🕒", "Cheapest departure",
                f"{slot.idxmin()} is the cheapest slot (₹{slot.min():,.0f}); {slot.idxmax()} is the costliest.", "green"))

        non = sub.loc[sub["Stops"] == 0, "Fare"].mean()
        con = sub.loc[sub["Stops"] > 0, "Fare"].mean()
        if pd.notna(non) and pd.notna(con) and non > 0:
            diff = (con / non - 1) * 100
            word = "costlier" if diff > 0 else "cheaper"
            t.append(("🔁", "Non-stop vs connecting",
                f"Connecting flights are {abs(diff):.0f}% {word} than non-stop, but take much longer.", "amber"))

        day = sub.groupby("Journey_day")["Fare"].mean().dropna()
        if len(day) > 1 and day.min() > 0:
            gap = (day.max() / day.min() - 1) * 100
            tip = "Don't over-plan." if gap < 10 else "It is worth picking the cheaper day."
            t.append(("📆", "Day of week",
                f"{day.idxmin()} is cheapest; the gap to {day.idxmax()} is {gap:.0f}%. {tip}", "blue"))

        corr = sub["Duration_in_hours"].corr(sub["Fare"])
        if pd.notna(corr):
            strength = "weak" if abs(corr) < 0.2 else "moderate" if abs(corr) < 0.5 else "strong"
            t.append(("⏳", "Duration vs fare",
                f"The link between duration and fare is {strength} (correlation {corr:.2f}). Check airline and booking time too.", "red"))

        out["timing"][cls] = t

    return out


# ============================ TAB 1 - OVERVIEW  ============================ 
@st.fragment
def render_overview(f, sel_class,recos):
    many_classes = len(sel_class) > 1

    # ---- Chart 1: average fare by class ------------------------------------
    st.markdown("#### Average fare by class (x = times the Economy fare)")

    by_class = f.groupby("Class")["Fare"].agg(["mean", "size"]).reindex(CLASS_ORDER).dropna().reset_index()
    by_class.columns = ["Class", "Average fare", "Flights"]

    economy = by_class.loc[by_class["Class"] == "Economy", "Average fare"]
    economy_fare = economy.iloc[0] if len(economy) else None

    by_class["Label"] = [
        f"₹{fare:,.0f}" + (f"<br>{fare / economy_fare:.1f}x" if economy_fare and cls != "Economy" else "")
        for cls, fare in zip(by_class["Class"], by_class["Average fare"])
    ]

    fig = px.bar(
        by_class,
        x="Class",
        y="Average fare",
        color="Class",
        text="Label",
        custom_data=["Flights"],
        color_discrete_map=CLASS_COLORS,
        category_orders={"Class": CLASS_ORDER},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont_size=11,
        hovertemplate="<b>%{x}</b><br>Average fare: ₹%{y:,.0f}<br>Flights: %{customdata[0]:,}<extra></extra>",
    )
    fig.update_yaxes(range=[0, by_class["Average fare"].max() * 1.3])
    draw(fig, "", "Average fare (₹)", legend=False)
    if economy_fare and len(by_class) > 1:
        top_row = by_class.loc[by_class["Average fare"].idxmax()]
        top_multiple = f"<bad>{top_row['Average fare'] / economy_fare:.1f}x</bad>"
        top_fare = f"<bad>₹{top_row['Average fare']:,.0f}</bad>"
        economy_text = f"<b>₹{economy_fare:,.0f}</b>"
        insight(f"<bad>{top_row['Class']} Class</bad> costs about {top_multiple} "
                     f"the Economy fare ({top_fare} vs {economy_text}).")

    # ---- Chart 2: fare vs days left ----------------------------------------
    st.markdown("#### Fare vs days left before departure")

    metric_options = {"Average fare (₹)": "Average fare", "Price change vs early booking (%)": "Change"}
    metric_label = st.radio(
        "Show",
        list(metric_options),
        index=int(many_classes),
        horizontal=True,
        key=f"r_metric_{many_classes}",     # several classes -> % view by default
    )
    metric = metric_options[metric_label]

    trend = (
        f.groupby(["Days_left", "Class"], as_index=False)["Fare"]
        .mean()
        .rename(columns={"Fare": "Average fare"})
    )
    early_fare = f[f["Days_left"] >= 22].groupby("Class")["Fare"].mean()
    early_fare = early_fare.reindex(trend["Class"].unique()).fillna(f.groupby("Class")["Fare"].mean())

    trend["Change"] = (trend["Average fare"] / trend["Class"].map(early_fare) - 1) * 100
    trend = trend[trend["Days_left"] <= 30]                       # show only the last 30 days (cleaner on phones)

    fig = px.line(
        trend,
        x="Days_left",
        y=metric,
        color="Class",
        color_discrete_map=CLASS_COLORS,
        category_orders={"Class": CLASS_ORDER},
    )
    fig.update_traces(line_width=2.5)
    fig.update_xaxes(autorange="reversed", dtick=5)

    if metric == "Change":
        fig.add_hline(y=0, line_dash="dot", line_color=THEME["line"])
        fig.update_yaxes(ticksuffix="%")
        fig.update_traces(
            hovertemplate="%{fullData.name}<br>Days left: %{x}<br>Change: %{y:+.0f}%<extra></extra>"
        )
        y_title = "Price change vs early booking (%)"
    else:
        fig.update_traces(
            hovertemplate="%{fullData.name}<br>Days left: %{x}<br>Average fare: ₹%{y:,.0f}<extra></extra>"
        )
        y_title = "Average fare (₹)"

    draw(fig, "Days left before departure", y_title, legend=many_classes)

    premiums = {
        cls: premium(f[f["Class"] == cls])
        for cls in CLASS_ORDER
        if cls in f["Class"].unique()
    }
    premiums = {cls: p for cls, p in premiums.items() if p is not None}

    if premiums:
        premium_text = ", ".join(f"{cls} " + (f"<b>{p * 100:+.0f}%</b>" if p <= 0.1 else f"<bad>{p * 100:+.0f}%</bad>") for cls, p in premiums.items())
        hardest = (f"<bad>{max(premiums, key=premiums.get)}</bad> has the biggest jump. "
                   if len(premiums) > 1 else "")

        insight(
            (
                f"Flights leaving within 3 days vs flights leaving 22+ days ahead: {premium_text}. "
                f"{hardest}"
            )
        )

    # ---- Chart 3: distribution of fares ------------------------------------
    st.markdown("#### Distribution of fares")
    view = st.radio("View", ["Histogram", "Box by class"], horizontal=True, key="r_dist")

    if view == "Histogram":
        low, cap = float(f["Fare"].min()), min(float(f["Fare"].max()), 100_000)   # fares above the cap go in the last bar
        edges = np.linspace(low, cap if cap > low else low + 1, 51)          # 50 bins, counted here
        middles = (edges[:-1] + edges[1:]) / 2                               # so the browser gets 50 rows, not 250k

        hist = pd.concat(
            [
                pd.DataFrame({
                    "Fare": middles,
                    "Flights": np.histogram(g["Fare"].clip(upper=cap), bins=edges)[0],
                    "Class": cls,
                })
                for cls, g in f.groupby("Class")
            ],
            ignore_index=True,
        )

        fig = px.bar(
            hist,
            x="Fare",
            y="Flights",
            color="Class",
            color_discrete_map=CLASS_COLORS,
            category_orders={"Class": CLASS_ORDER},
        )
        fig.update_traces(width=float(edges[1] - edges[0]))
        fig.update_layout(bargap=0)
        draw(fig, "Fare (₹)", "Flights", legend=many_classes)

        above = int((f["Fare"] > cap).sum())
        medians = f.groupby("Class")["Fare"].median().reindex(CLASS_ORDER).dropna()
        cut = medians.median()
        median_text = ", ".join(
            f"{cls} " + (f"<b>₹{m:,.0f}</b>" if m <= cut else f"<bad>₹{m:,.0f}</bad>")
            for cls, m in medians.items()
        )
        skew_text = (" A few very expensive flights pull the average above the median."
                     if f["Fare"].mean() > f["Fare"].median() else "")
        cap_text = f" The last bar also includes {above:,} flights priced above ₹{cap:,.0f}." if above else ""

        insight(f"Typical (median) fare: {median_text}.{skew_text}{cap_text}")

    else:
        sample = pd.concat([
            g.sample(min(len(g), 10_000), random_state=42)
            for _, g in f.groupby("Class")
        ])

        fig = px.box(
            sample,
            x="Class",
            y="Fare",
            color="Class",
            points=False,
            color_discrete_map=CLASS_COLORS,
            category_orders={"Class": CLASS_ORDER},
        )
        draw(fig, "", "Fare (₹)", legend=False)

        q = f.groupby("Class")["Fare"].quantile([0.25, 0.75]).unstack().reindex(CLASS_ORDER).dropna()

        if len(q) > 1:
            widest = (q[0.75] - q[0.25]).idxmax()
            low_end = f"<b>₹{q.loc[widest, 0.25]:,.0f}</b>"
            high_end = f"<bad>₹{q.loc[widest, 0.75]:,.0f}</bad>"
            insight(
                f"<b>{widest}</b> has the widest spread: its middle 50% of fares run from "
                        f"{low_end} to {high_end}."
            )
    reco_head("🎯 Smart Booking Tips")
    reco_cards(recos["overview"], "Based on all flights · stays the same with any filter")

# ============================ TAB 2 - AIRLINES & ROUTES ============================ 
@st.fragment
def render_airlines(f, f_all_classes, sel_class, airline_colors,recos):
    if len(sel_class) > 1:
        st.markdown("""
            <div style="
                background: rgba(59, 130, 246, 0.07);
                border-left: 3px solid #3B82F6;
                padding: 12px 16px;
                border-radius: 8px;
                margin: 10px 0 18px 0;
            ">
                <div style="font-weight: 600; margin-bottom: 4px;">
                    🧭 Refine your comparison
                </div>
                <div style="font-size: 14px; color: #9CA3AF;">
                    Several classes are selected, so averages mix cheap and expensive tickets.
                    Select one class in the sidebar for a fair comparison.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- numbers used by the KPIs and charts -------------------------------
    by_airline = f.groupby("Airline")["Fare"].agg(["mean", "size"]).sort_values("mean")
    price_gap = (by_airline["mean"].iloc[-1] / by_airline["mean"].iloc[0] - 1) * 100

    routes = f.groupby("Route")["Fare"].agg(["mean", "size"]).reset_index()
    routes.columns = ["Route", "Average fare", "Flights"]
    routes = routes[routes["Flights"] >= MIN_ROUTE_FLIGHTS].copy()
    routes["Short"] = routes["Route"].apply(
        lambda r: " → ".join(CITY_CODES.get(c, c) for c in r.split(" → "))
    )

    picks = [
        ("Costliest route", "blue", routes["Average fare"].idxmax() if not routes.empty else None),
        ("Cheapest route", "green", routes["Average fare"].idxmin() if not routes.empty else None),
    ]

    route_cards = []
    for label, colour, row_id in picks:
        if row_id is None:
            route_cards.append((label, "n/a", "Not enough flights", colour))
        else:
            route = routes.loc[row_id]
            route_cards.append((
                label,
                route["Short"],
                f"{route['Route']} · avg ₹{route['Average fare']:,.0f}",
                colour,
            ))

    kpi_row([
        ("Costliest airline", by_airline.index[-1], f"Avg fare ₹{by_airline['mean'].iloc[-1]:,.0f}", "red"),
        ("Airline price gap", f"{price_gap:.0f}%", "Costliest vs cheapest airline", "amber"),
        *route_cards,
    ])

    # ---- Chart 4: average fare by airline ----------------------------------
    st.markdown("#### Average fare by airline")

    airline_avg = by_airline.reset_index()
    airline_avg.columns = ["Airline", "Average fare", "Flights"]

    fig = px.bar(
        airline_avg,
        x="Average fare",
        y="Airline",
        orientation="h",
        color="Airline",
        color_discrete_map=airline_colors,
        category_orders={"Airline": airline_avg["Airline"].tolist()},
        text_auto=",.0f",
        custom_data=["Flights"],
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Average fare: ₹%{x:,.0f}<br>Flights: %{customdata[0]:,}<extra></extra>",
    )
    fig.update_xaxes(range=[0, airline_avg["Average fare"].max() * 1.15])
    draw(
        fig,
        "Average fare (₹)",
        "",
        legend=False,
        height=max(300, 42 * len(airline_avg) + 80),
        margin=dict(l=4, r=20),
    )

    # ---- Chart 5: fare spread by airline -----------------------------------
    st.markdown("#### Fare spread by airline")

    sample = pd.concat([
        g.sample(min(len(g), 5_000), random_state=42)
        for _, g in f.groupby("Airline")
    ])

    fig = px.box(
        sample,
        x="Airline",
        y="Fare",
        color="Airline",
        points=False,
        color_discrete_map=airline_colors,
        category_orders={"Airline": airline_avg["Airline"].tolist()[::-1]},
    )
    draw(fig, "", "Fare (₹)", legend=False)

    quartiles = f.groupby("Airline")["Fare"].quantile([0.25, 0.5, 0.75]).unstack()
    spread = ((quartiles[0.75] - quartiles[0.25]) / quartiles[0.5]).dropna()

    if len(spread) > 1:
        least = f"<bad>{spread.max() * 100:.0f}%</bad>"
        most = f"<b>{spread.min() * 100:.0f}%</b>"
        insight(
            (
                f"<bad>{spread.idxmax()}</bad> has the least predictable prices (the middle 50% of its fares spans "
                f"{least} of its median) and <b>{spread.idxmin()}</b> the most predictable "
                f"({most})."
                + (" Select one class in the sidebar for a fair comparison." if len(sel_class) > 1 else "")
            )
        )

    # ---- Chart 6: route heatmap --------------------------------------------
    heading("Average fare by route", CITY_INFO)

    grid = f.pivot_table(values="Fare", index="Source", columns="Destination", aggfunc="mean")
    grid = grid.rename(index=CITY_CODES, columns=CITY_CODES)              # AMD, BLR... instead of long city names
    real_routes = grid.stack().dropna()

    fig = px.imshow(
        grid.fillna(0),
        text_auto=",.0f",
        color_continuous_scale=HEAT_SCALE,
        aspect="auto",
        zmin=float(real_routes.min()) if not real_routes.empty else 0,
        labels=dict(x="To", y="From", color="Average fare (₹)"),
    )
    fig.update_traces(
        hovertemplate="From: %{y}<br>To: %{x}<br>Average fare: ₹%{z:,.0f}<extra></extra>"
    )
    fig.update_layout(coloraxis_showscale=False)
    draw(fig, "To", "From", legend=False, height=440)

    by_origin = grid.mean(axis=1).dropna().rename(index={v: k for k, v in CITY_CODES.items()})

    if len(by_origin) > 1:
        costliest = f"<bad>₹{by_origin.max():,.0f}</bad>"
        cheapest = f"<b>₹{by_origin.min():,.0f}</b>"
        insight(
            (
                f"Flights leaving <bad>{by_origin.idxmax()}</bad> are the costliest on average ({costliest}) "
                f"and flights leaving <b>{by_origin.idxmin()}</b> the cheapest ({cheapest})."
            )
        )

    # ---- Chart 7: top 10 routes --------------------------------------------
    heading("Top 10 routes", CITY_INFO)
    show = st.radio("Show", ["Costliest", "Cheapest", "Busiest"], horizontal=True, key="r_routes")

    if routes.empty:
        st.info(f"No route has {MIN_ROUTE_FLIGHTS} or more flights under the current filters.")

    else:
        if show == "Costliest":
            top = routes.nlargest(10, "Average fare")
        elif show == "Cheapest":
            top = routes.nsmallest(10, "Average fare")
        else:
            top = routes.nlargest(10, "Flights")

        top = top.sort_values("Average fare").copy()
        top["Label"] = [
            f"₹{fare:,.0f}<br>{n:,} flights"
            for fare, n in zip(top["Average fare"], top["Flights"])
        ]

        fig = px.bar(
            top,
            x="Average fare",
            y="Short",
            orientation="h",
            color="Average fare",
            color_continuous_scale=COST_SCALE,
            text="Label",
            custom_data=["Flights", "Route"],
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont_size=11,
            hovertemplate="<b>%{customdata[1]}</b><br>Average fare: ₹%{x:,.0f}<br>Flights: %{customdata[0]:,}<extra></extra>",
        )
        fig.update_xaxes(range=[0, top["Average fare"].max() * 1.5])
        fig.update_layout(coloraxis_showscale=False)

        if show == "Cheapest":
            fig.update_yaxes(autorange="reversed")

        draw(
            fig,
            "Average fare (₹)",
            "",
            legend=False,
            height=max(350, 45 * len(top) + 80),
            margin=dict(l=4, r=30),
        )

    # ---- Chart 8: class mix by airline -------------------------------------
    st.markdown("#### Class mix by airline")

    mix = f_all_classes.groupby(["Airline", "Class"]).size().reset_index(name="Flights")
    mix["Share (%)"] = mix["Flights"] / mix.groupby("Airline")["Flights"].transform("sum") * 100

    fig = px.bar(
        mix,
        x="Airline",
        y="Share (%)",
        color="Class",
        color_discrete_map=CLASS_COLORS,
        category_orders={"Class": CLASS_ORDER},
        custom_data=["Flights"],
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.0f}% of its flights "
                      "(%{customdata[0]:,} flights)<extra></extra>"
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    draw(fig, "", "Share of the airline's flights", height=410)

    premium_sellers = sorted(mix.loc[mix["Class"].isin(["Business", "First"]), "Airline"].unique())
    first_text = " ".join(
        f"<b>{r['Airline']}</b> has only {int(r['Flights']):,} First class flights ({r['Share (%)']:.1f}% of its flights), "
        "too few to show as a visible bar."
        for _, r in mix[mix["Class"] == "First"].iterrows()
    )

    insight(
        (
            f"This chart ignores the Class filter. Only <b>{', '.join(premium_sellers)}</b> sell Business or First class, "
            f"which is why they look expensive on average. {first_text} "
        ) if premium_sellers else None,
    )
    reco_by_class(recos, "airlines", "air")

# ============================ TAB 3 - TIMING, STOPS & DURATION ============================ 
@st.fragment
def render_timing(f, sel_class,recos):
    if len(sel_class) > 1:
        st.markdown("""
        <div style="
            background: rgba(59, 130, 246, 0.07);
            border-left: 3px solid #3B82F6;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 10px 0 18px 0;
        ">
            <div style="font-weight: 600; margin-bottom: 4px;">
                🧭 Refine your comparison
            </div>
            <div style="font-size: 14px; color: #9CA3AF;">
                Several classes are selected, so averages mix cheap and expensive tickets.
                Select one class in the sidebar for a fair comparison.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- numbers used by the KPIs ------------------------------------------
    nonstop_fare = f.loc[f["Stops"] == 0, "Fare"].mean()
    connecting_fare = f.loc[f["Stops"] > 0, "Fare"].mean()
    stops_premium = (
        f"{(connecting_fare / nonstop_fare - 1) * 100:+.0f}%"
        if pd.notna(nonstop_fare) and pd.notna(connecting_fare)
        else "n/a"
    )
    slot_fares = f.groupby("Departure")["Fare"].mean().dropna()

    kpi_row([
        ("Non-stop flights", f"{(f['Stops'] == 0).mean() * 100:.0f}%", "Share of the selected flights", "blue"),
        ("Average duration", f"{f['Duration_in_hours'].mean():.1f} h", f"Median {f['Duration_in_hours'].median():.1f} h", "green"),
        ("Connecting flights", stops_premium, "Fare of 1+ stop flights vs non-stop", "amber"),
        ("Cheapest departure", slot_fares.idxmin(), f"Avg fare ₹{slot_fares.min():,.0f}", "red"),
    ])

    # ---- Chart 9: average fare by time of day ------------------------------
    st.markdown("#### Average fare by time of day")

    dimension = st.radio("Time", ["Departure time", "Arrival time"], horizontal=True, key="r_time")
    column = "Departure" if dimension == "Departure time" else "Arrival"

    by_time = f.groupby(column)["Fare"].agg(["mean", "size"]).reindex(TIME_ORDER).dropna().reset_index()
    by_time.columns = [dimension, "Average fare", "Flights"]

    fig = px.bar(
        by_time,
        x=dimension,
        y="Average fare",
        color=dimension,
        color_discrete_map=TIME_COLORS,
        text_auto=",.0f",
        custom_data=["Flights"],
        category_orders={dimension: TIME_ORDER},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Average fare: ₹%{y:,.0f}<br>Flights: %{customdata[0]:,}<extra></extra>",
    )
    fig.update_yaxes(range=[0, by_time["Average fare"].max() * 1.2])
    draw(fig, "", "Average fare (₹)", legend=False)

    low, high = (
        by_time.loc[by_time["Average fare"].idxmin()],
        by_time.loc[by_time["Average fare"].idxmax()],
    )

    time_gap = f"<bad>{(high['Average fare'] / low['Average fare'] - 1) * 100:.0f}%</bad>"

    insight(
        f"<b>{low[dimension]}</b> is the cheapest slot and <bad>{high[dimension]}</bad> the costliest, "
                f"a gap of {time_gap}."
        if len(by_time) > 1 else None
    )

    # ---- Chart 10: average fare by day of week -----------------------------
    st.markdown("#### Average fare by day of week")

    by_day = f.groupby("Journey_day")["Fare"].agg(["mean", "size"]).reindex(DAY_ORDER).dropna().reset_index()
    by_day.columns = ["Day", "Average fare", "Flights"]

    fig = px.bar(
        by_day,
        x="Day",
        y="Average fare",
        color="Average fare",
        color_continuous_scale=COST_SCALE,
        text_auto=",.0f",
        custom_data=["Flights"],
        category_orders={"Day": DAY_ORDER},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont_size=10,
        hovertemplate="<b>%{x}</b><br>Average fare: ₹%{y:,.0f}<br>Flights: %{customdata[0]:,}<extra></extra>",
    )
    fig.update_yaxes(range=[0, by_day["Average fare"].max() * 1.2])
    fig.update_layout(coloraxis_showscale=False)
    draw(fig, "", "Average fare (₹)", legend=False)

    low, high = (
        by_day.loc[by_day["Average fare"].idxmin()],
        by_day.loc[by_day["Average fare"].idxmax()],
    )

    day_gap = f"<bad>{(high['Average fare'] / low['Average fare'] - 1) * 100:.0f}%</bad>"

    insight(
        f"The day of the week matters little: <b>{low['Day']}</b> is cheapest and <bad>{high['Day']}</bad> costliest, "
        f"a gap of {day_gap}. This also mixes in the days-left effect, because all prices were collected on one day."
        if len(by_day) > 1 else None
    )

    # ---- Chart 11: average fare by number of stops -------------------------
    st.markdown("#### Average fare by number of stops")

    stop_name = f["Stops"].map(STOP_LABELS).rename("Stops")
    by_stops = f.groupby([stop_name, f["Class"]])["Fare"].agg(["mean", "size"]).reset_index()
    by_stops.columns = ["Stops", "Class", "Average fare", "Flights"]

    fig = px.bar(
        by_stops,
        x="Stops",
        y="Average fare",
        color="Class",
        barmode="group",
        text_auto=",.0f",
        color_discrete_map=CLASS_COLORS,
        custom_data=["Flights"],
        category_orders={"Stops": list(STOP_LABELS.values()), "Class": CLASS_ORDER},
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont_size=10,
        hovertemplate="<b>%{x}</b> · %{fullData.name}<br>Average fare: ₹%{y:,.0f}<br>Flights: %{customdata[0]:,}<extra></extra>",
    )
    fig.update_yaxes(range=[0, by_stops["Average fare"].max() * 1.2])
    draw(fig, "", "Average fare (₹)", legend=len(sel_class) > 1)

    parts = []
    for cls in CLASS_ORDER:
        rows = by_stops[by_stops["Class"] == cls]
        direct, connecting = (
            rows[rows["Stops"] == "Non-stop"],
            rows[rows["Stops"] != "Non-stop"],
        )

        if len(direct) and len(connecting):
            connecting_avg = (connecting["Average fare"] * connecting["Flights"]).sum() / connecting["Flights"].sum()
            change = (connecting_avg / direct['Average fare'].iloc[0] - 1) * 100
            parts.append(f"{cls} " + (f"<b>{change:+.0f}%</b>" if change <= 0 else f"<bad>{change:+.0f}%</bad>"))

    insight(
        (
            f"Fare of flights with 1+ stops vs non-stop: {', '.join(parts)}. Connecting flights also take far longer, "
            "about 13 to 17 hours on average versus about 2 hours non-stop."
        ) if parts else None
    )

    # ---- Chart 12: duration vs fare ----------------------------------------
    st.markdown("#### Flight duration vs fare")

    dots = (
        f.sample(min(len(f), 10_000), random_state=42)
        .assign(**{"Stop type": lambda d: d["Stops"].map(STOP_LABELS)})
    )

    fig = px.scatter(
        dots,
        x="Duration_in_hours",
        y="Fare",
        color="Stop type",
        opacity=0.45,
        color_discrete_map=STOP_COLORS,
        category_orders={"Stop type": list(STOP_LABELS.values())},
        custom_data=["Airline", "Class", "Route"],
    )
    fig.update_traces(
        marker_size=5,
        hovertemplate="<b>%{customdata[2]}</b><br>%{customdata[0]} · %{customdata[1]}"
                      "<br>Duration: %{x:.1f} h<br>Fare: ₹%{y:,.0f}<extra></extra>",
    )
    draw(fig, "Flight duration (hours)", "Fare (₹)")

    if len(sel_class) == 1:
        r = f["Duration_in_hours"].corr(f["Fare"])
        strength = "weak" if abs(r) < 0.2 else "moderate" if abs(r) < 0.5 else "strong"

        insight(
            f"Duration and fare have a <b>{strength}</b> link (correlation {r:.2f}): "
            "airline, route and how soon the flight leaves matter as well."
        )
    reco_by_class(recos, "timing", "time")

# ============================ TAB 4 - FARE PREDICTOR ============================ 
@st.fragment
def render_predictor(df, model):
    classes = [c for c in CLASS_ORDER if c in df["Class"].unique()]
    cities = sorted(df["Source"].unique())
    stop_numbers = {label: n for n, label in STOP_LABELS.items()}

    # ---- inputs --------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    cls = c1.selectbox("Class", classes, key="p_class")
    src = c2.selectbox("From", cities, key="p_src")
    dst = c3.selectbox("To", [c for c in cities if c != src], key="p_dst")

    airlines = sorted(df.loc[df["Class"] == cls, "Airline"].unique())

    c1, c2, c3 = st.columns(3)
    airline = c1.selectbox("Airline", airlines, key="p_airline")
    stops = stop_numbers[c2.selectbox("Stops", list(stop_numbers), index=1, key="p_stops")]
    days_left = c3.slider("Days left before departure", 1, 50, 15, key="p_days")

    typical_hours = default_duration(df, airline, cls, src, dst, stops)

    c1, c2, c3, c4 = st.columns(4)
    departure = c1.selectbox("Departure time", TIME_ORDER, index=1, key="p_dep")
    arrival = c2.selectbox("Arrival time", TIME_ORDER, index=3, key="p_arr")
    weekday = c3.selectbox("Day of week", DAY_ORDER, key="p_day")
    duration = c4.number_input(
        "Duration (hours)",
        min_value=0.5,
        max_value=50.0,
        value=float(round(typical_hours, 2)),
        step=0.25,
        key=f"p_dur_{airline}|{cls}|{src}|{dst}|{stops}",
        help="Pre-filled with the typical duration for this choice.",
    )

    seen_in_data = (
        (df["Airline"] == airline)
        & (df["Class"] == cls)
        & (df["Source"] == src)
        & (df["Destination"] == dst)
        & (df["Stops"] == stops)
    ).any()

    if not seen_in_data:
        st.markdown("""
        <div style="
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 12px 16px;
            border-radius: 10px;
            margin: 10px 0 18px 0;
        ">
            <div style="
                font-weight: 600;
                margin-bottom: 5px;
                color: inherit;
            ">
                🧩 Prediction confidence may be lower
            </div>
            <div style="
                font-size: 14px;
                color: #9CA3AF;
            ">
                The selected airline, route, class and stop combination
                does not appear in the historical data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    trip = dict(
        Airline=airline,
        Class=cls,
        Source=src,
        Destination=dst,
        Departure=departure,
        Arrival=arrival,
        Journey_day=weekday,
        Stops=stops,
        Duration_in_hours=duration,
        Days_left=days_left,
    )
    predicted = float(model.predict(make_rows(1, **trip))[0])

    # ---- result cards --------------------------------------------------------
    route_fares = df.loc[
        (df["Class"] == cls) & (df["Source"] == src) & (df["Destination"] == dst),
        "Fare",
    ]

    if route_fares.empty:
        median_text = range_text = diff_text = "n/a"
    else:
        q1, median, q3 = route_fares.quantile([0.25, 0.5, 0.75])
        median_text = f"₹{median:,.0f}"
        range_text = f"₹{q1:,.0f} – ₹{q3:,.0f}"
        diff_text = f"{(predicted / median - 1) * 100:+.0f}%"

    st.divider()

    kpi_row([
        ("Predicted fare", f"₹{predicted:,.0f}", "Estimate for the trip above", "blue"),
        ("Route median", median_text, f"Median {cls} fare on this route", "green"),
        ("Typical range", range_text, "Middle 50% of fares on this route", "amber"),
        ("vs route median", diff_text, "Your predicted fare compared with the median", "red"),
    ])

    # ---- chart: predicted fare vs days left ------------------------------------
    st.markdown("#### Predicted fare vs days left")

    days = np.arange(1, 51)
    curve_input = make_rows(len(days), **trip)
    curve_input["Days_left"] = days
    curve = pd.DataFrame({"Days left": days, "Predicted fare": model.predict(curve_input)})

    fig = px.line(
        curve,
        x="Days left",
        y="Predicted fare",
        markers=True,
        color_discrete_sequence=[THEME["accent"]],
    )
    fig.add_vline(x=days_left, line_dash="dash", line_color=THEME["bad"])
    fig.update_xaxes(autorange="reversed")
    draw(fig, "Days left before departure (closer to departure →)", "Predicted fare (₹)", legend=False)

    # ---- chart: same trip on every airline selling this class --------------------
    st.markdown(f"#### Same trip on other airlines ({cls})")

    rows = []
    for name in airlines:
        hours = default_duration(df, name, cls, src, dst, stops)
        fare = float(model.predict(make_rows(1, **{**trip, "Airline": name, "Duration_in_hours": hours}))[0])
        rows.append({
            "Airline": name,
            "Predicted fare": fare,
            "Choice": "Your choice" if name == airline else "Other airlines",
        })

    compare = pd.DataFrame(rows).sort_values("Predicted fare")

    fig = px.bar(
        compare,
        x="Predicted fare",
        y="Airline",
        orientation="h",
        color="Choice",
        text_auto=",.0f",
        color_discrete_map={"Your choice": PAL[6], "Other airlines": PAL[2]},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, compare["Predicted fare"].max() * 1.15])
    draw(fig, "Predicted fare (₹)", "", height=max(300, 42 * len(compare) + 80))

    insight(
        "The model predicts flight fares with 95.3% variance explained and an average error of about 11%,"
        " with 64% of predictions falling within 11% of actual fares and 84.2% within 20%."
        " Since prices were collected on a single day,"
        " treat predictions as helpful estimates rather than real-time booking prices."
    )


# ============================ MAIN ============================ 
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()

    df = load_data()
    if df is None:
        st.error("Cleaned_dataset.csv was not found. Put it in the same folder as app.py and restart the app.")
        st.stop()

    model, model_source = load_model(df)
    f, f_all_classes, sel_class = render_sidebar(df, model_source)
    airline_colors = {name: PAL[i % len(PAL)] for i, name in enumerate(sorted(df["Airline"].unique()))}

    # KPI row above the tabs
    has_data = not f.empty

    if has_data:
        airline_avg = f.groupby("Airline")["Fare"].mean()
        prem = premium(f)

        kpi_row(
            [
                ("Average fare", f"₹{f['Fare'].mean():,.0f}", f"Median ₹{f['Fare'].median():,.0f}", "blue"),
                ("Total flights", f"{len(f):,}", f"{f['Airline'].nunique()} airlines · {f['Route'].nunique()} routes", "green"),
                ("Cheapest airline", airline_avg.idxmin(), f"Avg fare ₹{airline_avg.min():,.0f}", "amber"),
                (
                    "Last-minute premium",
                    f"{prem * 100:+.0f}%" if prem is not None else "n/a",
                    "<b>Within 3 days</b> vs 22+ days ahead",
                    "red",
                ),
            ],
            hero=True,
        )

    else:
        st.warning("No flights match the current filters. Select at least one value in every sidebar filter.")

    recos = build_recos(df)

    tab_overview, tab_airlines, tab_timing, tab_predictor = st.tabs(
        ["🏠 Overview", "✈️ Airlines & Routes", "⏱️ Timing, Stops & Duration", "🔮 Fare Predictor"]
    )

    with tab_overview:
        if has_data:
            render_overview(f, sel_class,recos)

    with tab_airlines:
        if has_data:
            render_airlines(f, f_all_classes, sel_class, airline_colors,recos)

    with tab_timing:
        if has_data:
            render_timing(f, sel_class,recos)

    with tab_predictor:
        render_predictor(df, model)


main()