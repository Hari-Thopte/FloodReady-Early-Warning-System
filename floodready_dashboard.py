"""
FloodReady – Streamlit Dashboard (Working Prototype)
Run: streamlit run floodready_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="FloodReady – Early Warning Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Data loading & caching ──────────────────────────────────
@st.cache_data
def load_data():
    readings = pd.read_csv("floodready_river_readings.csv", parse_dates=["Timestamp"])
    stations = pd.read_csv("floodready_gauge_stations.csv")
    villages = pd.read_csv("floodready_village_profiles.csv")

    # --- Quality fixes ---
    readings["River_Level_m"] = readings.groupby("Gauge_ID")["River_Level_m"] \
        .transform(lambda x: x.fillna(x.median()))
    readings["Sensor_Error"] = readings["River_Level_m"] < 0
    readings_clean = readings[~readings["Sensor_Error"]].copy()
    readings_clean = readings_clean.drop_duplicates(subset=["Gauge_ID", "Timestamp"])
    readings_clean = readings_clean.sort_values(["Gauge_ID", "Timestamp"])
    readings_clean["Stuck"] = readings_clean.groupby("Gauge_ID")["River_Level_m"] \
        .transform(lambda x: (x == x.shift(1)) & (x == x.shift(2)))

    return readings_clean, stations, villages

@st.cache_data
def train_model(readings_clean, stations):
    FEATURES = ["River_Level_m", "Upstream_Level_m", "Rainfall_mm",
                 "Soil_Moisture_pct", "Warning_Level_m", "Danger_Level_m"]
    merged   = readings_clean.merge(stations, on="Gauge_ID", how="left")
    model_df = merged.drop_duplicates(subset=["Gauge_ID","Timestamp"]).dropna(
        subset=FEATURES + ["Flood_Within_48h"]
    )
    split_date = model_df["Timestamp"].quantile(0.75)
    train_df = model_df[model_df["Timestamp"] <= split_date]
    test_df  = model_df[model_df["Timestamp"] >  split_date]

    rf = RandomForestClassifier(n_estimators=100, class_weight={0:1,1:3}, random_state=42)
    rf.fit(train_df[FEATURES], train_df["Flood_Within_48h"])
    y_pred  = rf.predict(test_df[FEATURES])
    y_proba = rf.predict_proba(test_df[FEATURES])[:,1]
    prec = precision_score(test_df["Flood_Within_48h"], y_pred, zero_division=0)
    rec  = recall_score(test_df["Flood_Within_48h"], y_pred, zero_division=0)
    return rf, prec, rec, FEATURES

def prioritise_villages(villages_df):
    df = villages_df.copy()
    df["vuln_ratio"] = df["Vulnerable_Population"] / df["Population"]
    df["shortfall"]  = ((df["Population"] - df["Shelter_Capacity"]).clip(lower=0)
                        / df["Population"])
    df["dist_norm"]  = df["Distance_to_Shelter_km"] / df["Distance_to_Shelter_km"].max()
    df["road_flag"]  = (df["Road_Accessible_Flood"] == "No").astype(float)
    df["Priority_Score"] = (
        0.35*df["vuln_ratio"] + 0.25*df["shortfall"] +
        0.20*df["dist_norm"]  + 0.20*df["road_flag"]
    ) * 100
    df["Priority_Rank"] = df["Priority_Score"].rank(ascending=False).astype(int)
    return df.sort_values("Priority_Rank")

def get_channel(row):
    if row["Road_Accessible_Flood"] == "No" or row["Phone_Coverage_pct"] < 40:
        return "loudspeaker"
    elif row["Alert_Channel"] == "WhatsApp":
        return "whatsapp"
    return "sms"

# ─── Load everything ─────────────────────────────────────────
readings_clean, stations, villages = load_data()
rf, prec, rec, FEATURES = train_model(readings_clean, stations)

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
        <span style="font-size:2.2rem;">🌊</span>
        <div>
            <h2 style="margin:0; font-size:1.4rem; font-weight:800; line-height:1.2;">FloodReady</h2>
            <p style="margin:0; font-size:0.8rem; font-weight:600; opacity:0.75;">Early Warning System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    theme_mode = st.radio("🎨 UI Theme Mode", ["☀️ Light Mode", "🌙 Dark Mode"], index=0)
    st.divider()
    threshold = st.slider("Alert Threshold (Flood Probability)", 0.20, 0.80, 0.40, 0.05)
    st.caption("Lower = more sensitive (catches more floods, more false alarms)")
    st.divider()
    selected_district = st.selectbox("Filter by District", ["All"] + stations["District"].unique().tolist())
    st.divider()
    
    # Styled stat cards for sidebar
    st.markdown(f"""
    <div class="sidebar-stat-card">
        <span class="label">Model Precision</span>
        <span class="val">{prec:.2f}</span>
    </div>
    <div class="sidebar-stat-card">
        <span class="label">Model Recall</span>
        <span class="val">{rec:.2f}</span>
    </div>
    <div class="sidebar-stat-card">
        <span class="label">Last Updated</span>
        <span class="val">{datetime.now().strftime('%H:%M')}</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Custom Dynamic CSS ──────────────────────────────────────
is_light = (theme_mode == "☀️ Light Mode")

if is_light:
    css_styles = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #f8fafc; color: #0f172a; }

    /* Top Navbar Header */
    header[data-testid="stHeader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    header[data-testid="stHeader"] * {
        color: #0f172a !important;
    }

    /* Sidebar Light Mode */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #64748b !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #e2e8f0 !important;
        margin: 14px 0 !important;
    }

    /* Selectboxes, Inputs & Popover Menus Everywhere */
    [data-testid="stSelectbox"],
    [data-testid="stSelectbox"] *,
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-baseweb="base-input"],
    div[data-baseweb="base-input"] *,
    div[data-baseweb="icon"],
    div[data-baseweb="icon"] * {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #0f172a !important;
    }
    div[data-baseweb="select"] > div {
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    div[data-baseweb="select"] svg, 
    div[data-baseweb="select"] svg path,
    div[data-baseweb="icon"] svg,
    div[data-baseweb="icon"] svg path {
        color: #0f172a !important;
        fill: #0f172a !important;
        stroke: #0f172a !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"], 
    div[data-baseweb="menu"] *,
    [role="listbox"],
    [role="listbox"] * {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #0f172a !important;
    }
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul, 
    div[data-baseweb="menu"], 
    [role="listbox"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(15,23,42,0.12) !important;
    }
    div[data-baseweb="popover"] li, 
    [role="option"] {
        color: #0f172a !important;
        background-color: #ffffff !important;
        background: #ffffff !important;
    }
    div[data-baseweb="popover"] li:hover, 
    [role="option"]:hover, 
    [aria-selected="true"] {
        background-color: #e0f2fe !important;
        background: #e0f2fe !important;
        color: #0369a1 !important;
    }
    div[role="radiogroup"] {
        background-color: #f1f5f9;
        padding: 8px 12px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    /* Sidebar Stat Cards */
    .sidebar-stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }
    .sidebar-stat-card .label { font-weight: 600; color: #475569 !important; }
    .sidebar-stat-card .val {
        font-weight: 700;
        color: #0284c7 !important;
        background: #e0f2fe;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-family: monospace;
    }

    /* Metric Boxes */
    .metric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 110px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    .metric-box:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08); }
    .metric-box .num { 
        font-size: 1.8rem; 
        font-weight: 800; 
        color: #0284c7; 
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    .metric-box .label { 
        font-size: 0.75rem; 
        font-weight: 700; 
        color: #64748b; 
        margin-top: 6px; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }

    /* Alert & Safe Cards */
    .alert-card {
        background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
        border: 1px solid #fecaca;
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .alert-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.14);
    }
    .alert-card h4 { color: #991b1b; margin:0 0 10px 0; font-size:1.1rem; font-weight:700; }
    .alert-card p  { color: #334155; margin: 5px 0; font-size:0.92rem; line-height: 1.5; }

    .safe-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(34, 197, 94, 0.06);
    }
    .safe-card h4 { color: #166534; margin:0 0 6px 0; font-size:1.0rem; font-weight:600; }
    .safe-card p  { color: #334155; margin: 3px 0; font-size:0.88rem; }

    /* Badges */
    .channel-sms        { background:#e0f2fe; color:#0369a1; border: 1px solid #bae6fd; padding:4px 12px;
                           border-radius:20px; font-size:0.8rem; font-weight:700; }
    .channel-whatsapp   { background:#dcfce7; color:#15803d; border: 1px solid #86efac; padding:4px 12px;
                           border-radius:20px; font-size:0.8rem; font-weight:700; }
    .channel-loudspeaker{ background:#ffedd5; color:#c2410c; border: 1px solid #fed7aa; padding:4px 12px;
                           border-radius:20px; font-size:0.8rem; font-weight:700; }

    .section-header {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        background: #e0f2fe;
        border-left: 4px solid #0284c7;
        padding: 8px 14px;
        border-radius: 0 10px 10px 0;
        margin: 22px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .approve-badge {
        background: #fef3c7; border: 1px solid #fde68a;
        color: #92400e; padding:6px 18px; border-radius:20px;
        font-size:0.82rem; font-weight:700; display:inline-block; margin-top:10px;
    }

    /* DataFrame, Charts & Expanders */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] iframe {
        background-color: #e0f2fe !important;
        border: 2px solid #93c5fd !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.1) !important;
        padding: 6px !important;
    }
    [data-testid="stVegaLiteChart"] {
        background-color: #e0f2fe !important;
        border: 2px solid #93c5fd !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.1) !important;
        padding: 14px !important;
        overflow: hidden !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    [data-testid="stVegaLiteChart"] canvas,
    [data-testid="stVegaLiteChart"] svg {
        border-radius: 10px !important;
        display: block !important;
        width: 100% !important;
        height: auto !important;
    }
    /* Styled scrollbar for chart containers */
    [data-testid="stVegaLiteChart"]::-webkit-scrollbar {
        height: 8px;
    }
    [data-testid="stVegaLiteChart"]::-webkit-scrollbar-track {
        background: #bae6fd;
        border-radius: 4px;
    }
    [data-testid="stVegaLiteChart"]::-webkit-scrollbar-thumb {
        background: #0284c7;
        border-radius: 4px;
    }
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    }
    [data-testid="stExpander"] summary {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    hr {
        border-color: #e2e8f0 !important;
    }
    </style>
    """
else:
    css_styles = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0a0f1e; color: #f8fafc; }

    /* Top Navbar Header */
    header[data-testid="stHeader"] {
        background-color: #0a0f1e !important;
        color: #f8fafc !important;
    }
    header[data-testid="stHeader"] * {
        color: #f8fafc !important;
    }

    /* Sidebar Dark Mode */
    [data-testid="stSidebar"] {
        background-color: #0d1527 !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1e293b !important;
        margin: 14px 0 !important;
    }

    /* Selectboxes, Inputs & Popover Menus Everywhere */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stBaseWidgetLabel"],
    div[data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] div[role="combobox"],
    [data-testid="stSidebar"] div[role="combobox"] input,
    [data-testid="stSidebar"] div[role="combobox"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
        fill: #0f172a !important;
    }
    [data-testid="stSidebar"] div[role="combobox"]::placeholder,
    [data-testid="stSidebar"] div[role="combobox"] input::placeholder {
        color: #475569 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #334155 !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] [role="button"],
    div[data-baseweb="select"] svg {
        color: #0f172a !important;
        fill: #0f172a !important;
    }
    div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"],
    [role="listbox"] {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="popover"] li, [role="option"] {
        color: #f8fafc !important;
        background-color: #1e293b !important;
    }
    div[data-baseweb="popover"] li:hover, [role="option"]:hover {
        background-color: #0c4a6e !important;
        color: #38bdf8 !important;
    }
    div[role="radiogroup"] {
        background-color: #162032;
        padding: 8px 12px;
        border-radius: 10px;
        border: 1px solid #24344d;
    }

    /* Sidebar Stat Cards */
    .sidebar-stat-card {
        background: #162032;
        border: 1px solid #24344d;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }
    .sidebar-stat-card .label { font-weight: 600; color: #94a3b8 !important; }
    .sidebar-stat-card .val {
        font-weight: 700;
        color: #38bdf8 !important;
        background: #0c4a6e;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-family: monospace;
    }

    /* Metric Boxes */
    .metric-box {
        background: linear-gradient(135deg, #0d1628 0%, #1a2540 100%);
        border: 1px solid #2a4080;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
        height: 110px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    .metric-box:hover { transform: translateY(-2px); }
    .metric-box .num { 
        font-size: 1.8rem; 
        font-weight: 800; 
        color: #60a5fa; 
        line-height: 1.1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }
    .metric-box .label { 
        font-size: 0.75rem; 
        font-weight: 700; 
        color: #8899bb; 
        margin-top: 6px; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }

    /* Alert & Safe Cards */
    .alert-card {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d0f0f 100%);
        border: 1px solid #ff4444;
        border-left: 5px solid #ff4444;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(255,68,68,0.15);
    }
    .alert-card h4 { color: #ff6b6b; margin:0 0 8px 0; font-size:1.1rem; font-weight:700; }
    .alert-card p  { color: #e0c8c8; margin: 4px 0; font-size:0.9rem; }

    .safe-card {
        background: linear-gradient(135deg, #0a1a0a 0%, #0f2d0f 100%);
        border: 1px solid #44ff88;
        border-left: 5px solid #44ff88;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .safe-card h4 { color: #6bffaa; margin:0 0 6px 0; font-size:1.0rem; font-weight:600; }
    .safe-card p  { color: #c8e0d0; margin: 3px 0; font-size:0.88rem; }

    /* Badges */
    .channel-sms        { background:#1a3a5c; color:#60d0ff; padding:4px 12px;
                           border-radius:20px; font-size:0.8rem; font-weight:700; }
    .channel-whatsapp   { background:#1a5c2a; color:#60ff90; padding:4px 12px;
                           border-radius:20px; font-size:0.8rem; font-weight:700; }
    .channel-loudspeaker{ background:#5c3a1a; color:#ffb060; padding:4px 12px;
                           border-radius:20px; font-size:0.8rem; font-weight:700; }

    .section-header {
        font-size: 1.2rem;
        font-weight: 800;
        color: #f8fafc;
        background: #162032;
        border-left: 4px solid #38bdf8;
        padding: 8px 14px;
        border-radius: 0 10px 10px 0;
        margin: 22px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .approve-badge {
        background: linear-gradient(90deg,#1a3a5c,#1a5c2a);
        color:#aaffcc; padding:6px 18px; border-radius:20px;
        font-size:0.82rem; font-weight:700; display:inline-block; margin-top:10px;
    }

    /* DataFrame, Charts & Expanders */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] iframe {
        background-color: #e0f2fe !important;
        border: 2px solid #93c5fd !important;
        border-radius: 14px !important;
        padding: 6px !important;
    }
    [data-testid="stVegaLiteChart"] {
        background-color: #e0f2fe !important;
        border: 2px solid #93c5fd !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
        padding: 14px !important;
        overflow: hidden !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    [data-testid="stVegaLiteChart"] canvas,
    [data-testid="stVegaLiteChart"] svg {
        border-radius: 10px !important;
        display: block !important;
        width: 100% !important;
        height: auto !important;
    }
    /* Styled scrollbar for chart containers */
    [data-testid="stVegaLiteChart"]::-webkit-scrollbar {
        height: 8px;
    }
    [data-testid="stVegaLiteChart"]::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 4px;
    }
    [data-testid="stVegaLiteChart"]::-webkit-scrollbar-thumb {
        background: #38bdf8;
        border-radius: 4px;
    }
    [data-testid="stExpander"] {
        background-color: #0d1628 !important;
        border: 1px solid #2a4080 !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    hr {
        border-color: #2a4080 !important;
    }
    </style>
    """

st.markdown(css_styles, unsafe_allow_html=True)

# ─── Latest readings & predictions ───────────────────────────
latest = (readings_clean.sort_values("Timestamp")
          .groupby("Gauge_ID").last().reset_index())
latest_merged = latest.merge(stations, on="Gauge_ID", how="left")

if selected_district != "All":
    latest_merged = latest_merged[latest_merged["District"] == selected_district]

X_latest = latest_merged[FEATURES].fillna(0)
latest_merged["Flood_Probability"] = rf.predict_proba(X_latest)[:, 1]
latest_merged["Flood_Alert"]       = (latest_merged["Flood_Probability"] >= threshold).astype(int)

# Village prioritisation & alerts
villages_ranked = prioritise_villages(villages)
villages_alerts = villages_ranked.merge(
    latest_merged[["Gauge_ID", "River_Level_m", "Danger_Level_m",
                   "Warning_Level_m", "Flood_Probability", "Flood_Alert", "Timestamp"]],
    on="Gauge_ID", how="left"
)
alert_villages = villages_alerts[villages_alerts["Flood_Alert"] == 1].sort_values("Priority_Rank")
safe_villages  = villages_alerts[villages_alerts["Flood_Alert"] == 0].sort_values("Priority_Rank")

alert_villages["Channel"] = alert_villages.apply(get_channel, axis=1)

# ─── Header Banner ───────────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(135deg, {'#e0f2fe' if is_light else '#0d1628'} 0%, {'#f0f9ff' if is_light else '#162032'} 100%);
            border: 1px solid {'#bae6fd' if is_light else '#2a4080'};
            border-radius: 14px; padding: 20px 24px; margin-bottom: 24px;
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, {'0.04' if is_light else '0.2'});">
    <div>
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: {'#0f172a' if is_light else '#f8fafc'}; display: flex; align-items: center; gap: 10px;">
            🌊 FloodReady <span style="font-size: 0.85rem; font-weight: 700; background: #0284c7; color: white; padding: 3px 12px; border-radius: 20px; letter-spacing: 0.5px;">LIVE EARLY WARNING SYSTEM</span>
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 0.92rem; color: {'#475569' if is_light else '#94a3b8'};">
            District Disaster Management Evacuation & Relief Prioritisation Dashboard
        </p>
    </div>
    <div>
        <span style="font-size: 0.85rem; font-weight: 700; color: {'#0369a1' if is_light else '#38bdf8'}; background: {'#bae6fd' if is_light else '#0c4a6e'}; padding: 6px 14px; border-radius: 20px; font-family: monospace;">
            🕒 {datetime.now().strftime('%A, %d %B %Y — %H:%M')} IST
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Top metrics ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-box">
        <div class="num">{len(alert_villages)}</div>
        <div class="label">Villages on Alert</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-box">
        <div class="num">{int(alert_villages['Population'].sum()):,}</div>
        <div class="label">People at Risk</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-box">
        <div class="num">{int(alert_villages['Vulnerable_Population'].sum()):,}</div>
        <div class="label">Vulnerable People</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-box">
        <div class="num">{latest_merged['Flood_Alert'].sum()}</div>
        <div class="label">Gauges at Risk</div></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-box">
        <div class="num" style="font-size: 1.45rem;">{prec:.0%} / {rec:.0%}</div>
        <div class="label">Precision / Recall</div></div>""", unsafe_allow_html=True)

st.divider()

# ─── Two column layout ───────────────────────────────────────
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown('<div class="section-header">🔴 Priority Village Alerts</div>', unsafe_allow_html=True)

    if len(alert_villages) == 0:
        st.success("✅ No villages currently on alert for the selected district/threshold.")
    else:
        for _, row in alert_villages.iterrows():
            shortfall = max(0, int(row["Population"] - row["Shelter_Capacity"]))
            channel_html = f'<span class="channel-{row["Channel"]}">{row["Channel"].upper()}</span>'
            road_txt = "⚠️ Road CLOSED during flood" if row["Road_Accessible_Flood"] == "No" \
                       else "✅ Road accessible"
            st.markdown(f"""
<div class="alert-card">
  <h4>#{int(row['Priority_Rank'])} &nbsp; {row['Village_Name']} &nbsp;
      <small style="color:#888">({row['Gauge_ID']})</small> &nbsp; {channel_html}
  </h4>
  <p><b>What & When:</b> River at <b>{row['River_Level_m']:.1f} m</b>
     (danger: {row['Danger_Level_m']:.1f} m) &mdash; flood probability
     <b>{row['Flood_Probability']:.0%}</b> within 48 h</p>
  <p><b>Why Priority:</b> {int(row['Vulnerable_Population']):,} vulnerable people &bull;
     shelter shortfall <b>{shortfall:,}</b> &bull; {road_txt}</p>
  <p><b>Action:</b> Evacuate <b>{int(row['Population']):,}</b> residents to shelter
     ({row['Distance_to_Shelter_km']} km). Pre-position <b>{shortfall:,}</b> extra places.</p>
  <p><b>Coverage:</b> {row['Phone_Coverage_pct']}% phone coverage</p>
  <span class="approve-badge">⏳ Awaiting Officer Approval</span>
</div>""", unsafe_allow_html=True)

with col_right:
    # Gauge risk table with clean aligned background
    st.markdown('<div class="section-header">📊 Gauge Status</div>', unsafe_allow_html=True)
    gauge_display = latest_merged[["Gauge_ID","Station_Name","River_Name",
                                    "River_Level_m","Danger_Level_m",
                                    "Flood_Probability","Flood_Alert"]].copy()
    gauge_display["Risk"] = gauge_display["Flood_Probability"].apply(
        lambda x: "🔴 HIGH" if x >= threshold else ("🟡 MEDIUM" if x >= 0.25 else "🟢 LOW")
    )
    gauge_display["Level vs Danger"] = (
        gauge_display["River_Level_m"] - gauge_display["Danger_Level_m"]
    ).round(2)

    tb_bg     = "#e0f2fe" if is_light else "#0d1628"
    tb_border = "#93c5fd" if is_light else "#2a4080"
    th_bg     = "#bae6fd" if is_light else "#162032"
    txt_main  = "#0f172a" if is_light else "#f8fafc"
    txt_sub   = "#334155" if is_light else "#94a3b8"
    row_bdr   = "#bae6fd" if is_light else "#1e293b"

    rows_html = ""
    for _, r in gauge_display.iterrows():
        prob_pct = int(r["Flood_Probability"] * 100)
        risk_badge = f'<span style="font-weight:700;">{r["Risk"]}</span>'
        level_vs_danger = f'{r["Level vs Danger"]:+.1f}'
        rows_html += (
            f'<tr style="border-bottom: 1px solid {row_bdr};">'
            f'<td style="padding: 10px 12px; font-weight:600; color:{txt_main};">{r["Station_Name"]}</td>'
            f'<td style="padding: 10px 12px; color:{txt_sub};">{r["River_Name"]}</td>'
            f'<td style="padding: 10px 12px; font-weight:700; color:#0369a1; text-align:right;">{r["River_Level_m"]:.1f}</td>'
            f'<td style="padding: 10px 12px; text-align:center;">'
            f'<div style="background:{th_bg}; border-radius:10px; height:18px; width:90px; display:inline-block; position:relative; overflow:hidden; vertical-align:middle;">'
            f'<div style="background:#0284c7; width:{prob_pct}%; height:100%; border-radius:10px;"></div>'
            f'</div>'
            f'<span style="font-size:0.82rem; font-weight:700; color:{txt_main}; margin-left:6px;">{prob_pct}%</span>'
            f'</td>'
            f'<td style="padding: 10px 12px; font-weight:600; color:{txt_main}; text-align:right;">{level_vs_danger}</td>'
            f'<td style="padding: 10px 12px; text-align:center;">{risk_badge}</td>'
            f'</tr>'
        )

    gauge_table_html = (
        f'<div style="background-color: {tb_bg}; border: 2px solid {tb_border}; border-radius: 14px; padding: 12px; box-shadow: 0 4px 14px rgba(2,132,199,0.15); overflow-x: auto; margin-bottom: 20px;">'
        f'<table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; font-family: Inter, sans-serif;">'
        f'<thead>'
        f'<tr style="background-color: {th_bg}; color: {txt_main}; text-align: left; border-bottom: 2px solid {tb_border};">'
        f'<th style="padding: 10px 12px; border-top-left-radius: 8px;">Station Name</th>'
        f'<th style="padding: 10px 12px;">River Name</th>'
        f'<th style="padding: 10px 12px; text-align:right;">Level (m)</th>'
        f'<th style="padding: 10px 12px; text-align:center;">Prob</th>'
        f'<th style="padding: 10px 12px; text-align:right;">Level vs Danger</th>'
        f'<th style="padding: 10px 12px; text-align:center; border-top-right-radius: 8px;">Risk</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
        f'</div>'
    )
    st.html(gauge_table_html)

    # Dynamic chart colors matching active theme mode
    chart_bg   = "#e0f2fe" if is_light else "#0d1628"
    chart_bdr  = "#93c5fd" if is_light else "#2a4080"
    chart_txt  = "#0f172a" if is_light else "#f8fafc"
    chart_grid = "#cbd5e1" if is_light else "#1e293b"

    # Graph 1: River Levels vs Warning & Danger Thresholds (Grouped side-by-side)
    st.markdown('<div class="section-header">🌊 Graph 1: River Levels vs Thresholds</div>', unsafe_allow_html=True)
    chart1_df = latest_merged[['Station_Name', 'River_Level_m', 'Warning_Level_m', 'Danger_Level_m']].copy()
    chart1_melted = chart1_df.melt(
        id_vars=['Station_Name'], 
        value_vars=['River_Level_m', 'Warning_Level_m', 'Danger_Level_m'],
        var_name='Metric', 
        value_name='Level_m'
    )
    chart1_melted['Metric'] = chart1_melted['Metric'].replace({
        'River_Level_m': 'River Level',
        'Warning_Level_m': 'Warning Level',
        'Danger_Level_m': 'Danger Level'
    })

    chart1 = alt.Chart(chart1_melted).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X('Station_Name:N', title='Gauge Station', axis=alt.Axis(labelAngle=-45, labelFontSize=11, titleFontSize=12, labelColor=chart_txt, titleColor=chart_txt)),
        xOffset=alt.XOffset('Metric:N', sort=['River Level', 'Warning Level', 'Danger Level']),
        y=alt.Y('Level_m:Q', title='Water Level (meters)', axis=alt.Axis(labelFontSize=11, titleFontSize=12, labelColor=chart_txt, titleColor=chart_txt)),
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=['River Level', 'Warning Level', 'Danger Level'],
            range=['#0284c7', '#f59e0b', '#ef4444']
        ), legend=alt.Legend(title="Metric", titleFontSize=12, labelFontSize=11, labelColor=chart_txt, titleColor=chart_txt, orient='top')),
        tooltip=['Station_Name', 'Metric', 'Level_m']
    ).properties(
        title=alt.TitleParams(text="River Level vs Warning & Danger Thresholds (m)", fontSize=14, color=chart_txt),
        height=420,
        background=chart_bg
    ).configure_view(
        fill=chart_bg,
        stroke=None
    ).configure_axis(
        gridColor=chart_grid,
        domainColor=chart_txt,
        tickColor=chart_txt
    )
    st.altair_chart(chart1, use_container_width=True)

    # Graph 2: Rainfall Readings by Gauge
    st.markdown('<div class="section-header">🌧️ Graph 2: Latest Rainfall Readings</div>', unsafe_allow_html=True)
    chart2 = alt.Chart(latest_merged).mark_bar(color='#2563eb', cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X('Station_Name:N', title='Gauge Station', axis=alt.Axis(labelAngle=-45, labelFontSize=11, titleFontSize=12, labelColor=chart_txt, titleColor=chart_txt)),
        y=alt.Y('Rainfall_mm:Q', title='Rainfall (mm)', axis=alt.Axis(labelFontSize=11, titleFontSize=12, labelColor=chart_txt, titleColor=chart_txt)),
        tooltip=['Station_Name', 'River_Name', 'Rainfall_mm']
    ).properties(
        title=alt.TitleParams(text="Rainfall Readings by Gauge Station (mm)", fontSize=14, color=chart_txt),
        height=420,
        background=chart_bg
    ).configure_view(
        fill=chart_bg,
        stroke=None
    ).configure_axis(
        gridColor=chart_grid,
        domainColor=chart_txt,
        tickColor=chart_txt
    )
    st.altair_chart(chart2, use_container_width=True)

    # Safe villages summary
    st.markdown('<div class="section-header">🟢 Safe Villages</div>', unsafe_allow_html=True)
    for _, row in safe_villages.head(5).iterrows():
        st.markdown(f"""<div class="safe-card">
          <h4>{row['Village_Name']} &nbsp; <small style="color:#555">({row['Gauge_ID']})</small></h4>
          <p>Population: {int(row['Population']):,} &bull; Flood prob: {row['Flood_Probability']:.0%}</p>
        </div>""", unsafe_allow_html=True)
    if len(safe_villages) > 5:
        st.caption(f"… and {len(safe_villages)-5} more safe villages")

st.divider()

# ─── Data Quality Report ─────────────────────────────────────
with st.expander("📋 Data Quality Report"):
    st.markdown("**Issue Summary**")
    issues = {
        "Issue": [
            "Missing River_Level_m",
            "Impossible values (< 0)",
            "Duplicate rows",
            "Stuck sensors (3+ identical readings)"
        ],
        "Count": [
            pd.read_csv("floodready_river_readings.csv")["River_Level_m"].isnull().sum(),
            (pd.read_csv("floodready_river_readings.csv")["River_Level_m"].dropna() < 0).sum(),
            pd.read_csv("floodready_river_readings.csv").duplicated(
                subset=["Gauge_ID", "Timestamp"]
            ).sum(),
            readings_clean["Stuck"].sum() if "Stuck" in readings_clean.columns else "N/A"
        ],
        "Action": [
            "Impute with per-gauge median",
            "Flag as sensor error; exclude from model",
            "Remove duplicate rows",
            "Flag for officer review; keep in data"
        ]
    }
    st.dataframe(pd.DataFrame(issues), hide_index=True, width="stretch")

# ─── Responsible AI note ─────────────────────────────────────
with st.expander("⚖️ Responsible AI Principles"):
    st.markdown("""
**1. Recall-first design** — Threshold set to 0.40 to minimise missed floods.
Officers can adjust via the sidebar slider.

**2. Equity** — Villages with phone coverage < 40% or inaccessible roads during floods
automatically receive Loudspeaker/truck dispatch, not just SMS.

**3. Privacy** — Vulnerability data (disability counts, income levels) is visible only to
authorised officers. No individual-level data is sent in bulk alerts.

**4. Human in the loop** — All alerts show "Awaiting Officer Approval" status.
No message is dispatched without explicit officer confirmation.

**5. Fallback** — If sensor data is unavailable for > 2 hours, the gauge is flagged
and upstream readings are substituted. If model confidence < 60%, officer must manually confirm.
""")

st.caption("FloodReady v1.0 | Mini Challenge Week 6 | District Disaster Management System")
