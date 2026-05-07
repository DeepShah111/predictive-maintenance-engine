"""
Streamlit Predictive Maintenance Dashboard
==========================================
Three tabs:
  1. Live Prediction  — real-time sensor input → failure probability gauge
  2. Batch Analysis   — upload CSV → download flagged machines
  3. Business Dashboard — cost comparison + interactive threshold slider

Run:
    streamlit run streamlit_app.py
"""

import io
import os
import sys
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    ARTIFACTS_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    NUM_FEATURES,
    CAT_FEATURES,
)
from src.feature_engineering import create_physics_features

# ---------------------------------------------------------------------------
# MODEL AUTO-DOWNLOAD (for Streamlit Cloud deployment)
# The model is excluded from GitHub (.gitignore) because it is a large binary.
# On first run in the cloud, gdown fetches it from Google Drive automatically.
# ---------------------------------------------------------------------------
_MODEL_GDRIVE_ID  = "1aZrX7R9Xl9VcC7WNrXzQ4CHAf-YshDth"
_MODEL_GDRIVE_URL = f"https://drive.google.com/uc?id={_MODEL_GDRIVE_ID}"
_MODEL_PATH       = ARTIFACTS_DIR / "models" / "lightgbm_champion.pkl"


def ensure_model_exists() -> None:
    """Download the champion model from Google Drive if not present locally.

    This runs once at app startup on Streamlit Cloud where the model file
    is not committed to the repository. Locally, the file already exists
    so gdown is never called.
    """
    if _MODEL_PATH.exists():
        return  # already present — local dev or subsequent cloud runs

    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner("⏳ Downloading model from Google Drive (first run only)…"):
        try:
            gdown.download(_MODEL_GDRIVE_URL, str(_MODEL_PATH), quiet=False)
        except Exception as exc:
            st.error(
                f"❌ Failed to download model: {exc}\n\n"
                "Check that the Google Drive file is shared as 'Anyone with the link'."
            )
            st.stop()

    if not _MODEL_PATH.exists():
        st.error("❌ Model download appeared to succeed but file is missing.")
        st.stop()


# Run the check immediately at import time
ensure_model_exists()

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PredictiveMaintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — clean light blue + white professional theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #1a2744 !important;
    }

    /* ── Page background: soft blue-gray ── */
    .stApp {
        background: #f0f4f9 !important;
        color: #1a2744 !important;
    }

    /* ── Main header card ── */
    .main-header {
        background: linear-gradient(135deg, #1a6fc4 0%, #2196f3 50%, #42a5f5 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(33, 150, 243, 0.25);
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: '⚙';
        position: absolute;
        right: 32px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.12;
        color: white;
    }
    .main-header h1 {
        font-family: 'DM Mono', monospace;
        font-size: 1.9rem;
        font-weight: 500;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: rgba(255,255,255,0.82);
        font-size: 0.9rem;
        margin: 0;
        font-weight: 300;
    }

    /* ── White cards ── */
    .white-card {
        background: #ffffff;
        border: 1px solid #e3eaf4;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 2px 12px rgba(26, 103, 196, 0.06);
    }

    /* ── Section label ── */
    .section-title {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #1a6fc4;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e3eaf4;
    }

    /* ── Risk badges ── */
    .risk-badge {
        display: inline-block;
        padding: 10px 28px;
        border-radius: 50px;
        font-family: 'DM Mono', monospace;
        font-weight: 500;
        font-size: 1rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
    }
    .risk-safe {
        background: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .risk-monitor {
        background: #fff8e1;
        color: #e65100;
        border: 2px solid #ffa726;
    }
    .risk-danger {
        background: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }

    /* ── Alert boxes ── */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #1565c0;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #ffa726;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #e65100;
    }
    .danger-box {
        background: #ffebee;
        border-left: 4px solid #ef5350;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #c62828;
    }

    /* ── Streamlit metric cards ── */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3eaf4;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(26,103,196,0.05);
    }
    div[data-testid="stMetric"] label {
        color: #5b7499 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] [data-testid="metric-container"] > div {
        color: #1a2744 !important;
        font-weight: 600;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 5px;
        gap: 4px;
        border: 1px solid #e3eaf4;
        box-shadow: 0 2px 8px rgba(26,103,196,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #5b7499;
        border-radius: 7px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a6fc4, #2196f3) !important;
        color: #ffffff !important;
        box-shadow: 0 3px 10px rgba(33,150,243,0.3);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1a6fc4, #42a5f5);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        padding: 10px 24px;
        box-shadow: 0 4px 14px rgba(33,150,243,0.3);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(33,150,243,0.4);
    }

    /* ── Sliders ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #1a6fc4 !important;
        border-color: #1a6fc4 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox [data-baseweb="select"] > div {
        background: #ffffff;
        border-color: #c5d8f0 !important;
        border-radius: 8px;
    }

    /* ── Divider ── */
    hr {
        border-color: #e3eaf4 !important;
        margin: 20px 0;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border: 1px solid #e3eaf4 !important;
        border-radius: 8px !important;
        color: #1a2744 !important;
        font-weight: 500;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #c5d8f0;
        border-radius: 10px;
        padding: 16px;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #e3eaf4;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Success/Error messages ── */
    .stSuccess {
        background: #e8f5e9 !important;
        border-left-color: #4caf50 !important;
        color: #2e7d32 !important;
    }
    .stError {
        background: #ffebee !important;
        border-left-color: #ef5350 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e3eaf4;
    }

    /* ── Number input ── */
    .stNumberInput input {
        background: #ffffff;
        border-color: #c5d8f0;
        border-radius: 8px;
        color: #1a2744;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# MODEL LOADING (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading production model…")
def load_model():
    candidates = list((ARTIFACTS_DIR / "models").glob("*_champion.pkl"))
    if not candidates:
        return None
    return joblib.load(candidates[0])


@st.cache_data(show_spinner=False)
def load_training_stats() -> dict:
    stats_path = ARTIFACTS_DIR / "training_stats.csv"
    if stats_path.exists():
        df = pd.read_csv(stats_path, index_col=0)
        return df.to_dict()
    return {
        "mean": {
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1538.0,
            "Torque [Nm]": 39.9,
            "Tool wear [min]": 107.9,
        },
        "std": {
            "Air temperature [K]": 2.0,
            "Process temperature [K]": 1.5,
            "Rotational speed [rpm]": 179.3,
            "Torque [Nm]": 9.97,
            "Tool wear [min]": 63.7,
        },
    }


# ---------------------------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------------------------

_FAILURE_MODE_RULES = {
    "Tool Wear Failure (TWF)":        lambda r: r["Tool wear [min]"] > 200,
    "Heat Dissipation Failure (HDF)": lambda r: r["Temp_Diff"] < 8.6,
    "Power Failure (PWF)":            lambda r: r["Power"] < 3500 or r["Power"] > 9000000,
    "Overstrain Failure (OSF)":       lambda r: r["Force_Ratio"] > 0.035,
    "Random Failure (RNF)":           lambda r: False,
}


def predict_single(model, row: dict) -> tuple[float, str, list[str]]:
    df_in = pd.DataFrame([{
        "Air temperature [K]":      row["air_temp"],
        "Process temperature [K]":  row["proc_temp"],
        "Rotational speed [rpm]":   row["rpm"],
        "Torque [Nm]":              row["torque"],
        "Tool wear [min]":          row["tool_wear"],
        "Type":                     row["machine_type"],
    }])
    df_phys = create_physics_features(df_in)
    enriched = df_phys.iloc[0].to_dict()
    feature_df = df_phys[NUM_FEATURES + CAT_FEATURES]
    prob = float(model.predict_proba(feature_df)[:, 1][0])

    if prob < 0.25:
        risk = "SAFE"
    elif prob < 0.55:
        risk = "MONITOR"
    else:
        risk = "DANGER"

    modes = [name for name, rule in _FAILURE_MODE_RULES.items() if rule(enriched)]
    return prob, risk, modes


def build_gauge(prob: float, risk: str) -> go.Figure:
    """Build a sleek semi-circular gauge with blue-white theme."""

    # Color palette per risk level
    color_map = {
        "SAFE":    "#2e7d32",
        "MONITOR": "#e65100",
        "DANGER":  "#c62828",
    }
    bar_color = color_map[risk]

    # Gradient arc colors
    steps = [
        {"range": [0, 25],   "color": "#e8f5e9"},   # soft green
        {"range": [25, 55],  "color": "#fff8e1"},   # soft amber
        {"range": [55, 100], "color": "#ffebee"},   # soft red
    ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={
            "suffix": "%",
            "font": {
                "size": 56,
                "color": bar_color,
                "family": "DM Mono, monospace",
            },
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#c5d8f0",
                "tickfont": {"color": "#5b7499", "family": "DM Mono", "size": 11},
                "dtick": 20,
            },
            "bar": {
                "color": bar_color,
                "thickness": 0.28,
            },
            "bgcolor": "#f8fafd",
            "borderwidth": 1,
            "bordercolor": "#e3eaf4",
            "steps": steps,
            "threshold": {
                "line": {"color": bar_color, "width": 4},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
        title={
            "text": "<b>Failure Probability</b>",
            "font": {
                "color": "#5b7499",
                "size": 13,
                "family": "DM Sans",
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        paper_bgcolor="#ffffff",
        font_color="#1a2744",
        margin=dict(t=50, b=10, l=30, r=30),
        height=300,
        plot_bgcolor="#ffffff",
    )

    # Add subtle center annotation showing risk label
    fig.add_annotation(
        x=0.5, y=0.18,
        text=f"<b>{risk}</b>",
        font=dict(size=13, color=bar_color, family="DM Mono"),
        showarrow=False,
        xref="paper", yref="paper",
    )

    return fig


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>⚙ PredictiveMaintenance AI</h1>
        <p>Real-time failure prediction &nbsp;·&nbsp; Business cost optimisation &nbsp;·&nbsp; AI4I 2020 dataset &nbsp;·&nbsp; LightGBM champion</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------
model = load_model()
if model is None:
    st.error(
        "🔴 **Model not found.** Run the training pipeline first:\n"
        "```\npython run_pipeline.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_live, tab_batch, tab_dashboard = st.tabs(
    ["⚡  Live Prediction", "📂  Batch Analysis", "📊  Business Dashboard"]
)

# ============================================================================
# TAB 1 — LIVE PREDICTION
# ============================================================================
with tab_live:
    st.markdown('<div class="section-title">Sensor Input Parameters</div>', unsafe_allow_html=True)

    col_inputs, col_results = st.columns([1, 1.2], gap="large")

    with col_inputs:
        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            format_func=lambda x: {
                "L": "L — Low quality tier",
                "M": "M — Medium quality tier",
                "H": "H — High quality tier",
            }[x],
            help="Quality tier of the machine (ordinal: L < M < H)",
        )
        air_temp = st.slider(
            "Air Temperature [K]",
            min_value=295.0, max_value=305.0, value=300.0, step=0.1,
        )
        proc_temp = st.slider(
            "Process Temperature [K]",
            min_value=305.0, max_value=315.0, value=310.0, step=0.1,
        )
        rpm = st.slider(
            "Rotational Speed [RPM]",
            min_value=1168, max_value=2886, value=1500, step=10,
        )
        torque = st.slider(
            "Torque [Nm]",
            min_value=3.8, max_value=76.6, value=40.0, step=0.5,
        )
        tool_wear = st.slider(
            "Tool Wear [min]",
            min_value=0, max_value=253, value=100, step=1,
        )

    reading = {
        "air_temp":    air_temp,
        "proc_temp":   proc_temp,
        "rpm":         rpm,
        "torque":      torque,
        "tool_wear":   tool_wear,
        "machine_type": machine_type,
    }

    with col_results:
        prob, risk, modes = predict_single(model, reading)

        # Gauge — white card wrapper
        st.markdown('<div class="white-card" style="padding:8px 16px 4px 16px;">', unsafe_allow_html=True)
        st.plotly_chart(
            build_gauge(prob, risk),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Risk badge
        badge_class = {
            "SAFE":    "risk-safe",
            "MONITOR": "risk-monitor",
            "DANGER":  "risk-danger",
        }[risk]
        badge_label = {
            "SAFE":    "✅  SAFE — No Action Required",
            "MONITOR": "⚠️  MONITOR — Schedule Inspection",
            "DANGER":  "🚨  DANGER — Maintenance Now",
        }[risk]
        st.markdown(
            f'<div style="margin:14px 0 6px 0">'
            f'<span class="risk-badge {badge_class}">{badge_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Failure modes
        st.markdown('<div class="section-title" style="margin-top:18px">Likely Failure Modes</div>', unsafe_allow_html=True)
        if modes:
            for m in modes:
                st.markdown(f'<div class="warning-box">⚠️ &nbsp;{m}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">✅ &nbsp;No specific failure pattern detected</div>', unsafe_allow_html=True)

    # Cost analysis
    st.divider()
    st.markdown('<div class="section-title">Cost Impact Analysis</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    maint_cost    = 500
    expected_cost = prob * COST_FALSE_NEGATIVE
    savings       = expected_cost - maint_cost if prob > 0.05 else 0

    with c1:
        st.metric("Failure Probability", f"{prob*100:.1f}%")
    with c2:
        st.metric("Cost if Ignored (Expected)", f"${expected_cost:,.0f}")
    with c3:
        st.metric("Preventive Maintenance Cost", f"${maint_cost:,}")
    with c4:
        delta_str = f"Save ${savings:,.0f}" if savings > 0 else "No action needed"
        st.metric("Recommendation", delta_str)

    # Physics features
    with st.expander("🔬  Derived Physics Features"):
        df_phys = create_physics_features(pd.DataFrame([{
            "Air temperature [K]":    air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]":            torque,
            "Tool wear [min]":        tool_wear,
        }]))
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Temp Differential", f"{df_phys['Temp_Diff'].iloc[0]:.2f} K")
        with p2:
            st.metric("Mechanical Power", f"{df_phys['Power'].iloc[0]:,.0f} W")
        with p3:
            st.metric("Force Ratio", f"{df_phys['Force_Ratio'].iloc[0]:.5f}")


# ============================================================================
# TAB 2 — BATCH ANALYSIS
# ============================================================================
with tab_batch:
    st.markdown('<div class="section-title">Upload Machine Readings CSV</div>', unsafe_allow_html=True)

    col_upload, col_template = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Drop a CSV with sensor readings",
            type=["csv"],
            help="Required columns: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Type",
        )

    with col_template:
        st.markdown('<div class="section-title">Download Template</div>', unsafe_allow_html=True)
        template_df = pd.DataFrame({
            "machine_id":               ["MACHINE-001", "MACHINE-002", "MACHINE-003", "MACHINE-004", "MACHINE-005", "MACHINE-006", "MACHINE-007", "MACHINE-008", "MACHINE-009", "MACHINE-010", "MACHINE-011", "MACHINE-012"],
            "Type":                     ["H",           "M",           "L",           "L",           "M",           "L",           "M",           "L",           "H",           "L",           "M",           "H"          ],
            "Air temperature [K]":      [300.0,         300.0,         302.0,         300.0,         300.0,         303.0,         300.0,         301.0,         299.0,         300.0,         300.0,         300.0        ],
            "Process temperature [K]":  [310.0,         310.0,         309.0,         310.0,         310.0,         309.5,         310.0,         311.0,         309.0,         310.0,         310.0,         310.0        ],
            "Rotational speed [rpm]":   [2000,          1700,          1200,          1168,          1500,          1500,          2886,          1168,          1800,          1500,          1500,          1600         ],
            "Torque [Nm]":              [30.0,          42.0,          65.0,          70.0,          55.0,          40.0,          76.6,          68.0,          35.0,          40.0,          45.0,          38.0         ],
            "Tool wear [min]":          [25,            80,            240,           245,           210,           100,           100,           160,           50,            253,           175,           130          ],
        })
        st.download_button(
            "📥  Download Template CSV",
            data=template_df.to_csv(index=False),
            file_name="machine_readings_template.csv",
            mime="text/csv",
        )
        st.dataframe(template_df[["machine_id", "Type", "Tool wear [min]"]], use_container_width=True, height=150)

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅  Loaded {len(batch_df):,} machine readings")

            required = [
                "Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type",
            ]
            missing_cols = [c for c in required if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.stop()

            with st.spinner("Running predictions…"):
                df_phys    = create_physics_features(batch_df)
                feature_df = df_phys[NUM_FEATURES + CAT_FEATURES]
                probs      = model.predict_proba(feature_df)[:, 1]

            results = batch_df.copy()
            results["Failure_Probability_%"] = (probs * 100).round(2)
            results["Risk_Level"] = pd.cut(
                probs,
                bins=[-np.inf, 0.25, 0.55, np.inf],
                labels=["SAFE", "MONITOR", "DANGER"],
            )
            results["Expected_Cost_$"] = (probs * COST_FALSE_NEGATIVE).round(0).astype(int)
            results = results.sort_values("Failure_Probability_%", ascending=False).reset_index(drop=True)

            # KPIs
            st.divider()
            st.markdown('<div class="section-title">Batch Summary</div>', unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            n_danger  = (results["Risk_Level"] == "DANGER").sum()
            n_monitor = (results["Risk_Level"] == "MONITOR").sum()
            n_safe    = (results["Risk_Level"] == "SAFE").sum()
            total_risk = results["Expected_Cost_$"].sum()

            with k1: st.metric("Total Machines",     f"{len(results):,}")
            with k2: st.metric("🚨 Critical",        n_danger,  delta=f"{n_danger/len(results)*100:.1f}%")
            with k3: st.metric("⚠️ Monitor",         n_monitor)
            with k4: st.metric("✅ Safe",             n_safe)
            with k5: st.metric("Total Cost at Risk", f"${total_risk:,.0f}")

            # Distribution chart
            fig_dist = px.histogram(
                results,
                x="Failure_Probability_%",
                nbins=30,
                color_discrete_sequence=["#2196f3"],
                title="Failure Probability Distribution",
                template="plotly_white",
            )
            fig_dist.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafd",
                font_color="#1a2744",
                title_font_family="DM Mono",
                title_font_color="#1a6fc4",
                xaxis=dict(gridcolor="#e3eaf4"),
                yaxis=dict(gridcolor="#e3eaf4"),
            )
            fig_dist.add_vline(x=25, line_dash="dash", line_color="#ffa726", annotation_text="MONITOR", annotation_font_color="#e65100")
            fig_dist.add_vline(x=55, line_dash="dash", line_color="#ef5350", annotation_text="DANGER",  annotation_font_color="#c62828")
            st.plotly_chart(fig_dist, use_container_width=True)

            # Flagged machines table
            st.markdown('<div class="section-title">Machines Requiring Attention (sorted by risk)</div>', unsafe_allow_html=True)
            flagged = results[results["Risk_Level"] != "SAFE"]

            def color_risk(val):
                colors = {
                    "DANGER":  "color: #c62828; font-weight:600",
                    "MONITOR": "color: #e65100; font-weight:600",
                    "SAFE":    "color: #2e7d32",
                }
                return colors.get(val, "")

            st.dataframe(
                flagged.style.map(color_risk, subset=["Risk_Level"]),
                use_container_width=True,
                height=350,
            )

            st.download_button(
                "📥  Download Full Results CSV",
                data=results.to_csv(index=False).encode(),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Error processing file: {exc}")


# ============================================================================
# TAB 3 — BUSINESS DASHBOARD
# ============================================================================
with tab_dashboard:
    st.markdown('<div class="section-title">Maintenance Strategy Cost Comparison</div>', unsafe_allow_html=True)

    with st.expander("⚙️  Adjust Assumptions", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_machines   = st.number_input("Fleet size (machines)", 100, 10_000, 500, step=50)
            failure_rate = st.slider("Annual failure rate (%)", 1.0, 20.0, 3.4, 0.1) / 100
        with col_b:
            cost_reactive    = st.number_input("Reactive failure cost ($)",    1_000, 100_000, 10_000, step=500)
            cost_preventive  = st.number_input("Preventive maintenance cost ($)", 100, 5_000, 500, step=100)
        with col_c:
            model_recall = st.slider("Model recall (%)", 50, 100, 94, 1) / 100
            model_fpr    = st.slider("Model false positive rate (%)", 0, 30, 9, 1) / 100

    n_failures      = int(n_machines * failure_rate)
    n_no_failure    = n_machines - n_failures
    reactive_cost   = n_failures * cost_reactive
    full_preventive = n_machines * cost_preventive
    model_fn        = int(n_failures * (1 - model_recall))
    model_fp        = int(n_no_failure * model_fpr)
    model_tp        = n_failures - model_fn
    model_cost      = (model_fn * cost_reactive) + ((model_tp + model_fp) * cost_preventive)

    strategies = pd.DataFrame({
        "Strategy":    ["Reactive\n(fix on break)", "Full Preventive\n(inspect all)", "This Model\n(AI-driven)"],
        "Annual Cost": [reactive_cost, full_preventive, model_cost],
    })

    fig_bar = px.bar(
        strategies,
        x="Strategy",
        y="Annual Cost",
        color="Strategy",
        color_discrete_sequence=["#ef5350", "#ffa726", "#42a5f5"],
        title=f"Annual Cost Comparison — {n_machines:,} machine fleet",
        template="plotly_white",
        text="Annual Cost",
    )
    fig_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", marker_line_width=0)
    fig_bar.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafd",
        font_color="#1a2744",
        font_family="DM Sans",
        title_font_family="DM Mono",
        title_font_color="#1a6fc4",
        showlegend=False,
        height=400,
        xaxis=dict(gridcolor="#e3eaf4"),
        yaxis=dict(gridcolor="#e3eaf4"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    saving_vs_reactive   = reactive_cost - model_cost
    saving_vs_preventive = full_preventive - model_cost
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Savings vs Reactive",         f"${saving_vs_reactive:,.0f}",    delta=f"{saving_vs_reactive/max(reactive_cost,1)*100:.1f}%")
    with c2: st.metric("Savings vs Full Preventive",  f"${saving_vs_preventive:,.0f}",  delta=f"{saving_vs_preventive/max(full_preventive,1)*100:.1f}%")
    with c3: st.metric("Failures Caught",             f"{model_tp}/{n_failures}",        delta=f"Recall {model_recall*100:.0f}%")

    st.divider()

    # Threshold slider
    st.markdown('<div class="section-title">Live Threshold Optimisation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Drag the threshold slider to see how FP/FN counts and total cost change in real time. '
        'Lower threshold → catch more failures (higher recall) but more false alarms.</div>',
        unsafe_allow_html=True,
    )

    threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.32, 0.01, format="%.2f")

    base_recall = min(1.0, 0.94 + (0.32 - threshold) * 0.5)
    base_fpr    = max(0.0, 0.09 + (0.32 - threshold) * 0.8)
    sim_tp = int(68 * base_recall);  sim_fn = 68 - sim_tp
    sim_fp = int(1932 * base_fpr);   sim_tn = 1932 - sim_fp
    sim_cost   = sim_fn * COST_FALSE_NEGATIVE + sim_fp * COST_FALSE_POSITIVE
    sim_recall = sim_tp / 68
    sim_prec   = sim_tp / max(sim_tp + sim_fp, 1)
    sim_f1     = 2 * sim_prec * sim_recall / max(sim_prec + sim_recall, 1e-9)

    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1: st.metric("Threshold",       f"{threshold:.2f}")
    with t2: st.metric("True Positives",  sim_tp)
    with t3: st.metric("False Negatives", sim_fn, delta=f"-${sim_fn*COST_FALSE_NEGATIVE:,.0f}", delta_color="inverse")
    with t4: st.metric("False Positives", sim_fp, delta=f"-${sim_fp*COST_FALSE_POSITIVE:,.0f}", delta_color="inverse")
    with t5: st.metric("Total Cost",      f"${sim_cost:,.0f}")
    with t6: st.metric("F1 Score",        f"{sim_f1:.3f}")

    # Cost curve
    thresholds = np.arange(0.05, 0.96, 0.01)
    costs, recalls = [], []
    for t in thresholds:
        r    = min(1.0, 0.94 + (0.32 - t) * 0.5)
        fpr_t = max(0.0, 0.09 + (0.32 - t) * 0.8)
        fn_t = 68 - int(68 * r);  fp_t = int(1932 * fpr_t)
        costs.append(fn_t * COST_FALSE_NEGATIVE + fp_t * COST_FALSE_POSITIVE)
        recalls.append(r * 100)

    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Scatter(
        x=thresholds, y=costs,
        name="Business Cost ($)",
        line=dict(color="#ef5350", width=2.5),
        yaxis="y",
    ))
    fig_thresh.add_trace(go.Scatter(
        x=thresholds, y=recalls,
        name="Recall (%)",
        line=dict(color="#2196f3", width=2.5, dash="dash"),
        yaxis="y2",
    ))
    fig_thresh.add_vline(
        x=threshold,
        line_dash="dot",
        line_color="#1a6fc4",
        line_width=2,
        annotation_text=f"  Current: {threshold:.2f}",
        annotation_font_color="#1a6fc4",
        annotation_font_size=12,
    )
    fig_thresh.update_layout(
        title="Cost & Recall vs Decision Threshold",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafd",
        font_color="#1a2744",
        font_family="DM Sans",
        title_font_family="DM Mono",
        title_font_color="#1a6fc4",
        xaxis=dict(title="Threshold", gridcolor="#e3eaf4"),
        yaxis=dict(title="Business Cost ($)", gridcolor="#e3eaf4", color="#ef5350"),
        yaxis2=dict(title="Recall (%)", overlaying="y", side="right", color="#2196f3"),
        legend=dict(bgcolor="#ffffff", bordercolor="#e3eaf4", borderwidth=1),
        height=400,
    )
    st.plotly_chart(fig_thresh, use_container_width=True)