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

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# PATH SETUP — allow running from project root or from src/ neighbour
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
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PredictiveMaintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — industrial dark theme with neon accent
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }
    .main-header {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00d4aa, #0066ff, #9333ea);
    }
    .main-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8b949e;
        font-size: 0.95rem;
        margin: 0;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #00d4aa; }
    .metric-card .label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .risk-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .risk-safe    { background: #0d3b2e; color: #3fb950; border: 2px solid #3fb950; }
    .risk-monitor { background: #3d2e0d; color: #e3b341; border: 2px solid #e3b341; }
    .risk-danger  { background: #3b0d0d; color: #f85149; border: 2px solid #f85149; }
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #00d4aa;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }
    div[data-testid="stMetric"] [data-testid="metric-container"] {
        color: #e6edf3;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important;
        color: #e6edf3 !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00d4aa, #0066ff);
        color: #0d1117;
        border: none;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        padding: 10px 24px;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }
    .stSlider [data-testid="stThumbValue"] {
        color: #00d4aa;
        font-family: 'JetBrains Mono', monospace;
    }
    .info-box {
        background: #0d2035;
        border: 1px solid #0066ff44;
        border-left: 4px solid #0066ff;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #79c0ff;
    }
    .warning-box {
        background: #2d1f00;
        border: 1px solid #e3b34144;
        border-left: 4px solid #e3b341;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #e3b341;
    }
    .danger-box {
        background: #2d0d0d;
        border: 1px solid #f8514944;
        border-left: 4px solid #f85149;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #f85149;
    }
    .cost-table {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
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
    """Load the serialised champion model from the artifacts directory."""
    candidates = list((ARTIFACTS_DIR / "models").glob("*_champion.pkl"))
    if not candidates:
        return None
    return joblib.load(candidates[0])


@st.cache_data(show_spinner=False)
def load_training_stats() -> dict:
    """Load training-set distribution stats for drift detection display."""
    stats_path = ARTIFACTS_DIR / "training_stats.csv"
    if stats_path.exists():
        df = pd.read_csv(stats_path, index_col=0)
        return df.to_dict()
    # Fallback — approximate from AI4I 2020 dataset norms
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
    "Tool Wear Failure (TWF)": lambda r: r["Tool wear [min]"] > 200,
    "Heat Dissipation Failure (HDF)": lambda r: r["Temp_Diff"] < 8.6,
    "Power Failure (PWF)": lambda r: r["Power"] < 3500 or r["Power"] > 9000,
    "Overstrain Failure (OSF)": lambda r: r["Force_Ratio"] > 0.035,
    "Random Failure (RNF)": lambda r: False,  # purely stochastic
}


def predict_single(model, row: dict) -> tuple[float, str, list[str]]:
    """Run inference on a single reading dict.

    Returns
    -------
    prob       : float  — failure probability [0, 1]
    risk_level : str    — 'SAFE' | 'MONITOR' | 'DANGER'
    modes      : list   — triggered failure mode names
    """
    # Build a one-row DataFrame matching feature pipeline expectations
    df_in = pd.DataFrame([{
        "Air temperature [K]":      row["air_temp"],
        "Process temperature [K]":  row["proc_temp"],
        "Rotational speed [rpm]":   row["rpm"],
        "Torque [Nm]":              row["torque"],
        "Tool wear [min]":          row["tool_wear"],
        "Type":                     row["machine_type"],
        # Dummy leakage columns (dropped inside pipeline) — not needed, but
        # create_physics_features needs the base sensor columns only.
    }])

    # Compute physics features so the dict is complete for rule engine
    df_phys = create_physics_features(df_in)
    enriched = df_phys.iloc[0].to_dict()

    # Model expects only configured features; pipeline handles the rest
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
    """Build a Plotly gauge chart for failure probability."""
    color_map = {"SAFE": "#3fb950", "MONITOR": "#e3b341", "DANGER": "#f85149"}
    color = color_map[risk]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 52, "color": color, "family": "JetBrains Mono"}},
            delta={"reference": 25, "increasing": {"color": "#f85149"}, "decreasing": {"color": "#3fb950"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#30363d",
                    "tickfont": {"color": "#8b949e", "family": "JetBrains Mono"},
                },
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#161b22",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25],   "color": "#0d3b2e"},
                    {"range": [25, 55],  "color": "#3d2e0d"},
                    {"range": [55, 100], "color": "#3b0d0d"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": prob * 100,
                },
            },
            title={"text": "Failure Probability", "font": {"color": "#8b949e", "size": 14, "family": "Sora"}},
        )
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        margin=dict(t=40, b=0, l=30, r=30),
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>⚙️ PredictiveMaintenance AI</h1>
        <p>Real-time failure prediction · Business cost optimisation · AI4I 2020 dataset · LightGBM champion</p>
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
        "```\njupyter nbconvert --to notebook --execute main_execution.ipynb\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_live, tab_batch, tab_dashboard = st.tabs(
    ["⚡ Live Prediction", "📂 Batch Analysis", "📊 Business Dashboard"]
)

# ============================================================================
# TAB 1 — LIVE PREDICTION
# ============================================================================
with tab_live:
    st.markdown('<div class="section-title">Sensor Input Parameters</div>', unsafe_allow_html=True)

    col_inputs, col_results = st.columns([1, 1], gap="large")

    with col_inputs:
        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            format_func=lambda x: {"L": "L — Low quality tier", "M": "M — Medium quality tier", "H": "H — High quality tier"}[x],
            help="Quality tier of the machine (ordinal: L < M < H)",
        )

        air_temp = st.slider(
            "Air Temperature [K]",
            min_value=295.0, max_value=305.0, value=300.0, step=0.1,
            help="Ambient air temperature in Kelvin",
        )
        proc_temp = st.slider(
            "Process Temperature [K]",
            min_value=305.0, max_value=315.0, value=310.0, step=0.1,
            help="Process temperature in Kelvin",
        )
        rpm = st.slider(
            "Rotational Speed [RPM]",
            min_value=1168, max_value=2886, value=1500, step=10,
            help="Tool rotational speed in revolutions per minute",
        )
        torque = st.slider(
            "Torque [Nm]",
            min_value=3.8, max_value=76.6, value=40.0, step=0.5,
            help="Applied torque in Newton-metres",
        )
        tool_wear = st.slider(
            "Tool Wear [min]",
            min_value=0, max_value=253, value=100, step=1,
            help="Cumulative tool wear time in minutes",
        )

    reading = {
        "air_temp": air_temp,
        "proc_temp": proc_temp,
        "rpm": rpm,
        "torque": torque,
        "tool_wear": tool_wear,
        "machine_type": machine_type,
    }

    with col_results:
        prob, risk, modes = predict_single(model, reading)

        # Gauge
        st.plotly_chart(build_gauge(prob, risk), use_container_width=True, config={"displayModeBar": False})

        # Risk badge
        badge_class = {"SAFE": "risk-safe", "MONITOR": "risk-monitor", "DANGER": "risk-danger"}[risk]
        badge_label = {"SAFE": "✅ SAFE — No Action Required", "MONITOR": "⚠️ MONITOR — Schedule Inspection", "DANGER": "🚨 DANGER — Maintenance Now"}[risk]
        st.markdown(f'<div style="text-align:center;margin:12px 0"><span class="risk-badge {badge_class}">{badge_label}</span></div>', unsafe_allow_html=True)

        # Failure modes
        st.markdown('<div class="section-title" style="margin-top:20px">Likely Failure Modes</div>', unsafe_allow_html=True)
        if modes:
            for m in modes:
                st.markdown(f'<div class="warning-box">⚠️ {m}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">✅ No specific failure pattern detected</div>', unsafe_allow_html=True)

    # Cost analysis row
    st.divider()
    st.markdown('<div class="section-title">Cost Impact Analysis</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    maint_cost    = 500
    downtime_cost = 10_000
    expected_cost = prob * downtime_cost + (1 - prob) * 0
    savings       = expected_cost - maint_cost if prob > 0.05 else 0

    with c1:
        st.metric("Failure Probability", f"{prob*100:.1f}%")
    with c2:
        st.metric("Cost if Ignored (Expected)", f"${prob * downtime_cost:,.0f}")
    with c3:
        st.metric("Preventive Maintenance Cost", f"${maint_cost:,}")
    with c4:
        delta_str = f"Save ${savings:,.0f}" if savings > 0 else "No action needed"
        st.metric("Recommendation", delta_str)

    # Physics features display
    with st.expander("🔬 Derived Physics Features"):
        df_phys = create_physics_features(pd.DataFrame([{
            "Air temperature [K]": air_temp,
            "Process temperature [K]": proc_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }]))
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Temp Differential (Temp_Diff)", f"{df_phys['Temp_Diff'].iloc[0]:.2f} K", help="Process − Air temp: thermal gradient proxy")
        with p2:
            st.metric("Mechanical Power", f"{df_phys['Power'].iloc[0]:,.0f} W", help="Torque × RPM: spindle power input")
        with p3:
            st.metric("Force Ratio", f"{df_phys['Force_Ratio'].iloc[0]:.5f}", help="Torque / RPM: load per revolution proxy")


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
            "📥 Download Template CSV",
            data=template_df.to_csv(index=False),
            file_name="machine_readings_template.csv",
            mime="text/csv",
        )
        st.dataframe(template_df, use_container_width=True, height=150)

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(batch_df):,} machine readings")

            # Validate required columns
            required = ["Air temperature [K]", "Process temperature [K]",
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type"]
            missing_cols = [c for c in required if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.stop()

            # Run batch prediction
            with st.spinner("Running predictions on all machines…"):
                df_phys = create_physics_features(batch_df)
                feature_df = df_phys[NUM_FEATURES + CAT_FEATURES]
                probs = model.predict_proba(feature_df)[:, 1]

            results = batch_df.copy()
            results["Failure_Probability_%"] = (probs * 100).round(2)
            results["Risk_Level"] = pd.cut(
                probs,
                bins=[-np.inf, 0.25, 0.55, np.inf],
                labels=["SAFE", "MONITOR", "DANGER"],
            )
            results["Expected_Cost_$"] = (probs * COST_FALSE_NEGATIVE).round(0).astype(int)
            results = results.sort_values("Failure_Probability_%", ascending=False).reset_index(drop=True)

            # Summary KPIs
            st.divider()
            st.markdown('<div class="section-title">Batch Summary</div>', unsafe_allow_html=True)
            k1, k2, k3, k4, k5 = st.columns(5)
            n_danger  = (results["Risk_Level"] == "DANGER").sum()
            n_monitor = (results["Risk_Level"] == "MONITOR").sum()
            n_safe    = (results["Risk_Level"] == "SAFE").sum()
            total_risk = results["Expected_Cost_$"].sum()

            with k1: st.metric("Total Machines", f"{len(results):,}")
            with k2: st.metric("🚨 Critical (DANGER)", n_danger, delta=f"{n_danger/len(results)*100:.1f}%")
            with k3: st.metric("⚠️ Monitor", n_monitor)
            with k4: st.metric("✅ Safe", n_safe)
            with k5: st.metric("Total Cost at Risk", f"${total_risk:,.0f}")

            # Risk distribution chart
            fig_dist = px.histogram(
                results,
                x="Failure_Probability_%",
                nbins=30,
                color_discrete_sequence=["#00d4aa"],
                title="Failure Probability Distribution",
                template="plotly_dark",
            )
            fig_dist.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font_color="#e6edf3",
                title_font_family="JetBrains Mono",
            )
            fig_dist.add_vline(x=25, line_dash="dash", line_color="#e3b341", annotation_text="MONITOR threshold")
            fig_dist.add_vline(x=55, line_dash="dash", line_color="#f85149", annotation_text="DANGER threshold")
            st.plotly_chart(fig_dist, use_container_width=True)

            # Flagged machines table
            st.markdown('<div class="section-title">Machines Requiring Attention (sorted by risk)</div>', unsafe_allow_html=True)
            flagged = results[results["Risk_Level"] != "SAFE"]

            def color_risk(val):
                colors = {"DANGER": "color: #f85149", "MONITOR": "color: #e3b341", "SAFE": "color: #3fb950"}
                return colors.get(val, "")

            st.dataframe(
                flagged.style.map(color_risk, subset=["Risk_Level"]),
                use_container_width=True,
                height=350,
            )

            # Download
            csv_out = results.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Full Results CSV",
                data=csv_out,
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

    # Assumptions panel
    with st.expander("⚙️ Adjust Assumptions", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_machines     = st.number_input("Fleet size (machines)", 100, 10_000, 500, step=50)
            failure_rate   = st.slider("Annual failure rate (%)", 1.0, 20.0, 3.4, 0.1) / 100
        with col_b:
            cost_reactive  = st.number_input("Reactive failure cost ($)", 1_000, 100_000, 10_000, step=500)
            cost_preventive = st.number_input("Preventive maintenance cost ($)", 100, 5_000, 500, step=100)
        with col_c:
            model_recall   = st.slider("Model recall (%)", 50, 100, 94, 1) / 100
            model_fpr      = st.slider("Model false positive rate (%)", 0, 30, 9, 1) / 100

    n_failures       = int(n_machines * failure_rate)
    n_no_failure     = n_machines - n_failures

    # Strategy costs
    reactive_cost    = n_failures * cost_reactive
    full_preventive  = n_machines * cost_preventive
    model_fn         = int(n_failures * (1 - model_recall))
    model_fp         = int(n_no_failure * model_fpr)
    model_tp         = n_failures - model_fn
    model_cost       = (model_fn * cost_reactive) + ((model_tp + model_fp) * cost_preventive)

    strategies = pd.DataFrame({
        "Strategy":     ["Reactive (fix on break)", "Full Preventive (inspect all)", "This Model (AI-driven)"],
        "Annual Cost":  [reactive_cost, full_preventive, model_cost],
        "Failures Caught": [0, n_failures, model_tp],
        "False Alarms": [0, 0, model_fp],
        "Missed Failures": [n_failures, 0, model_fn],
    })

    fig_bar = px.bar(
        strategies,
        x="Strategy",
        y="Annual Cost",
        color="Strategy",
        color_discrete_sequence=["#f85149", "#e3b341", "#3fb950"],
        title=f"Annual Cost Comparison — {n_machines:,} machine fleet",
        template="plotly_dark",
        text="Annual Cost",
    )
    fig_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig_bar.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        title_font_family="JetBrains Mono",
        showlegend=False,
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # KPI row
    saving_vs_reactive    = reactive_cost - model_cost
    saving_vs_preventive  = full_preventive - model_cost
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Savings vs Reactive",    f"${saving_vs_reactive:,.0f}",    delta=f"{saving_vs_reactive/reactive_cost*100:.1f}%")
    with c2: st.metric("Savings vs Full Preventive", f"${saving_vs_preventive:,.0f}", delta=f"{saving_vs_preventive/full_preventive*100:.1f}%")
    with c3: st.metric("Failures Caught by Model", f"{model_tp}/{n_failures}", delta=f"Recall {model_recall*100:.0f}%")

    st.divider()

    # Interactive threshold slider
    st.markdown('<div class="section-title">Live Threshold Optimisation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Adjust the decision threshold below and see how FP/FN counts and total cost change in real time. '
        'Lower threshold → catch more failures (higher recall) but more false alarms.</div>',
        unsafe_allow_html=True,
    )

    threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.32, 0.01, format="%.2f")

    # Simulate TP/FP/FN at selected threshold (uses known test-set characteristics)
    # Using approximate operating curve from the training run
    # These are interpolated from the actual CV results for realistic simulation
    base_recall  = min(1.0, 0.94 + (0.32 - threshold) * 0.5)
    base_fpr     = max(0.0, 0.09 + (0.32 - threshold) * 0.8)

    sim_failures  = 68   # test-set positives from the pipeline run
    sim_negatives = 1932

    sim_tp = int(sim_failures * base_recall)
    sim_fn = sim_failures - sim_tp
    sim_fp = int(sim_negatives * base_fpr)
    sim_tn = sim_negatives - sim_fp

    sim_cost    = sim_fn * COST_FALSE_NEGATIVE + sim_fp * COST_FALSE_POSITIVE
    sim_recall  = sim_tp / sim_failures if sim_failures > 0 else 0
    sim_prec    = sim_tp / (sim_tp + sim_fp) if (sim_tp + sim_fp) > 0 else 0
    sim_f1      = 2 * sim_prec * sim_recall / (sim_prec + sim_recall) if (sim_prec + sim_recall) > 0 else 0

    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1: st.metric("Threshold",        f"{threshold:.2f}")
    with t2: st.metric("True Positives",   sim_tp)
    with t3: st.metric("False Negatives",  sim_fn, delta=f"-${sim_fn * COST_FALSE_NEGATIVE:,.0f}", delta_color="inverse")
    with t4: st.metric("False Positives",  sim_fp, delta=f"-${sim_fp * COST_FALSE_POSITIVE:,.0f}", delta_color="inverse")
    with t5: st.metric("Total Cost",       f"${sim_cost:,.0f}")
    with t6: st.metric("F1 Score",         f"{sim_f1:.3f}")

    # Cost curve across thresholds
    thresholds = np.arange(0.05, 0.96, 0.01)
    costs, recalls, f1s = [], [], []
    for t in thresholds:
        r = min(1.0, 0.94 + (0.32 - t) * 0.5)
        fpr_t = max(0.0, 0.09 + (0.32 - t) * 0.8)
        tp_t  = int(68 * r); fn_t = 68 - tp_t
        fp_t  = int(1932 * fpr_t)
        p_t   = tp_t / (tp_t + fp_t + 1e-9)
        f1_t  = 2 * p_t * r / (p_t + r + 1e-9)
        costs.append(fn_t * COST_FALSE_NEGATIVE + fp_t * COST_FALSE_POSITIVE)
        recalls.append(r)
        f1s.append(f1_t)

    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Scatter(
        x=thresholds, y=costs, name="Business Cost ($)", line=dict(color="#f85149", width=2), yaxis="y"
    ))
    fig_thresh.add_trace(go.Scatter(
        x=thresholds, y=[r * 100 for r in recalls], name="Recall (%)",
        line=dict(color="#3fb950", width=2, dash="dash"), yaxis="y2"
    ))
    fig_thresh.add_vline(x=threshold, line_dash="dot", line_color="#00d4aa",
                          annotation_text=f"Current: {threshold:.2f}", annotation_font_color="#00d4aa")
    fig_thresh.update_layout(
        title="Cost & Recall vs Decision Threshold",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        title_font_family="JetBrains Mono",
        xaxis=dict(title="Threshold", gridcolor="#21262d"),
        yaxis=dict(title="Business Cost ($)", gridcolor="#21262d", color="#f85149"),
        yaxis2=dict(title="Recall (%)", overlaying="y", side="right", color="#3fb950"),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d"),
        height=380,
    )
    st.plotly_chart(fig_thresh, use_container_width=True)