"""
P6 - Supplier Relationship Management Dashboard
UE23CS342BA1 Supply Chain Management for Engineers
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from ml_models import (
    train_delay_model, detect_anomalies,
    segment_suppliers, forecast_performance
)

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "srm_database.db")

# ─── THEME ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRM Dashboard | P6",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

NAVY   = "#0D1B2A"
BLUE   = "#1B4F8A"
ACCENT = "#00C9A7"
WARN   = "#F4A261"
DANGER = "#E63946"
LIGHT  = "#F8F9FA"
CARD   = "#162032"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {NAVY};
    color: {LIGHT};
  }}
  .main {{ background-color: {NAVY}; }}
  .block-container {{ padding: 1.5rem 2rem; }}

  h1,h2,h3 {{ font-family: 'Syne', sans-serif; color: {LIGHT}; }}

  .kpi-card {{
    background: {CARD};
    border-left: 4px solid {ACCENT};
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 10px;
  }}
  .kpi-card.warn {{ border-left-color: {WARN}; }}
  .kpi-card.danger {{ border-left-color: {DANGER}; }}

  .kpi-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;
                color: #8899aa; margin-bottom: 4px; }}
  .kpi-value {{ font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: {LIGHT}; }}
  .kpi-sub   {{ font-size: 0.78rem; color: {ACCENT}; margin-top: 2px; }}

  .section-header {{
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: {ACCENT};
    border-bottom: 1px solid #1e3050;
    padding-bottom: 6px;
    margin: 18px 0 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  .stSelectbox > div > div {{ background: {CARD} !important; color: {LIGHT} !important; }}
  .stMultiSelect > div > div {{ background: {CARD} !important; }}
  [data-baseweb="tag"] {{ background: {BLUE} !important; }}

  .sidebar .sidebar-content {{ background: {CARD}; }}
  section[data-testid="stSidebar"] {{ background: {CARD}; }}
  section[data-testid="stSidebar"] * {{ color: {LIGHT} !important; }}

  .stDataFrame {{ background: {CARD}; }}
  iframe {{ border-radius: 8px; }}

  .ml-badge {{
    display: inline-block;
    background: {BLUE};
    color: {ACCENT};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-left: 8px;
    vertical-align: middle;
  }}

  div[data-testid="metric-container"] {{
    background: {CARD};
    border-radius: 8px;
    padding: 10px;
  }}
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADERS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    suppliers   = pd.read_sql("SELECT * FROM suppliers", conn)
    perf        = pd.read_sql("SELECT * FROM supplier_performance", conn)
    orders      = pd.read_sql("SELECT * FROM purchase_orders", conn)
    comms       = pd.read_sql("SELECT * FROM communications", conn)
    incidents   = pd.read_sql("SELECT * FROM quality_incidents", conn)
    materials   = pd.read_sql("SELECT * FROM materials", conn)
    sm          = pd.read_sql("SELECT * FROM supplier_materials", conn)
    conn.close()
    return suppliers, perf, orders, comms, incidents, materials, sm

@st.cache_data
def load_ml():
    dm    = train_delay_model()
    anom  = detect_anomalies()
    segs, feats, cstats = segment_suppliers()
    fc    = forecast_performance()
    return dm, anom, segs, feats, cstats, fc

def plotly_dark(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=LIGHT, family="DM Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(gridcolor="#1e3050", zerolinecolor="#1e3050")
    fig.update_yaxes(gridcolor="#1e3050", zerolinecolor="#1e3050")
    return fig

suppliers, perf, orders, comms, incidents, materials, sm = load_data()
dm, anom, segs, feats, cstats, fc = load_ml()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 SRM Dashboard")
    st.markdown("**P6 · Supplier Relationship Mgmt**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Executive Overview",
        "🏭 Supplier Performance",
        "📦 Orders & Procurement",
        "🔬 Quality & Incidents",
        "💬 Communications",
        "🤖 ML Insights",
    ])
    st.markdown("---")
    sel_region   = st.multiselect("Filter: Region",   suppliers["region"].unique(),   default=list(suppliers["region"].unique()))
    sel_category = st.multiselect("Filter: Category", suppliers["category"].unique(), default=list(suppliers["category"].unique()))
    sel_status   = st.multiselect("Filter: Status",   suppliers["status"].unique(),   default=list(suppliers["status"].unique()))

# Apply global filters
flt_sids = suppliers[
    suppliers["region"].isin(sel_region) &
    suppliers["category"].isin(sel_category) &
    suppliers["status"].isin(sel_status)
]["supplier_id"].tolist()

flt_perf  = perf[perf["supplier_id"].isin(flt_sids)]
flt_orders= orders[orders["supplier_id"].isin(flt_sids)]
flt_comms = comms[comms["supplier_id"].isin(flt_sids)]
flt_inc   = incidents[incidents["supplier_id"].isin(flt_sids)]
flt_sup   = suppliers[suppliers["supplier_id"].isin(flt_sids)]


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview")
    st.markdown("*Key supply chain health metrics at a glance*")

    # KPI ROW
    k1,k2,k3,k4,k5 = st.columns(5)
    total_sup = len(flt_sup)
    active_sup = (flt_sup["status"]=="Active").sum()
    avg_score  = flt_perf["overall_score"].mean()
    otd        = flt_perf["on_time_delivery"].mean()
    total_spend= flt_orders["total_value"].sum()
    avg_delay  = flt_orders["delay_days"].mean()
    open_inc   = (flt_inc["resolved"]==0).sum()

    def kpi(col, label, value, sub, cls=""):
        col.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    kpi(k1, "Total Suppliers",      total_sup,                  f"{active_sup} active")
    kpi(k2, "Avg Performance Score",f"{avg_score:.1f}",          "out of 100")
    kpi(k3, "On-Time Delivery",     f"{otd:.1f}%",              "last 24 months", "")
    kpi(k4, "Total Spend",          f"₹{total_spend/1e6:.1f}M", "across all POs", "warn")
    kpi(k5, "Open Incidents",       open_inc,                   "unresolved", "danger")

    st.markdown('<div class="section-header">Performance Trends & Spend Distribution</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        # Monthly avg performance trend
        trend = flt_perf.groupby("eval_month")["overall_score"].mean().reset_index()
        fig = px.area(trend, x="eval_month", y="overall_score",
                      title="Average Supplier Performance Score (24 Months)",
                      color_discrete_sequence=[ACCENT])
        fig.update_traces(fill='tozeroy', fillcolor=f"rgba(0,201,167,0.12)")
        st.plotly_chart(plotly_dark(fig, 300), use_container_width=True)

    with c2:
        # Spend by category
        cat_spend = flt_orders.merge(materials[["material_id","category"]], on="material_id")
        cat_spend = cat_spend.groupby("category")["total_value"].sum().reset_index()
        fig = px.pie(cat_spend, names="category", values="total_value",
                     title="Spend by Category",
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(plotly_dark(fig, 300), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Supplier status breakdown
        status_cnt = flt_sup["status"].value_counts().reset_index()
        status_cnt.columns = ["status","count"]
        colors = {
            "Active": ACCENT, "Under Review": WARN, "Blacklisted": DANGER
        }
        fig = px.bar(status_cnt, x="status", y="count", title="Suppliers by Status",
                     color="status", color_discrete_map=colors)
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

    with c4:
        # OTD by region
        reg_perf = flt_perf.merge(flt_sup[["supplier_id","region"]], on="supplier_id")
        reg_perf = reg_perf.groupby("region")["on_time_delivery"].mean().reset_index()
        fig = px.bar(reg_perf, x="region", y="on_time_delivery",
                     title="Avg On-Time Delivery % by Region",
                     color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: SUPPLIER PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Supplier Performance":
    st.markdown("# 🏭 Supplier Performance")

    # Supplier selector
    sup_map = dict(zip(flt_sup["name"], flt_sup["supplier_id"]))
    sel_name = st.selectbox("Select Supplier", list(sup_map.keys()))
    sel_sid  = sup_map[sel_name]
    sup_perf = perf[perf["supplier_id"]==sel_sid].sort_values("eval_month")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Score",        f"{sup_perf['overall_score'].mean():.1f}")
    c2.metric("On-Time Delivery", f"{sup_perf['on_time_delivery'].mean():.1f}%")
    c3.metric("Avg Quality Score",f"{sup_perf['quality_score'].mean():.1f}")
    c4.metric("Avg Defect Rate",  f"{sup_perf['defect_rate'].mean()*100:.2f}%")

    st.markdown('<div class="section-header">Monthly KPI Breakdown</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Overall Score","On-Time Delivery (%)","Quality Score","Defect Rate (%)"])
    kws = dict(mode="lines+markers", line_shape="spline")
    fig.add_trace(go.Scatter(x=sup_perf.eval_month, y=sup_perf.overall_score,
        line=dict(color=ACCENT), **kws), row=1, col=1)
    fig.add_trace(go.Scatter(x=sup_perf.eval_month, y=sup_perf.on_time_delivery,
        line=dict(color=BLUE), **kws), row=1, col=2)
    fig.add_trace(go.Scatter(x=sup_perf.eval_month, y=sup_perf.quality_score,
        line=dict(color=WARN), **kws), row=2, col=1)
    fig.add_trace(go.Scatter(x=sup_perf.eval_month, y=sup_perf.defect_rate*100,
        line=dict(color=DANGER), **kws), row=2, col=2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(plotly_dark(fig, 420), use_container_width=True)

    st.markdown('<div class="section-header">All Supplier Comparison (Heatmap)</div>', unsafe_allow_html=True)
    avg_perf = perf.merge(suppliers[["supplier_id","name"]], on="supplier_id")
    heat = avg_perf.groupby("name")[["on_time_delivery","quality_score","fill_rate","overall_score"]].mean().round(1)
    fig = px.imshow(heat,
        color_continuous_scale="Teal",
        labels=dict(color="Score"),
        title="Supplier KPI Heatmap")
    st.plotly_chart(plotly_dark(fig, 380), use_container_width=True)

    st.markdown('<div class="section-header">Lead Time Distribution by Supplier</div>', unsafe_allow_html=True)
    lt = sm.merge(suppliers[["supplier_id","name"]], on="supplier_id")
    fig = px.box(lt, x="name", y="lead_time_days",
                 color_discrete_sequence=[ACCENT],
                 title="Lead Time (Days) per Supplier")
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(plotly_dark(fig, 300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: ORDERS & PROCUREMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Orders & Procurement":
    st.markdown("# 📦 Orders & Procurement")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total POs",           len(flt_orders))
    c2.metric("Total Spend",         f"₹{flt_orders['total_value'].sum()/1e6:.2f}M")
    c3.metric("Avg Delay (days)",    f"{flt_orders['delay_days'].mean():.1f}")
    c4.metric("On-Time POs",         f"{(flt_orders['delay_days']==0).mean()*100:.1f}%")

    st.markdown('<div class="section-header">Order Status & Monthly Spend</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        status_cnt = flt_orders["status"].value_counts().reset_index()
        status_cnt.columns = ["status","count"]
        fig = px.pie(status_cnt, names="status", values="count", title="PO Status Distribution",
                     color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(plotly_dark(fig, 300), use_container_width=True)

    with c2:
        flt_orders["month"] = pd.to_datetime(flt_orders["order_date"]).dt.to_period("M").astype(str)
        monthly = flt_orders.groupby("month")["total_value"].sum().reset_index()
        fig = px.bar(monthly, x="month", y="total_value",
                     title="Monthly Procurement Spend",
                     color_discrete_sequence=[BLUE])
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(plotly_dark(fig, 300), use_container_width=True)

    st.markdown('<div class="section-header">Delay Analysis</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        delay_sup = flt_orders.merge(flt_sup[["supplier_id","name"]], on="supplier_id")
        delay_sup = delay_sup.groupby("name")["delay_days"].mean().reset_index().sort_values("delay_days", ascending=False)
        fig = px.bar(delay_sup, x="name", y="delay_days",
                     title="Avg Delay per Supplier (days)",
                     color="delay_days", color_continuous_scale="Reds")
        fig.update_layout(xaxis_tickangle=-30, showlegend=False)
        st.plotly_chart(plotly_dark(fig, 300), use_container_width=True)

    with c4:
        fig = px.histogram(flt_orders, x="delay_days",
                           title="Delay Days Distribution",
                           color_discrete_sequence=[WARN], nbins=20)
        st.plotly_chart(plotly_dark(fig, 300), use_container_width=True)

    st.markdown('<div class="section-header">Recent Purchase Orders</div>', unsafe_allow_html=True)
    disp = flt_orders.merge(flt_sup[["supplier_id","name"]], on="supplier_id")
    disp = disp.merge(materials[["material_id","material_name"]], on="material_id")
    disp = disp[["po_id","name","material_name","order_date","expected_date","status","quantity","total_value","delay_days"]]
    disp.columns = ["PO ID","Supplier","Material","Order Date","Expected","Status","Qty","Value (₹)","Delay (d)"]
    st.dataframe(disp.sort_values("Order Date", ascending=False).head(30),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: QUALITY & INCIDENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Quality & Incidents":
    st.markdown("# 🔬 Quality & Incidents")

    total_inc = len(flt_inc)
    unresolved= (flt_inc["resolved"]==0).sum()
    avg_defect= (flt_inc["defect_qty"]/flt_inc["batch_qty"]).mean()*100
    critical  = (flt_inc["severity"]=="Critical").sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Incidents",  total_inc)
    c2.metric("Unresolved",       unresolved)
    c3.metric("Avg Defect Rate",  f"{avg_defect:.2f}%")
    c4.metric("Critical Cases",   critical)

    st.markdown('<div class="section-header">Incident Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sev = flt_inc["severity"].value_counts().reset_index()
        sev.columns = ["severity","count"]
        color_map = {"Low":ACCENT, "Medium":WARN, "High":"#F4743B", "Critical":DANGER}
        fig = px.bar(sev, x="severity", y="count", color="severity",
                     color_discrete_map=color_map, title="Incidents by Severity")
        fig.update_layout(showlegend=False)
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

    with c2:
        dtype = flt_inc["defect_type"].value_counts().reset_index()
        dtype.columns = ["type","count"]
        fig = px.pie(dtype, names="type", values="count", title="Defect Type Breakdown",
                     color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        inc_sup = flt_inc.merge(flt_sup[["supplier_id","name"]], on="supplier_id")
        inc_count = inc_sup.groupby("name").size().reset_index(name="incidents").sort_values("incidents", ascending=False)
        fig = px.bar(inc_count, x="name", y="incidents",
                     title="Incidents per Supplier", color_discrete_sequence=[DANGER])
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

    with c4:
        flt_inc["defect_pct"] = flt_inc["defect_qty"] / flt_inc["batch_qty"] * 100
        fig = px.scatter(flt_inc, x="batch_qty", y="defect_pct",
                         color="severity", size="defect_qty",
                         color_discrete_map=color_map,
                         title="Defect % vs Batch Size")
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

    st.markdown('<div class="section-header">Open Incidents</div>', unsafe_allow_html=True)
    open_df = flt_inc[flt_inc["resolved"]==0].merge(flt_sup[["supplier_id","name"]], on="supplier_id")
    open_df = open_df.merge(materials[["material_id","material_name"]], on="material_id")
    disp = open_df[["incident_id","name","material_name","incident_date","defect_type","severity","defect_qty","batch_qty"]]
    disp.columns = ["ID","Supplier","Material","Date","Defect Type","Severity","Defects","Batch"]
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5: COMMUNICATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Communications":
    st.markdown("# 💬 Communications")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Communications", len(flt_comms))
    c2.metric("Avg Response Time",    f"{flt_comms['response_days'].mean():.1f} days")
    c3.metric("Unresolved",           (flt_comms["resolved"]==0).sum())
    c4.metric("Resolution Rate",      f"{flt_comms['resolved'].mean()*100:.1f}%")

    st.markdown('<div class="section-header">Channel & Subject Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        ch = flt_comms["channel"].value_counts().reset_index()
        ch.columns = ["channel","count"]
        fig = px.pie(ch, names="channel", values="count", title="Communication Channels",
                     color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

    with c2:
        subj = flt_comms["subject"].value_counts().reset_index().head(6)
        subj.columns = ["subject","count"]
        fig = px.bar(subj, x="count", y="subject", orientation="h",
                     title="Top Communication Topics",
                     color_discrete_sequence=[BLUE])
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        resp_sup = flt_comms.merge(flt_sup[["supplier_id","name"]], on="supplier_id")
        resp_avg = resp_sup.groupby("name")["response_days"].mean().reset_index().sort_values("response_days")
        fig = px.bar(resp_avg, x="name", y="response_days",
                     title="Avg Response Time per Supplier (days)",
                     color_discrete_sequence=[WARN])
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)

    with c4:
        flt_comms["month"] = pd.to_datetime(flt_comms["comm_date"]).dt.to_period("M").astype(str)
        monthly = flt_comms.groupby("month").size().reset_index(name="count")
        fig = px.line(monthly, x="month", y="count",
                      title="Monthly Communication Volume",
                      color_discrete_sequence=[ACCENT], markers=True)
        st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6: ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Insights":
    st.markdown("# 🤖 ML Insights")
    st.markdown("*Four machine learning models applied to supplier relationship data*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌲 Delay Forecasting",
        "🚨 Anomaly Detection",
        "🗂️ Supplier Segmentation",
        "📈 Performance Forecast"
    ])

    # ── TAB 1: Random Forest Delay Prediction ─────────────────────────────
    with tab1:
        st.markdown("### 🌲 Delay Forecasting — Random Forest Regressor")
        st.markdown("""
        **Algorithm:** Random Forest Regressor  
        **Goal:** Predict how many days a purchase order will be delayed based on supplier profile, material type, and historical performance.  
        **Business Impact:** Proactively flags high-risk orders so managers can mitigate supply disruptions before they occur.
        """)
        c1, c2 = st.columns(2)
        c1.metric("Mean Absolute Error", f"{dm['mae']} days")
        c2.metric("R² Score", dm['r2'])

        c1, c2 = st.columns(2)
        with c1:
            fi = dm["feature_importances"].reset_index()
            fi.columns = ["Feature","Importance"]
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         title="Feature Importances",
                         color="Importance", color_continuous_scale="Teal")
            st.plotly_chart(plotly_dark(fig, 350), use_container_width=True)

        with c2:
            pred_df = pd.DataFrame({"Actual": dm["y_test"].values, "Predicted": dm["y_pred"]})
            fig = px.scatter(pred_df, x="Actual", y="Predicted",
                             title="Actual vs Predicted Delay Days",
                             color_discrete_sequence=[ACCENT])
            fig.add_trace(go.Scatter(x=[0,20], y=[0,20],
                mode="lines", line=dict(dash="dash", color=WARN), name="Perfect"))
            st.plotly_chart(plotly_dark(fig, 350), use_container_width=True)

    # ── TAB 2: Anomaly Detection ───────────────────────────────────────────
    with tab2:
        st.markdown("### 🚨 Anomaly Detection — Isolation Forest")
        st.markdown("""
        **Algorithm:** Isolation Forest (unsupervised)  
        **Goal:** Detect months where a supplier's performance deviates unusually from their historical baseline.  
        **Business Impact:** Early warning system for supplier deterioration — catches issues before they escalate into supply failures.
        """)
        anom_flt = anom[anom["supplier_id"].isin(flt_sids)]
        n_anom = anom_flt["is_anomaly"].sum()
        c1, c2 = st.columns(2)
        c1.metric("Anomalous Records Detected", int(n_anom))
        c2.metric("Anomaly Rate", f"{n_anom/len(anom_flt)*100:.1f}%")

        fig = px.scatter(anom_flt, x="eval_month", y="overall_score",
                         color="is_anomaly",
                         color_discrete_map={True: DANGER, False: ACCENT},
                         hover_data=["name","on_time_delivery","quality_score"],
                         title="Supplier Performance Anomalies (Red = Anomalous Month)",
                         symbol="is_anomaly")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(plotly_dark(fig, 380), use_container_width=True)

        st.markdown("**Flagged Anomalous Records**")
        flagged = anom_flt[anom_flt["is_anomaly"]].sort_values("anomaly_score")
        st.dataframe(flagged[["name","eval_month","overall_score","on_time_delivery",
                               "quality_score","defect_rate"]].round(2),
                     use_container_width=True, hide_index=True)

    # ── TAB 3: Supplier Segmentation ──────────────────────────────────────
    with tab3:
        st.markdown("### 🗂️ Supplier Segmentation — K-Means Clustering")
        st.markdown("""
        **Algorithm:** K-Means (k=4 clusters)  
        **Goal:** Segment suppliers into strategic tiers based on performance, spend, and reliability.  
        **Business Impact:** Enables tiered supplier management — focus strategic collaboration on top suppliers, and action plans on underperformers.
        """)
        seg_flt = segs[segs["supplier_id"].isin(flt_sids)]

        c1, c2 = st.columns(2)

        with c1:
            fig = px.scatter(seg_flt, x="avg_otd", y="avg_quality",
                             color="segment", size="total_spend",
                             hover_data=["name","avg_score","total_orders"],
                             title="Supplier Segments (OTD vs Quality)",
                             color_discrete_sequence=[ACCENT, BLUE, WARN, DANGER])
            st.plotly_chart(plotly_dark(fig, 380), use_container_width=True)

        with c2:
            seg_cnt = seg_flt["segment"].value_counts().reset_index()
            seg_cnt.columns = ["segment","count"]
            fig = px.pie(seg_cnt, names="segment", values="count",
                         title="Segment Distribution",
                         color_discrete_sequence=[ACCENT, BLUE, WARN, DANGER])
            st.plotly_chart(plotly_dark(fig, 380), use_container_width=True)

        st.markdown("**Supplier Segment Table**")
        disp = seg_flt[["name","category","segment","avg_score","avg_otd",
                         "avg_quality","total_spend","total_orders"]].round(2)
        disp.columns = ["Supplier","Category","Segment","Avg Score","OTD %","Quality","Total Spend","POs"]
        st.dataframe(disp.sort_values("Avg Score", ascending=False),
                     use_container_width=True, hide_index=True)

    # ── TAB 4: Performance Forecast ───────────────────────────────────────
    with tab4:
        st.markdown("### 📈 6-Month Performance Forecast — Random Forest Time Series")
        st.markdown("""
        **Algorithm:** Rolling-window Random Forest on time-series data  
        **Goal:** Predict each supplier's overall performance score for the next 6 months.  
        **Business Impact:** Allows procurement managers to proactively renegotiate contracts or seek alternative suppliers before performance degrades.
        """)
        fc_flt = fc[fc["supplier_id"].isin(flt_sids)]
        fc_flt = fc_flt.merge(suppliers[["supplier_id","name"]], on="supplier_id")

        sel_sup_ml = st.selectbox("Select Supplier for Forecast", flt_sup["name"].tolist())
        sel_sid_ml = flt_sup[flt_sup["name"]==sel_sup_ml]["supplier_id"].iloc[0]

        historical = perf[perf["supplier_id"]==sel_sid_ml][["eval_month","overall_score"]].rename(
            columns={"eval_month":"month","overall_score":"score"})
        historical["type"] = "Historical"

        forecast_rows = fc_flt[fc_flt["supplier_id"]==sel_sid_ml][["month","forecasted_score"]].rename(
            columns={"forecasted_score":"score"})
        forecast_rows["type"] = "Forecast"

        combined = pd.concat([historical, forecast_rows], ignore_index=True)

        fig = px.line(combined, x="month", y="score", color="type",
                      color_discrete_map={"Historical": ACCENT, "Forecast": WARN},
                      title=f"Performance Forecast: {sel_sup_ml}",
                      markers=True)
        fig.add_vrect(x0=historical["month"].iloc[-1], x1=forecast_rows["month"].iloc[-1],
                      fillcolor="rgba(244,162,97,0.06)", line_width=0,
                      annotation_text="Forecast Zone")
        st.plotly_chart(plotly_dark(fig, 380), use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Current Score",      f"{historical['score'].iloc[-1]:.1f}")
        c2.metric("6-Month Forecast",   f"{forecast_rows['score'].mean():.1f} avg")
