"""
P6 - Supplier Relationship Management — INTEGRATED DASHBOARD
UE23CS342BA1 Supply Chain Management for Engineers

Integrates:
  Person 1: Spend treemap, OTIF scorecard, Lead Time Predictor (joblib model)
  Person 2: Full 6-page dashboard, 4 ML models, SQLite DB
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_models import (
    train_delay_model, detect_anomalies,
    segment_suppliers, forecast_performance
)

_HERE      = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(_HERE, "data", "srm_database.db")
MODEL_PATH = os.path.join(_HERE, "lead_time_model.pkl")

# ─── THEME ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SRM Dashboard | P6", page_icon="🔗",
                   layout="wide", initial_sidebar_state="expanded")

NAVY=  "#0D1B2A"; BLUE=  "#1B4F8A"; ACCENT="#00C9A7"
WARN=  "#F4A261"; DANGER="#E63946"; LIGHT= "#F8F9FA"; CARD= "#162032"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
  html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;background:{NAVY};color:{LIGHT};}}
  .main{{background:{NAVY};}} .block-container{{padding:1.5rem 2rem;}}
  h1,h2,h3{{font-family:'Syne',sans-serif;color:{LIGHT};}}
  .kpi-card{{background:{CARD};border-left:4px solid {ACCENT};border-radius:10px;padding:18px 20px;margin-bottom:10px;}}
  .kpi-card.warn{{border-left-color:{WARN};}} .kpi-card.danger{{border-left-color:{DANGER};}}
  .kpi-label{{font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;color:#8899aa;margin-bottom:4px;}}
  .kpi-value{{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{LIGHT};}}
  .kpi-sub{{font-size:.78rem;color:{ACCENT};margin-top:2px;}}
  .section-header{{font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{ACCENT};
    border-bottom:1px solid #1e3050;padding-bottom:6px;margin:18px 0 12px;
    text-transform:uppercase;letter-spacing:.08em;}}
  section[data-testid="stSidebar"]{{background:{CARD};}}
  section[data-testid="stSidebar"] *{{color:{LIGHT} !important;}}
  div[data-testid="metric-container"]{{background:{CARD};border-radius:8px;padding:10px;}}
</style>
""", unsafe_allow_html=True)


# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    out  = {t: pd.read_sql(f"SELECT * FROM {t}", conn) for t in
            ["suppliers","supplier_performance","purchase_orders",
             "communications","quality_incidents","materials","supplier_materials"]}
    conn.close()
    return out

@st.cache_data
def load_ml():
    dm            = train_delay_model()
    anom          = detect_anomalies()
    segs,feats,cs = segment_suppliers()
    fc            = forecast_performance()
    return dm, anom, segs, feats, cs, fc

@st.cache_resource
def load_pkl():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def dark(fig, h=320):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=LIGHT,family="DM Sans"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=10,r=10,t=30,b=10))
    fig.update_xaxes(gridcolor="#1e3050",zerolinecolor="#1e3050")
    fig.update_yaxes(gridcolor="#1e3050",zerolinecolor="#1e3050")
    return fig

def kpi(col, label, value, sub, cls=""):
    col.markdown(f"""<div class="kpi-card {cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

d          = load_data()
suppliers  = d["suppliers"];      perf      = d["supplier_performance"]
orders     = d["purchase_orders"];comms     = d["communications"]
incidents  = d["quality_incidents"];materials = d["materials"]
sm         = d["supplier_materials"]
dm, anom, segs, feats, cstats, fc = load_ml()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 SRM Dashboard\n**P6 · Supplier Relationship Mgmt**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Executive Overview",
        "🏭 Supplier Performance",
        "📦 Orders & Procurement",
        "🔬 Quality & Incidents",
        "💬 Communications",
        "🤖 ML Insights",
        "🧮 Lead Time Predictor",
    ])
    st.markdown("---")
    sel_reg = st.multiselect("Region",   suppliers["region"].unique(),   default=list(suppliers["region"].unique()))
    sel_cat = st.multiselect("Category", suppliers["category"].unique(), default=list(suppliers["category"].unique()))
    sel_sts = st.multiselect("Status",   suppliers["status"].unique(),   default=list(suppliers["status"].unique()))

flt_sup   = suppliers[suppliers["region"].isin(sel_reg)&suppliers["category"].isin(sel_cat)&suppliers["status"].isin(sel_sts)]
flt_sids  = flt_sup["supplier_id"].tolist()
flt_perf  = perf[perf["supplier_id"].isin(flt_sids)]
flt_ord   = orders[orders["supplier_id"].isin(flt_sids)].copy()
flt_comms = comms[comms["supplier_id"].isin(flt_sids)].copy()
flt_inc   = incidents[incidents["supplier_id"].isin(flt_sids)].copy()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview")
    k1,k2,k3,k4,k5 = st.columns(5)
    kpi(k1,"Total Suppliers",    len(flt_sup),                              f"{(flt_sup['status']=='Active').sum()} active")
    kpi(k2,"Avg Performance",    f"{flt_perf['overall_score'].mean():.1f}", "out of 100")
    kpi(k3,"On-Time Delivery",   f"{flt_perf['on_time_delivery'].mean():.1f}%","last 24 months")
    kpi(k4,"Total Spend",        f"₹{flt_ord['total_value'].sum()/1e6:.1f}M","all POs","warn")
    kpi(k5,"Open Incidents",     int((flt_inc['resolved']==0).sum()),        "unresolved","danger")

    st.markdown('<div class="section-header">Spend by Category & Material (Treemap)</div>', unsafe_allow_html=True)
    spend = flt_ord.merge(materials[["material_id","material_name","category"]], on="material_id")
    spend["total_cost"] = spend["quantity"]*spend["unit_price"]
    fig = px.treemap(spend, path=["category","material_name"], values="total_cost",
                     color_continuous_scale="Teal", title="Procurement Spend by Category & Material")
    st.plotly_chart(dark(fig, 380), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        trend = flt_perf.groupby("eval_month")["overall_score"].mean().reset_index()
        fig = px.area(trend,x="eval_month",y="overall_score",title="Avg Performance Score (24 Months)",
                      color_discrete_sequence=[ACCENT])
        fig.update_traces(fill='tozeroy',fillcolor="rgba(0,201,167,0.12)")
        st.plotly_chart(dark(fig,300), use_container_width=True)
    with c2:
        cs2 = spend.groupby("category")["total_cost"].sum().reset_index()
        fig = px.pie(cs2,names="category",values="total_cost",title="Spend Share by Category",
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_traces(textposition='inside',textinfo='percent+label')
        st.plotly_chart(dark(fig,300), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        sc = flt_sup["status"].value_counts().reset_index(); sc.columns=["status","count"]
        fig = px.bar(sc,x="status",y="count",color="status",title="Suppliers by Status",
                     color_discrete_map={"Active":ACCENT,"Under Review":WARN,"Blacklisted":DANGER})
        fig.update_layout(showlegend=False)
        st.plotly_chart(dark(fig,280), use_container_width=True)
    with c4:
        rp = flt_perf.merge(flt_sup[["supplier_id","region"]],on="supplier_id")
        rp = rp.groupby("region")["on_time_delivery"].mean().reset_index()
        fig = px.bar(rp,x="region",y="on_time_delivery",title="Avg OTD % by Region",
                     color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_range=[0,100])
        st.plotly_chart(dark(fig,280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SUPPLIER PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Supplier Performance":
    st.markdown("# 🏭 Supplier Performance")

    st.markdown('<div class="section-header">OTIF Scorecard</div>', unsafe_allow_html=True)
    otif = flt_ord.merge(flt_sup[["supplier_id","name"]],on="supplier_id")
    otif["on_time"] = otif["delay_days"]==0
    sc = otif.groupby("name").agg(
        OTIF_pct=("on_time","mean"), Total_Orders=("po_id","count"),
        Avg_Delay=("delay_days","mean"), Total_Spend=("total_value","sum")
    ).reset_index()
    sc["OTIF_pct"]=(sc["OTIF_pct"]*100).round(2); sc["Avg_Delay"]=sc["Avg_Delay"].round(2)
    sc.columns=["Supplier","OTIF %","Total Orders","Avg Delay (days)","Total Spend (₹)"]
    st.dataframe(sc.sort_values("OTIF %",ascending=False), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Individual Supplier KPIs</div>', unsafe_allow_html=True)
    sup_map  = dict(zip(flt_sup["name"],flt_sup["supplier_id"]))
    sel_name = st.selectbox("Select Supplier", list(sup_map.keys()))
    sel_sid  = sup_map[sel_name]
    sp       = perf[perf["supplier_id"]==sel_sid].sort_values("eval_month")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Score",       f"{sp['overall_score'].mean():.1f}")
    c2.metric("On-Time Delivery",f"{sp['on_time_delivery'].mean():.1f}%")
    c3.metric("Quality Score",   f"{sp['quality_score'].mean():.1f}")
    c4.metric("Avg Defect Rate", f"{sp['defect_rate'].mean()*100:.2f}%")

    fig = make_subplots(rows=2,cols=2,subplot_titles=["Overall Score","On-Time Delivery (%)","Quality Score","Defect Rate (%)"])
    kws = dict(mode="lines+markers",line_shape="spline")
    fig.add_trace(go.Scatter(x=sp.eval_month,y=sp.overall_score,    line=dict(color=ACCENT),**kws),row=1,col=1)
    fig.add_trace(go.Scatter(x=sp.eval_month,y=sp.on_time_delivery, line=dict(color=BLUE),  **kws),row=1,col=2)
    fig.add_trace(go.Scatter(x=sp.eval_month,y=sp.quality_score,    line=dict(color=WARN),  **kws),row=2,col=1)
    fig.add_trace(go.Scatter(x=sp.eval_month,y=sp.defect_rate*100,  line=dict(color=DANGER),**kws),row=2,col=2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(dark(fig,420), use_container_width=True)

    st.markdown('<div class="section-header">All Supplier KPI Heatmap</div>', unsafe_allow_html=True)
    heat = perf.merge(suppliers[["supplier_id","name"]],on="supplier_id")
    heat = heat.groupby("name")[["on_time_delivery","quality_score","fill_rate","overall_score"]].mean().round(1)
    fig  = px.imshow(heat,color_continuous_scale="Teal",title="Supplier KPI Heatmap")
    st.plotly_chart(dark(fig,360), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ORDERS & PROCUREMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Orders & Procurement":
    st.markdown("# 📦 Orders & Procurement")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total POs",   len(flt_ord))
    c2.metric("Total Spend", f"₹{flt_ord['total_value'].sum()/1e6:.2f}M")
    c3.metric("Avg Delay",   f"{flt_ord['delay_days'].mean():.1f} days")
    c4.metric("On-Time POs", f"{(flt_ord['delay_days']==0).mean()*100:.1f}%")

    c1,c2 = st.columns(2)
    with c1:
        sc = flt_ord["status"].value_counts().reset_index(); sc.columns=["status","count"]
        fig = px.pie(sc,names="status",values="count",title="PO Status",
                     color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(dark(fig,300), use_container_width=True)
    with c2:
        flt_ord["month"] = pd.to_datetime(flt_ord["order_date"]).dt.to_period("M").astype(str)
        mv = flt_ord.groupby("month")["total_value"].sum().reset_index()
        fig = px.bar(mv,x="month",y="total_value",title="Monthly Spend",color_discrete_sequence=[BLUE])
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(dark(fig,300), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        ds = flt_ord.merge(flt_sup[["supplier_id","name"]],on="supplier_id")
        ds = ds.groupby("name")["delay_days"].mean().reset_index().sort_values("delay_days",ascending=False)
        fig = px.bar(ds,x="name",y="delay_days",title="Avg Delay per Supplier",
                     color="delay_days",color_continuous_scale="Reds")
        fig.update_layout(xaxis_tickangle=-30,showlegend=False)
        st.plotly_chart(dark(fig,300), use_container_width=True)
    with c4:
        fig = px.histogram(flt_ord,x="delay_days",nbins=20,title="Delay Distribution",
                           color_discrete_sequence=[WARN])
        st.plotly_chart(dark(fig,300), use_container_width=True)

    st.markdown('<div class="section-header">Recent Purchase Orders</div>', unsafe_allow_html=True)
    disp = flt_ord.merge(flt_sup[["supplier_id","name"]],on="supplier_id")
    disp = disp.merge(materials[["material_id","material_name"]],on="material_id")
    disp = disp[["po_id","name","material_name","order_date","expected_date","status","quantity","total_value","delay_days"]]
    disp.columns=["PO ID","Supplier","Material","Order Date","Expected","Status","Qty","Value (₹)","Delay (d)"]
    st.dataframe(disp.sort_values("Order Date",ascending=False).head(30),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — QUALITY & INCIDENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Quality & Incidents":
    st.markdown("# 🔬 Quality & Incidents")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Incidents", len(flt_inc))
    c2.metric("Unresolved",      int((flt_inc["resolved"]==0).sum()))
    c3.metric("Avg Defect Rate", f"{(flt_inc['defect_qty']/flt_inc['batch_qty']).mean()*100:.2f}%")
    c4.metric("Critical Cases",  int((flt_inc["severity"]=="Critical").sum()))

    cm = {"Low":ACCENT,"Medium":WARN,"High":"#F4743B","Critical":DANGER}
    c1,c2 = st.columns(2)
    with c1:
        sv = flt_inc["severity"].value_counts().reset_index(); sv.columns=["severity","count"]
        fig = px.bar(sv,x="severity",y="count",color="severity",color_discrete_map=cm,title="By Severity")
        fig.update_layout(showlegend=False)
        st.plotly_chart(dark(fig,280), use_container_width=True)
    with c2:
        dt = flt_inc["defect_type"].value_counts().reset_index(); dt.columns=["type","count"]
        fig = px.pie(dt,names="type",values="count",title="Defect Types",
                     color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(dark(fig,280), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        ic = flt_inc.merge(flt_sup[["supplier_id","name"]],on="supplier_id")
        ic = ic.groupby("name").size().reset_index(name="incidents").sort_values("incidents",ascending=False)
        fig = px.bar(ic,x="name",y="incidents",title="Incidents per Supplier",color_discrete_sequence=[DANGER])
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(dark(fig,280), use_container_width=True)
    with c4:
        flt_inc["defect_pct"] = flt_inc["defect_qty"]/flt_inc["batch_qty"]*100
        fig = px.scatter(flt_inc,x="batch_qty",y="defect_pct",color="severity",
                         size="defect_qty",color_discrete_map=cm,title="Defect % vs Batch Size")
        st.plotly_chart(dark(fig,280), use_container_width=True)

    st.markdown('<div class="section-header">Open Incidents</div>', unsafe_allow_html=True)
    op = flt_inc[flt_inc["resolved"]==0].merge(flt_sup[["supplier_id","name"]],on="supplier_id")
    op = op.merge(materials[["material_id","material_name"]],on="material_id")
    st.dataframe(op[["incident_id","name","material_name","incident_date","defect_type","severity","defect_qty","batch_qty"]
        ].rename(columns={"incident_id":"ID","name":"Supplier","material_name":"Material","incident_date":"Date",
                           "defect_type":"Defect Type","severity":"Severity","defect_qty":"Defects","batch_qty":"Batch"}),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COMMUNICATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Communications":
    st.markdown("# 💬 Communications")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Comms",       len(flt_comms))
    c2.metric("Avg Response Time",  f"{flt_comms['response_days'].mean():.1f} days")
    c3.metric("Unresolved",         int((flt_comms["resolved"]==0).sum()))
    c4.metric("Resolution Rate",    f"{flt_comms['resolved'].mean()*100:.1f}%")

    c1,c2 = st.columns(2)
    with c1:
        ch = flt_comms["channel"].value_counts().reset_index(); ch.columns=["channel","count"]
        fig = px.pie(ch,names="channel",values="count",title="Channels",
                     color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(dark(fig,280), use_container_width=True)
    with c2:
        sb = flt_comms["subject"].value_counts().reset_index().head(6); sb.columns=["subject","count"]
        fig = px.bar(sb,x="count",y="subject",orientation="h",title="Top Topics",
                     color_discrete_sequence=[BLUE])
        st.plotly_chart(dark(fig,280), use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        rs = flt_comms.merge(flt_sup[["supplier_id","name"]],on="supplier_id")
        rs = rs.groupby("name")["response_days"].mean().reset_index().sort_values("response_days")
        fig = px.bar(rs,x="name",y="response_days",title="Avg Response Time per Supplier",
                     color_discrete_sequence=[WARN])
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(dark(fig,280), use_container_width=True)
    with c4:
        flt_comms["month"] = pd.to_datetime(flt_comms["comm_date"]).dt.to_period("M").astype(str)
        mv = flt_comms.groupby("month").size().reset_index(name="count")
        fig = px.line(mv,x="month",y="count",title="Monthly Communication Volume",
                      color_discrete_sequence=[ACCENT],markers=True)
        st.plotly_chart(dark(fig,280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Insights":
    st.markdown("# 🤖 ML Insights")
    st.markdown("*Four machine learning models applied to supplier relationship data*")
    tab1,tab2,tab3,tab4 = st.tabs(["🌲 Delay Forecasting","🚨 Anomaly Detection",
                                   "🗂️ Supplier Segmentation","📈 Performance Forecast"])
    with tab1:
        st.markdown("### Random Forest Regressor — Delivery Delay Prediction")
        st.markdown("**Goal:** Predict how many days a PO will be delayed. **Impact:** Proactively flag high-risk orders.")
        c1,c2 = st.columns(2); c1.metric("MAE",f"{dm['mae']} days"); c2.metric("R²",dm['r2'])
        c1,c2 = st.columns(2)
        with c1:
            fi = dm["feature_importances"].reset_index(); fi.columns=["Feature","Importance"]
            fig = px.bar(fi,x="Importance",y="Feature",orientation="h",title="Feature Importances",
                         color="Importance",color_continuous_scale="Teal")
            st.plotly_chart(dark(fig,350), use_container_width=True)
        with c2:
            pd_df = pd.DataFrame({"Actual":dm["y_test"].values,"Predicted":dm["y_pred"]})
            fig = px.scatter(pd_df,x="Actual",y="Predicted",title="Actual vs Predicted",
                             color_discrete_sequence=[ACCENT])
            fig.add_trace(go.Scatter(x=[0,20],y=[0,20],mode="lines",line=dict(dash="dash",color=WARN),name="Perfect"))
            st.plotly_chart(dark(fig,350), use_container_width=True)

    with tab2:
        st.markdown("### Isolation Forest — Anomaly Detection")
        st.markdown("**Goal:** Flag months of abnormal supplier performance. **Impact:** Early warning before failures.")
        af = anom[anom["supplier_id"].isin(flt_sids)]
        na = af["is_anomaly"].sum()
        c1,c2 = st.columns(2); c1.metric("Anomalies",int(na)); c2.metric("Rate",f"{na/len(af)*100:.1f}%")
        fig = px.scatter(af,x="eval_month",y="overall_score",color="is_anomaly",
                         color_discrete_map={True:DANGER,False:ACCENT},
                         hover_data=["name","on_time_delivery","quality_score"],
                         title="Performance Anomalies (Red = Flagged)",symbol="is_anomaly")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(dark(fig,380), use_container_width=True)
        st.dataframe(af[af["is_anomaly"]][["name","eval_month","overall_score","on_time_delivery",
                                           "quality_score","defect_rate"]].round(2),
                     use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### K-Means Clustering — Supplier Segmentation")
        st.markdown("**Goal:** Tier suppliers into Strategic/Reliable/At-Risk/Underperformer. **Impact:** Focused supplier management.")
        sf = segs[segs["supplier_id"].isin(flt_sids)]
        c1,c2 = st.columns(2)
        with c1:
            fig = px.scatter(sf,x="avg_otd",y="avg_quality",color="segment",size="total_spend",
                             hover_data=["name","avg_score"],title="Segments (OTD vs Quality)",
                             color_discrete_sequence=[ACCENT,BLUE,WARN,DANGER])
            st.plotly_chart(dark(fig,360), use_container_width=True)
        with c2:
            sc2 = sf["segment"].value_counts().reset_index(); sc2.columns=["segment","count"]
            fig = px.pie(sc2,names="segment",values="count",title="Segment Distribution",
                         color_discrete_sequence=[ACCENT,BLUE,WARN,DANGER])
            st.plotly_chart(dark(fig,360), use_container_width=True)
        disp = sf[["name","category","segment","avg_score","avg_otd","avg_quality","total_spend","total_orders"]].round(2)
        disp.columns=["Supplier","Category","Segment","Avg Score","OTD %","Quality","Total Spend","POs"]
        st.dataframe(disp.sort_values("Avg Score",ascending=False), use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("### Rolling Random Forest — 6-Month Performance Forecast")
        st.markdown("**Goal:** Predict each supplier's score 6 months ahead. **Impact:** Proactive contract management.")
        fc_f = fc[fc["supplier_id"].isin(flt_sids)].merge(suppliers[["supplier_id","name"]],on="supplier_id")
        sel  = st.selectbox("Supplier", flt_sup["name"].tolist())
        sid  = flt_sup[flt_sup["name"]==sel]["supplier_id"].iloc[0]
        hist = perf[perf["supplier_id"]==sid][["eval_month","overall_score"]].rename(columns={"eval_month":"month","overall_score":"score"})
        hist["type"]="Historical"
        fcast= fc_f[fc_f["supplier_id"]==sid][["month","forecasted_score"]].rename(columns={"forecasted_score":"score"})
        fcast["type"]="Forecast"
        comb = pd.concat([hist,fcast],ignore_index=True)
        fig  = px.line(comb,x="month",y="score",color="type",markers=True,
                       color_discrete_map={"Historical":ACCENT,"Forecast":WARN},title=f"Forecast: {sel}")
        fig.add_vrect(x0=hist["month"].iloc[-1],x1=fcast["month"].iloc[-1],
                      fillcolor="rgba(244,162,97,0.06)",line_width=0,annotation_text="Forecast Zone")
        st.plotly_chart(dark(fig,380), use_container_width=True)
        c1,c2=st.columns(2)
        c1.metric("Current Score",   f"{hist['score'].iloc[-1]:.1f}")
        c2.metric("6-Month Forecast",f"{fcast['score'].mean():.1f} avg")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — LEAD TIME PREDICTOR (Person 1's feature)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Lead Time Predictor":
    st.markdown("# 🧮 AI Lead Time Predictor")
    st.info("Predict delivery lead time based on supplier, material, and order details. Powered by Random Forest.")

    c1,c2,c3 = st.columns(3)
    with c1: sel_sup   = st.selectbox("Supplier",  flt_sup["name"].tolist())
    with c2: sel_mat   = st.selectbox("Material",  materials["material_name"].tolist())
    with c3: sel_month = st.slider("Month of Order", 1, 12, datetime.now().month)
    sel_qty = st.slider("Order Quantity", 100, 2000, 500, step=50)

    sid      = flt_sup[flt_sup["name"]==sel_sup]["supplier_id"].iloc[0]
    mid      = materials[materials["material_name"]==sel_mat]["material_id"].iloc[0]
    sup_row  = suppliers[suppliers["supplier_id"]==sid].iloc[0]
    mat_row  = materials[materials["material_id"]==mid].iloc[0]
    sm_row   = sm[(sm["supplier_id"]==sid)&(sm["material_id"]==mid)]
    lead_time= int(sm_row["lead_time_days"].iloc[0]) if len(sm_row) else 14
    price    = float(mat_row["unit_price"])
    pa       = perf[perf["supplier_id"]==sid].tail(6)
    otd      = pa["on_time_delivery"].mean()  if len(pa) else 85.0
    qual     = pa["quality_score"].mean()     if len(pa) else 80.0
    defect   = pa["defect_rate"].mean()       if len(pa) else 0.03
    resp     = pa["response_time_hrs"].mean() if len(pa) else 24.0
    credit   = int(sup_row["credit_score"])
    reg_enc  = {"North":0,"South":1,"East":2,"West":3,"Central":4}.get(sup_row["region"],2)
    features = [sel_qty, price, sel_qty*price, lead_time, otd, qual, defect, resp, credit, reg_enc]

    if st.button("🔮 Predict Lead Time", use_container_width=True):
        raw   = dm["model"].predict([features])[0]
        total = lead_time + max(0, round(raw, 1))
        eta   = datetime.now() + timedelta(days=total)

        r1,r2,r3 = st.columns(3)
        r1.metric("Base Lead Time",  f"{lead_time} days")
        r2.metric("Predicted Delay", f"+{max(0,round(raw,1))} days")
        r3.metric("Total ETA",       f"{int(total)} days")
        st.success(f"📅 Estimated arrival: **{eta.strftime('%d %B %Y')}**")

        risk = "🟢 Low Risk" if raw<2 else ("🟡 Medium Risk" if raw<5 else "🔴 High Risk")
        st.markdown(f"### Delivery Risk: {risk}")

        st.markdown('<div class="section-header">Supplier Context</div>', unsafe_allow_html=True)
        ctx = pd.DataFrame({
            "Metric":["On-Time Delivery","Quality Score","Defect Rate","Credit Score","Region"],
            "Value": [f"{otd:.1f}%", f"{qual:.1f}", f"{defect*100:.2f}%", credit, sup_row["region"]]
        })
        st.dataframe(ctx, use_container_width=True, hide_index=True)
