import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import joblib
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="SCM P6: Supplier Dashboard", layout="wide")

# 1. Load Data
conn = sqlite3.connect('SCM_P6_Database.db')
df_suppliers = pd.read_sql_query("SELECT * FROM Suppliers", conn)
df_pos = pd.read_sql_query("SELECT * FROM Purchase_Orders", conn)
df_del = pd.read_sql_query("SELECT * FROM Deliveries", conn)
df_mat = pd.read_sql_query("SELECT * FROM Materials", conn)
conn.close()

# Merge data for analysis
df_merged = df_pos.merge(df_del, on="PO_ID").merge(df_suppliers, on="Supplier_ID")

st.title("📦 Supplier Relationship Management Dashboard (P6)")
st.markdown("---")

# --- SECTION 1: PROCUREMENT SPEND ---
st.header("1. Procurement Spend Analysis")
# Calculate spend per category
df_spend = df_pos.merge(df_mat, on="Material_ID")
df_spend['Total_Cost'] = df_spend['Quantity_Ordered'] * df_spend['Unit_Price']
fig_spend = px.treemap(df_spend, path=['Material_Category', 'Material_ID'], values='Total_Cost', 
                       title="Spend by Category and Material")
st.plotly_chart(fig_spend, use_container_width=True)

# --- SECTION 2: SUPPLIER SCORECARD ---
st.header("2. Supplier Scorecard (Performance)")
# Calculate OTIF (On-Time In-Full) and Defect Rates
df_merged['On_Time'] = df_merged['Actual_Delivery_Date'] <= df_merged['Expected_Delivery_Date']
scorecard = df_merged.groupby('Supplier_Name').agg({
    'On_Time': 'mean',
    'Defects_Count': 'mean',
    'PO_ID': 'count'
}).rename(columns={'On_Time': 'OTIF %', 'Defects_Count': 'Avg Defects', 'PO_ID': 'Total Orders'})

scorecard['OTIF %'] = (scorecard['OTIF %'] * 100).round(2)
st.table(scorecard)

# --- SECTION 3: ML LEAD TIME PREDICTION ---
st.header("3. 🤖 AI Lead Time Predictor")
st.info("Predict how many days a delivery will take based on current trends.")

# Load the ML model you built in the previous step
model = joblib.load('lead_time_model.pkl')

col1, col2, col3 = st.columns(3)
with col1:
    sel_sup = st.selectbox("Select Supplier", df_suppliers['Supplier_ID'])
with col2:
    sel_mat = st.selectbox("Select Material", df_mat['Material_ID'])
with col3:
    sel_month = st.slider("Select Month of Order", 1, 12, 4)

# Factorize IDs as we did in training (simple mapping)
sup_idx = list(df_suppliers['Supplier_ID']).index(sel_sup)
mat_idx = list(df_mat['Material_ID']).index(sel_mat)

if st.button("Predict Lead Time"):
    prediction = model.predict([[sup_idx, mat_idx, sel_month]])
    days = round(prediction[0], 1)
    st.success(f"Predicted Lead Time: **{days} days**")
    
    expected_arrival = datetime.now() + timedelta(days=days)
    st.write(f"If ordered today, expected arrival: {expected_arrival.strftime('%Y-%m-%d')}")