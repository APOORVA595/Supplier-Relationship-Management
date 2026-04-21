# P6 - Supplier Relationship Management Dashboard
## UE23CS342BA1 — Supply Chain Management for Engineers

### Project Structure
```
scm_p6/
├── app.py              ← Main Streamlit dashboard
├── generate_data.py    ← Database schema + sample data generator
├── ml_models.py        ← All 4 ML models
├── requirements.txt
└── data/
    └── srm_database.db ← SQLite database (auto-generated)
```

### Database Schema (7 Tables)
- **suppliers** — Master supplier registry (10 suppliers, 5 categories, 5 regions)
- **materials** — Material catalog (20 items across 5 categories)
- **supplier_materials** — Supplier-material linkage with lead times & negotiated prices
- **purchase_orders** — 400 POs with order/expected/actual dates, delay tracking
- **supplier_performance** — 24-month KPI data per supplier (OTD, quality, defect rate)
- **communications** — 300 interaction records (channel, subject, response time)
- **quality_incidents** — 150 defect incidents with severity and resolution tracking

### ML Models
| # | Model | Algorithm | Purpose |
|---|-------|-----------|---------|
| 1 | Delay Forecasting | Random Forest Regressor | Predict PO delay days |
| 2 | Anomaly Detection | Isolation Forest | Flag abnormal supplier performance |
| 3 | Supplier Segmentation | K-Means (k=4) | Tier suppliers strategically |
| 4 | Performance Forecast | Rolling RF Time Series | 6-month score prediction |

### Dashboard Pages (6)
1. 📊 Executive Overview — KPIs, spend trends, OTD by region
2. 🏭 Supplier Performance — Per-supplier KPI drill-down, heatmap, lead times
3. 📦 Orders & Procurement — PO status, monthly spend, delay analysis
4. 🔬 Quality & Incidents — Defect breakdown, severity, open incidents
5. 💬 Communications — Channel analytics, response times, volume trends
6. 🤖 ML Insights — All 4 ML model outputs with explanations

### Setup & Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate database (only needed once)
python generate_data.py

# 3. Launch dashboard
streamlit run app.py
```

Dashboard opens at: http://localhost:8501
