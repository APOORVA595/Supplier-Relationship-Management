"""
ML Models for P6 - Supplier Relationship Management
1. Lead Time Forecasting (Random Forest)
2. Anomaly Detection on supplier performance (Isolation Forest)
3. Supplier Segmentation / Clustering (KMeans)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sqlite3
import os
import warnings
warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "srm_database.db")

def get_conn():
    return sqlite3.connect(DB_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Lead Time & Delay Forecasting (Random Forest Regressor)
# ─────────────────────────────────────────────────────────────────────────────
def train_delay_model():
    conn = get_conn()
    df = pd.read_sql("""
        SELECT po.delay_days, po.quantity, po.unit_price, po.total_value,
               sm.lead_time_days, sp.on_time_delivery, sp.quality_score,
               sp.defect_rate, sp.response_time_hrs, s.credit_score,
               CASE s.region
                 WHEN 'North' THEN 0 WHEN 'South' THEN 1
                 WHEN 'East' THEN 2 WHEN 'West' THEN 3 ELSE 4
               END as region_enc
        FROM purchase_orders po
        JOIN supplier_materials sm ON po.supplier_id=sm.supplier_id AND po.material_id=sm.material_id
        JOIN suppliers s ON po.supplier_id=s.supplier_id
        LEFT JOIN (
            SELECT supplier_id, AVG(on_time_delivery) as on_time_delivery,
                   AVG(quality_score) as quality_score,
                   AVG(defect_rate) as defect_rate,
                   AVG(response_time_hrs) as response_time_hrs
            FROM supplier_performance GROUP BY supplier_id
        ) sp ON po.supplier_id=sp.supplier_id
        WHERE po.status='Delivered'
    """, conn)
    conn.close()

    df = df.dropna()
    features = ["quantity","unit_price","total_value","lead_time_days",
                "on_time_delivery","quality_score","defect_rate",
                "response_time_hrs","credit_score","region_enc"]
    X = df[features]
    y = df["delay_days"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    return {
        "model": model,
        "features": features,
        "mae": round(mae, 2),
        "r2": round(r2, 3),
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Anomaly Detection on Supplier Performance (Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies():
    conn = get_conn()
    df = pd.read_sql("""
        SELECT sp.supplier_id, s.name, sp.eval_month,
               sp.on_time_delivery, sp.quality_score, sp.defect_rate,
               sp.response_time_hrs, sp.fill_rate, sp.cost_variance_pct, sp.overall_score
        FROM supplier_performance sp
        JOIN suppliers s ON sp.supplier_id=s.supplier_id
    """, conn)
    conn.close()

    features = ["on_time_delivery","quality_score","defect_rate",
                "response_time_hrs","fill_rate","cost_variance_pct"]
    X = df[features].fillna(df[features].mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.07, random_state=42)
    df["anomaly"] = iso.fit_predict(X_scaled)
    df["anomaly_score"] = iso.decision_function(X_scaled)
    df["is_anomaly"] = df["anomaly"] == -1

    return df[["supplier_id","name","eval_month","overall_score",
               "on_time_delivery","quality_score","defect_rate","is_anomaly","anomaly_score"]]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Supplier Segmentation using KMeans Clustering
# ─────────────────────────────────────────────────────────────────────────────
def segment_suppliers():
    conn = get_conn()
    df = pd.read_sql("""
        SELECT s.supplier_id, s.name, s.category, s.credit_score,
               AVG(sp.on_time_delivery) as avg_otd,
               AVG(sp.quality_score)    as avg_quality,
               AVG(sp.defect_rate)      as avg_defect,
               AVG(sp.overall_score)    as avg_score,
               AVG(sp.fill_rate)        as avg_fill,
               AVG(sp.cost_variance_pct) as avg_cost_var
        FROM suppliers s
        JOIN supplier_performance sp ON s.supplier_id=sp.supplier_id
        GROUP BY s.supplier_id
    """, conn)

    po = pd.read_sql("""
        SELECT supplier_id,
               COUNT(*) as total_orders,
               SUM(total_value) as total_spend,
               AVG(delay_days) as avg_delay
        FROM purchase_orders GROUP BY supplier_id
    """, conn)
    conn.close()

    df = df.merge(po, on="supplier_id", how="left").fillna(0)

    features = ["avg_otd","avg_quality","avg_defect","avg_score",
                "avg_fill","avg_cost_var","total_orders","total_spend","avg_delay","credit_score"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Label clusters meaningfully
    cluster_stats = df.groupby("cluster")[["avg_score","avg_otd","total_spend"]].mean()
    score_rank = cluster_stats["avg_score"].rank(ascending=False)
    label_map = {
        score_rank.idxmin():  "⭐ Strategic Partner",
        score_rank.nlargest(2).index[-1]: "✅ Reliable Supplier",
        score_rank.nsmallest(2).index[-1]: "⚠️ At-Risk Supplier",
        score_rank.idxmax():  "🔴 Underperformer",
    }
    df["segment"] = df["cluster"].map(label_map).fillna("✅ Reliable Supplier")

    return df, features, cluster_stats


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: 6-Month Performance Score Forecast per supplier
# ─────────────────────────────────────────────────────────────────────────────
def forecast_performance():
    conn = get_conn()
    df = pd.read_sql("""
        SELECT supplier_id, eval_month, overall_score
        FROM supplier_performance
        ORDER BY supplier_id, eval_month
    """, conn)
    conn.close()

    forecasts = []
    for sid, group in df.groupby("supplier_id"):
        scores = group["overall_score"].values
        if len(scores) < 6:
            continue
        # Use last 18 months to forecast next 6 using rolling RF
        n = len(scores)
        window = 3
        X_list, y_list = [], []
        for i in range(window, n):
            X_list.append(scores[i-window:i])
            y_list.append(scores[i])
        if len(X_list) < 5:
            continue
        X_arr = np.array(X_list)
        y_arr = np.array(y_list)
        m = RandomForestRegressor(n_estimators=50, random_state=42)
        m.fit(X_arr, y_arr)

        # Forecast next 6 steps
        preds = []
        current = list(scores[-window:])
        for _ in range(6):
            pred = m.predict([current])[0]
            preds.append(round(pred, 2))
            current = current[1:] + [pred]

        last_month = group["eval_month"].iloc[-1]
        months = pd.date_range(last_month, periods=7, freq="MS")[1:].strftime("%Y-%m").tolist()
        for mo, sc in zip(months, preds):
            forecasts.append({"supplier_id": sid, "month": mo, "forecasted_score": sc})

    return pd.DataFrame(forecasts)


if __name__ == "__main__":
    print("Training delay model...")
    dm = train_delay_model()
    print(f"  MAE={dm['mae']} days | R²={dm['r2']}")
    print("\nDetecting anomalies...")
    anom = detect_anomalies()
    print(f"  Anomalous records: {anom['is_anomaly'].sum()}")
    print("\nSegmenting suppliers...")
    segs, _, _ = segment_suppliers()
    print(segs[["name","segment"]])
    print("\nForecasting performance...")
    fc = forecast_performance()
    print(f"  Forecasted rows: {len(fc)}")
    print("\n✅ All ML models ready.")
