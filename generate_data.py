"""
P6 - Supplier Relationship Management
Database schema design + realistic sample data generator
Uses SQLite via SQLAlchemy
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "srm_database.db")

# ─── MASTER DATA ──────────────────────────────────────────────────────────────
SUPPLIER_NAMES = [
    "NexaTech Materials", "GlobalForge Ltd", "PrimeAlloy Co",
    "FastTrack Supplies", "TrueSource Inc", "Vertex Components",
    "Alliance Parts", "Ironclad Vendors", "StellarSupply Corp", "OmniTrade Solutions"
]
CATEGORIES = ["Raw Materials", "Packaging", "Electronics", "Mechanical Parts", "Chemicals"]
MATERIALS = {
    "Raw Materials":     ["Steel Sheets", "Aluminum Rods", "Copper Wire", "Polypropylene Pellets"],
    "Packaging":         ["Corrugated Boxes", "Bubble Wrap", "Stretch Film", "Foam Inserts"],
    "Electronics":       ["Capacitors", "Resistors", "PCB Boards", "Diodes"],
    "Mechanical Parts":  ["Bearings", "Bolts & Nuts", "Gaskets", "Shafts"],
    "Chemicals":         ["Lubricants", "Solvents", "Adhesives", "Coatings"],
}
REGIONS = ["North", "South", "East", "West", "Central"]
STATUSES = ["Active", "Active", "Active", "Under Review", "Blacklisted"]

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_schema(conn):
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS suppliers (
        supplier_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT NOT NULL,
        category        TEXT NOT NULL,
        region          TEXT NOT NULL,
        contact_email   TEXT,
        phone           TEXT,
        status          TEXT DEFAULT 'Active',
        onboarded_date  DATE,
        contract_expiry DATE,
        credit_score    INTEGER
    );

    CREATE TABLE IF NOT EXISTS materials (
        material_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name   TEXT NOT NULL,
        category        TEXT NOT NULL,
        unit            TEXT NOT NULL,
        unit_price      REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS supplier_materials (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id     INTEGER REFERENCES suppliers(supplier_id),
        material_id     INTEGER REFERENCES materials(material_id),
        lead_time_days  INTEGER,
        min_order_qty   INTEGER,
        negotiated_price REAL
    );

    CREATE TABLE IF NOT EXISTS purchase_orders (
        po_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id     INTEGER REFERENCES suppliers(supplier_id),
        material_id     INTEGER REFERENCES materials(material_id),
        order_date      DATE NOT NULL,
        expected_date   DATE NOT NULL,
        actual_date     DATE,
        quantity        INTEGER NOT NULL,
        unit_price      REAL NOT NULL,
        total_value     REAL NOT NULL,
        status          TEXT DEFAULT 'Pending',
        delay_days      INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS supplier_performance (
        perf_id             INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id         INTEGER REFERENCES suppliers(supplier_id),
        eval_month          TEXT NOT NULL,
        on_time_delivery    REAL,
        quality_score       REAL,
        defect_rate         REAL,
        response_time_hrs   REAL,
        fill_rate           REAL,
        cost_variance_pct   REAL,
        overall_score       REAL
    );

    CREATE TABLE IF NOT EXISTS communications (
        comm_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id     INTEGER REFERENCES suppliers(supplier_id),
        comm_date       DATE NOT NULL,
        channel         TEXT,
        subject         TEXT,
        response_days   INTEGER,
        resolved        INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS quality_incidents (
        incident_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id     INTEGER REFERENCES suppliers(supplier_id),
        material_id     INTEGER REFERENCES materials(material_id),
        incident_date   DATE NOT NULL,
        defect_qty      INTEGER,
        batch_qty       INTEGER,
        defect_type     TEXT,
        severity        TEXT,
        resolved        INTEGER DEFAULT 0,
        resolution_days INTEGER
    );
    """)
    conn.commit()

def generate_suppliers(conn):
    rows = []
    for i, name in enumerate(SUPPLIER_NAMES):
        cat = CATEGORIES[i % len(CATEGORIES)]
        onboarded = datetime(2018, 1, 1) + timedelta(days=random.randint(0, 1200))
        expiry = onboarded + timedelta(days=random.randint(730, 1460))
        rows.append((
            name, cat,
            random.choice(REGIONS),
            f"contact@{name.lower().replace(' ', '')}.com",
            f"+91-{random.randint(7000000000,9999999999)}",
            random.choice(STATUSES),
            onboarded.strftime("%Y-%m-%d"),
            expiry.strftime("%Y-%m-%d"),
            random.randint(600, 900)
        ))
    conn.executemany("""
        INSERT INTO suppliers (name,category,region,contact_email,phone,status,onboarded_date,contract_expiry,credit_score)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()

def generate_materials(conn):
    rows = []
    for cat, mats in MATERIALS.items():
        for mat in mats:
            unit = "kg" if cat in ["Raw Materials","Chemicals"] else "pcs"
            price = round(random.uniform(5, 500), 2)
            rows.append((mat, cat, unit, price))
    conn.executemany("""
        INSERT INTO materials (material_name,category,unit,unit_price) VALUES (?,?,?,?)
    """, rows)
    conn.commit()

def generate_supplier_materials(conn):
    suppliers = pd.read_sql("SELECT supplier_id, category FROM suppliers", conn)
    materials = pd.read_sql("SELECT material_id, category, unit_price FROM materials", conn)
    rows = []
    for _, sup in suppliers.iterrows():
        mats = materials[materials.category == sup.category]
        for _, mat in mats.iterrows():
            lead = random.randint(3, 30)
            moq = random.randint(50, 500)
            disc = random.uniform(0.85, 1.0)
            rows.append((int(sup.supplier_id), int(mat.material_id), lead, moq,
                         round(mat.unit_price * disc, 2)))
    conn.executemany("""
        INSERT INTO supplier_materials (supplier_id,material_id,lead_time_days,min_order_qty,negotiated_price)
        VALUES (?,?,?,?,?)
    """, rows)
    conn.commit()

def generate_purchase_orders(conn, n=400):
    sm = pd.read_sql("""
        SELECT sm.supplier_id, sm.material_id, sm.lead_time_days, sm.negotiated_price
        FROM supplier_materials sm
    """, conn)
    statuses = ["Delivered", "Delivered", "Delivered", "In Transit", "Pending", "Cancelled"]
    rows = []
    for _ in range(n):
        row = sm.sample(1).iloc[0]
        order_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))
        lead = int(row.lead_time_days)
        expected = order_date + timedelta(days=lead)
        delay = max(0, int(np.random.exponential(2)))
        actual = expected + timedelta(days=delay) if random.random() > 0.1 else None
        qty = random.randint(100, 2000)
        price = float(row.negotiated_price)
        st = random.choice(statuses)
        rows.append((
            int(row.supplier_id), int(row.material_id),
            order_date.strftime("%Y-%m-%d"),
            expected.strftime("%Y-%m-%d"),
            actual.strftime("%Y-%m-%d") if actual else None,
            qty, price, round(qty * price, 2),
            st, delay
        ))
    conn.executemany("""
        INSERT INTO purchase_orders
        (supplier_id,material_id,order_date,expected_date,actual_date,quantity,unit_price,total_value,status,delay_days)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()

def generate_performance(conn):
    suppliers = pd.read_sql("SELECT supplier_id FROM suppliers", conn)
    months = pd.date_range("2023-01", periods=24, freq="MS").strftime("%Y-%m").tolist()
    rows = []
    for _, sup in suppliers.iterrows():
        # Give each supplier a base profile, with drift over time
        base_otd   = random.uniform(0.70, 0.98)
        base_qual  = random.uniform(70, 98)
        base_defect= random.uniform(0.01, 0.08)
        for i, month in enumerate(months):
            trend = i * random.uniform(-0.002, 0.003)
            otd   = min(1.0, max(0.5, base_otd + trend + random.gauss(0, 0.03)))
            qual  = min(100, max(50, base_qual + trend*100 + random.gauss(0, 2)))
            defect= max(0, base_defect - trend*0.5 + random.gauss(0, 0.005))
            resp  = random.uniform(1, 72)
            fill  = min(1.0, max(0.6, otd + random.gauss(0, 0.05)))
            cost_var = random.gauss(0, 5)
            overall = round(0.35*otd*100 + 0.30*qual/100*100 + 0.20*fill*100
                            + 0.10*(100-defect*1000) + 0.05*(100-min(resp,72)/72*100), 2)
            rows.append((int(sup.supplier_id), month,
                         round(otd*100,2), round(qual,2), round(defect,4),
                         round(resp,1), round(fill*100,2), round(cost_var,2),
                         min(100, max(0, overall))))
    conn.executemany("""
        INSERT INTO supplier_performance
        (supplier_id,eval_month,on_time_delivery,quality_score,defect_rate,
         response_time_hrs,fill_rate,cost_variance_pct,overall_score)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()

def generate_communications(conn, n=300):
    suppliers = pd.read_sql("SELECT supplier_id FROM suppliers", conn)
    channels = ["Email", "Phone", "Portal", "Meeting"]
    subjects = ["Delivery Delay", "Price Negotiation", "Quality Issue", "Contract Renewal",
                "New Order", "Invoice Dispute", "Performance Review", "Emergency Procurement"]
    rows = []
    for _ in range(n):
        sid = int(suppliers.sample(1).iloc[0].supplier_id)
        date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))
        resp = random.randint(0, 14)
        rows.append((sid, date.strftime("%Y-%m-%d"),
                     random.choice(channels), random.choice(subjects),
                     resp, int(random.random() > 0.1)))
    conn.executemany("""
        INSERT INTO communications (supplier_id,comm_date,channel,subject,response_days,resolved)
        VALUES (?,?,?,?,?,?)
    """, rows)
    conn.commit()

def generate_quality_incidents(conn, n=150):
    sm = pd.read_sql("SELECT supplier_id, material_id FROM supplier_materials", conn)
    defect_types = ["Dimensional Error", "Surface Defect", "Material Contamination",
                    "Wrong Specification", "Packaging Damage"]
    severities = ["Low", "Medium", "High", "Critical"]
    rows = []
    for _ in range(n):
        row = sm.sample(1).iloc[0]
        date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))
        batch = random.randint(500, 5000)
        defects = random.randint(1, int(batch * 0.1))
        sev = random.choice(severities)
        res = int(random.random() > 0.2)
        res_days = random.randint(1, 30) if res else None
        rows.append((int(row.supplier_id), int(row.material_id),
                     date.strftime("%Y-%m-%d"), defects, batch,
                     random.choice(defect_types), sev, res, res_days))
    conn.executemany("""
        INSERT INTO quality_incidents
        (supplier_id,material_id,incident_date,defect_qty,batch_qty,defect_type,severity,resolved,resolution_days)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()

def build_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = get_connection()
    print("Creating schema...")
    create_schema(conn)
    print("Generating suppliers...")
    generate_suppliers(conn)
    print("Generating materials...")
    generate_materials(conn)
    print("Linking supplier-materials...")
    generate_supplier_materials(conn)
    print("Generating purchase orders (400)...")
    generate_purchase_orders(conn)
    print("Generating 24-month performance data...")
    generate_performance(conn)
    print("Generating communications (300)...")
    generate_communications(conn)
    print("Generating quality incidents (150)...")
    generate_quality_incidents(conn)
    conn.close()
    print(f"\n✅ Database ready at: {DB_PATH}")

if __name__ == "__main__":
    build_database()
