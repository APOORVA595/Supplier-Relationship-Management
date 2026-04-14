import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Generate Suppliers (e.g., 10 suppliers)
suppliers = pd.DataFrame({
    'Supplier_ID': [f'S{i:03d}' for i in range(1, 11)],
    'Supplier_Name': ['Global Link', 'Prime Parts', 'Swift Supply', 'Nexus Corp', 'Alpha Mfg', 
                      'Beta Tech', 'Orion Ltd', 'Zenith Co', 'Apex Inc', 'Vertex Ltd'],
    'Supplier_Category': ['Raw Materials', 'Electronics', 'Packaging', 'Raw Materials', 'Chemicals',
                          'Electronics', 'Logistics', 'Raw Materials', 'Packaging', 'Chemicals'],
    'Location': ['Bangalore', 'Mumbai', 'Chennai', 'Delhi', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad', 'Surat', 'Jaipur'],
    'Contract_Tier': ['Gold', 'Silver', 'Bronze', 'Gold', 'Silver', 'Bronze', 'Gold', 'Silver', 'Bronze', 'Gold']
})

# 2. Generate Materials (e.g., 15 materials)
materials = pd.DataFrame({
    'Material_ID': [f'M{i:03d}' for i in range(1, 16)],
    'Material_Name': [f'Component_{i}' for i in range(1, 16)],
    'Material_Category': np.random.choice(['Category A', 'Category B', 'Category C'], 15),
    'Unit_Price': np.random.uniform(10, 500, 15).round(2),
    'Standard_Lead_Time_Days': np.random.randint(5, 20, 15)
})

# 3. Generate Purchase Orders & Deliveries (300 rows)
num_orders = 300
order_data = []
delivery_data = []

base_date = datetime(2025, 1, 1)

for i in range(1, num_orders + 1):
    po_id = f'PO{i:03d}'
    s_id = np.random.choice(suppliers['Supplier_ID'])
    m_id = np.random.choice(materials['Material_ID'])
    
    order_date = base_date + timedelta(days=np.random.randint(0, 400))
    qty = np.random.randint(50, 1000)
    
    # Logic for Lead Time Prediction (Person 1's ML Model)
    # We add a random delay to simulate real-world lead times
    std_lead = materials.loc[materials['Material_ID'] == m_id, 'Standard_Lead_Time_Days'].values[0]
    actual_lead = std_lead + np.random.randint(-2, 10) 
    
    expected_delivery = order_date + timedelta(days=std_lead.item())
    actual_delivery = order_date + timedelta(days=actual_lead.item())
    
    # Logic for Anomaly Detection (Person 2's ML Model)
    # Occasionally inject a huge delay or many defects
    defects = 0
    if np.random.random() < 0.1: # 10% chance of a performance issue
        actual_delivery += timedelta(days=20) # Huge delay
        defects = np.random.randint(20, 50) # High defects
    else:
        defects = np.random.randint(0, 5)

    order_data.append([po_id, s_id, m_id, order_date.date(), expected_delivery.date(), qty])
    delivery_data.append([f'DEL{i:03d}', po_id, actual_delivery.date(), qty - np.random.randint(0, 5), defects, 'Pass' if defects < 10 else 'Fail'])

po_df = pd.DataFrame(order_data, columns=['PO_ID', 'Supplier_ID', 'Material_ID', 'Order_Date', 'Expected_Delivery_Date', 'Quantity_Ordered'])
del_df = pd.DataFrame(delivery_data, columns=['Delivery_ID', 'PO_ID', 'Actual_Delivery_Date', 'Quantity_Received', 'Defects_Count', 'Quality_Status'])

# Save to CSV
suppliers.to_csv('suppliers.csv', index=False)
materials.to_csv('materials.csv', index=False)
po_df.to_csv('purchase_orders.csv', index=False)
del_df.to_csv('deliveries.csv', index=False)

print("300 rows of sample data generated successfully!")