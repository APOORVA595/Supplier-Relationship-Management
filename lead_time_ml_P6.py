import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Load data from the database you just created
conn = sqlite3.connect('SCM_P6_Database.db')
query = """
SELECT po.Supplier_ID, po.Material_ID, po.Order_Date, del.Actual_Delivery_Date
FROM Purchase_Orders po
JOIN Deliveries del ON po.PO_ID = del.PO_ID
"""
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Preprocessing
# Calculate actual lead time in days
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Actual_Delivery_Date'] = pd.to_datetime(df['Actual_Delivery_Date'])
df['Lead_Time'] = (df['Actual_Delivery_Date'] - df['Order_Date']).dt.days

# Feature Engineering: Convert categorical IDs into numbers the ML model can understand
df['Supplier_ID_Factor'] = pd.factorize(df['Supplier_ID'])[0]
df['Material_ID_Factor'] = pd.factorize(df['Material_ID'])[0]
df['Month'] = df['Order_Date'].dt.month

# Define Features (X) and Target (y)
X = df[['Supplier_ID_Factor', 'Material_ID_Factor', 'Month']]
y = df['Lead_Time']

# 3. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Model Training Complete!")
print(f"Mean Absolute Error: {mae:.2f} days")

# 5. Save the model and the mappings for the dashboard
joblib.dump(model, 'lead_time_model.pkl')
print("Model saved as 'lead_time_model.pkl'")