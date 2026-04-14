import sqlite3
import pandas as pd

# Connect to (or create) the database file
conn = sqlite3.connect('SCM_P6_Database.db')
cursor = conn.cursor()

# List of files we generated in Step 2
files = {
    'Suppliers': 'suppliers.csv',
    'Materials': 'materials.csv',
    'Purchase_Orders': 'purchase_orders.csv',
    'Deliveries': 'deliveries.csv'
}

print("Populating database...")

for table_name, csv_file in files.items():
    # Load the CSV into a temporary DataFrame
    df = pd.read_csv(csv_file)
    
    # Write the data to the SQLite database
    # This automatically creates the table structure based on the CSV headers
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Table '{table_name}' populated successfully.")

# Verify the data
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("\nTables currently in Database:", cursor.fetchall())

conn.close()
print("\nDatabase setup complete: SCM_P6_Database.db is ready.")