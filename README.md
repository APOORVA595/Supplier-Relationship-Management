# P6: Supplier Relationship Management Dashboard

This project is a comprehensive Supply Chain Management (SCM) tool designed to improve supplier collaboration, track material performance, and predict delivery lead times using Machine Learning.

## 👥 Team Members
* **Apoorva Biradar** (PES1UG23CS095) - Database Design, Lead Time ML Model, Procurement Dashboard.
* **Aparajitha Chandan** (PES1UG23CS094) - Anomaly Detection, Supplier Risk Scoring, Performance Dashboard.

## 🚀 Features
* **Interactive Dashboard:** Built with Streamlit for real-time SCM insights.
* **AI Lead Time Predictor:** A Random Forest Regressor that predicts delivery dates with an accuracy (MAE) of 4.91 days.
* **Supplier Scorecard:** Track OTIF (On-Time In-Full) and defect rates across all vendors.
* **Spend Analysis:** Visual treemap of procurement costs by material category.
* **Anomaly Detection (Coming Soon):** Identifying outliers in delivery delays and quality failures.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Database:** SQLite3
* **Dashboard:** Streamlit, Plotly
* **Machine Learning:** Scikit-Learn, Joblib, Pandas, Numpy

## 📥 Installation & Setup

1. **Clone the repository:**
   git clone [https://github.com/APOORVA595/Supplier-Relationship-Management.git](https://github.com/APOORVA595/Supplier-Relationship-Management.git)
   cd Supplier-Relationship-Management

2. **Install dependencies:**
   pip install -r requirements.txt

3. **Run the Dashboard:**
   python -m streamlit run dashboard_P6.py