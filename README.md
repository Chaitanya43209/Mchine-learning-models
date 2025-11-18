# 🧠 Dynamic Pricing System (End-to-End ML Project)

### 🚀 Real-time Price Optimization using Machine Learning, FastAPI & Streamlit

This project implements a production-style **Dynamic Pricing System** that predicts demand and recommends optimal product prices to **maximize revenue in real-time** — similar to how Amazon, Uber, airlines, and hotels price dynamically.

---

## 🎯 Project Goals

- Predict how price affects demand (Price Elasticity Modeling)
- Maximize Expected Revenue = price × predicted demand
- Recommend optimal price in real-time
- Expose ML model using a FastAPI microservice
- Build an interactive Streamlit dashboard
- Demonstrate real-world MLOps architecture

---

## 🧠 Key Features

✔ Price elasticity estimation  
✔ Real-time demand prediction  
✔ Optimal price recommendation  
✔ Fully deployed FastAPI model service  
✔ Interactive Streamlit dashboard  
✔ Feature engineering with seasonality & rolling trends  
✔ Easily extendable to Airflow, Postgres, Docker & K8s  

---

## 📊 Business Context & KPIs

**Problem:** Static pricing causes loss in revenue or sales.  
**Solution:** Dynamically adjust prices based on demand.

### Business Metrics Tracked:
- Revenue lift
- ARPU
- Conversion rate
- Price elasticity
- Stockouts avoided
- Customer trust (no pricing shocks)

### Success Criteria:
- Higher revenue vs. baseline rules
- Stable elasticity estimates
- Low regret when deployed live


### Components:
- **Python / Pandas** – data prep
- **Scikit-learn** – Ridge Regression model
- **Joblib** – model persistence
- **FastAPI** – model serving
- **Uvicorn** – async web server
- **Streamlit** – interactive UI
- **Airflow (optional)** – automated retraining
- **PostgreSQL (optional)** – sales + price logging
- **Docker / Kubernetes (optional)** – scalable deployment

---

## 🧩 ML Model Details

- Log-Log Ridge Regression
- Predicts `log(quantity)` from `log(price)` and context
- Elasticity = slope of log(price)

### Features Used:
- log_price
- sin_doy (seasonality)
- cos_doy
- is_weekend
- rolling_price_7
- rolling_qty_7

---

## 🚀 How to Run Locally

### 1️⃣ Train the model
```bash
python train.py


python -m uvicorn app.main:app --reload --port 8001

streamlit run streamlit_app.py
{
  "best": { "price": 97.5, "pred_qty": 56.3, "rev": 5480.7 },
  "candidates": [...]
}

