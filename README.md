# Opportunity Scout AI 🚀

Opportunity Scout AI is a machine learning-powered platform designed to identify high-potential startup opportunities. It combines historical data, market trends, and real-time news sentiment to predict venture viability.

## 📁 Project Structure

```text
Opportunity_Scout_AI/
├── backend/            # FastAPI server, ML models, and data storage
│   ├── models/         # Trained XGBoost/LightGBM models (.joblib, .json)
│   ├── app.py          # Main API server
│   └── saved_startups.json  # Database for evaluated ventures
├── Frontend/           # Web interface (HTML, CSS, JS)
├── Data/               # Raw and processed datasets (CSV)
├── Notebooks/          # ML experiments and model training notebooks
└── scrapers/           # Data collection scripts (GDELT, Trends, TechCrunch)
```

## 🚀 Quick Start

### 1. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Start the FastAPI server:
   ```bash
   python app.py
   ```
   *The API will run at `http://127.0.0.1:8000`.*

### 2. Frontend Setup
1. Open `Frontend/index.html` in any modern web browser.
2. Click **Open Dashboard** to start exploring opportunities.

## ✨ Key Features
- **ML Scoring**: Uses a trained XGBoost model to predict startup success based on 10+ features.
- **XAI (Explainable AI)**: Identifies the primary driver behind Setiap viability score.
- **Interactive Dashboard**: Save and track evaluated ventures in real-time.
- **Market Scrapers**: Automated tools to fetch macro-economic and industry-specific data.

## 🛠 Tech Stack
- **Backend**: FastAPI, Pydantic, Scikit-Learn, Joblib, XGBoost
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **Data**: Pandas, NumPy