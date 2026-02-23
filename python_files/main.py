from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from sqlalchemy import create_engine

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Replace 'YOUR_PASSWORD'
DB_URL = "postgresql://postgres:password@localhost:5432/opportunity_scout"
engine = create_engine(DB_URL)

FEATURES = ['founded_year', 'funding_total_usd', 'funding_rounds', 'trend_slope', 
            'news_volume', 'news_sentiment', 'reddit_density', 'gdp_growth', 
            'inflation', 'startup_density']

class StartupEval(BaseModel):
    name: str
    founded_year: int
    funding_total_usd: float
    funding_rounds: int
    country: str
    industry: str

@app.get("/top-opportunities")
def get_dashboard_data():
    try:
        model = joblib.load('opportunity_model.joblib')
        df = pd.read_sql("SELECT * FROM startups ORDER BY id DESC LIMIT 20", engine)
        df.columns = [c.lower() for c in df.columns]
        
        # Create temporary X for the model using existing columns + defaults for market data
        X = pd.DataFrame()
        X['founded_year'] = df['founded_year']
        X['funding_total_usd'] = df['funding_total_usd']
        X['funding_rounds'] = df['funding_rounds']
        for feat in ['trend_slope', 'news_volume', 'news_sentiment', 'reddit_density', 'gdp_growth', 'inflation', 'startup_density']:
            X[feat] = 0.5

        scores = model.predict(X[FEATURES])
        df['ai_predicted_score'] = np.clip(scores, 0, 100)
        
        importances = model.feature_importances_
        explanations = []
        for _, row in X.iterrows():
            top_idx = np.argmax(row.values * importances)
            explanations.append(f"Strong {FEATURES[top_idx].replace('_', ' ').title()}")
        
        df['explanation'] = explanations
        return df[['name', 'industry', 'funding_total_usd', 'ai_predicted_score', 'explanation']].to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate")
def evaluate_startup(data: StartupEval):
    try:
        model = joblib.load('opportunity_model.joblib')
        # Map user input to model features
        input_vals = {
            'founded_year': data.founded_year,
            'funding_total_usd': data.funding_total_usd,
            'funding_rounds': data.funding_rounds,
            'trend_slope': 0.6, 'news_volume': 50, 'news_sentiment': 0.7,
            'reddit_density': 1.0, 'gdp_growth': 3.0, 'inflation': 4.0, 'startup_density': 0.5
        }
        X = pd.DataFrame([input_vals])[FEATURES]
        score = float(np.clip(model.predict(X)[0], 0, 100))
        
        # XAI Driver
        importances = model.feature_importances_
        top_idx = np.argmax(X.iloc[0].values * importances)
        reason = f"Driven by {FEATURES[top_idx].replace('_', ' ').title()}"
        
        # CRITICAL: These keys must match script.js exactly
        return {
            "name": data.name,
            "score": round(score, 2),
            "explanation": reason
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/save-startup")
def save_startup(data: StartupEval):
    try:
        df_save = pd.DataFrame([data.dict()])
        df_save['ai_predicted_score'] = 75.0 # Placeholder score for DB record
        df_save.to_sql('startups', engine, if_exists='append', index=False)
        return {"status": "saved"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)