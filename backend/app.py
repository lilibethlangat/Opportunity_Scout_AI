from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'opportunity_model.joblib')
ZSCORE_PATH = os.path.join(BASE_DIR, 'models', 'zscore_params.csv')
STORE_PATH = os.path.join(BASE_DIR, 'saved_startups.json')

# Load model once at startup
_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


# --- JSON file store helpers ---
def load_store() -> list:
    """Load saved startups from JSON file. Returns empty list if file doesn't exist."""
    if not os.path.exists(STORE_PATH):
        return []
    with open(STORE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_store(records: list):
    """Write the full list of startups to the JSON file."""
    with open(STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)


app = FastAPI(title="Opportunity Scout AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Categories, countries, statuses used during training ---
ALL_CATEGORIES = [
    'AI & Machine Learning', 'Communications & Telecom', 'Consumer & Lifestyle',
    'Customer Service & CRM', 'Data & Analytics', 'E-Commerce & Retail',
    'Education & Training', 'Energy & Clean Tech', 'Enterprise & Business Services',
    'Finance & Fintech', 'Food & Beverage', 'Gaming', 'Government & Politics',
    'HR & Recruiting', 'Healthcare & Life Sciences', 'Internet & Web Services',
    'Manufacturing & Industrial', 'Marketing & Advertising', 'Media & Entertainment',
    'Real Estate & Construction', 'Security & Privacy', 'Social & Community',
    'Sports & Recreation', 'Technology & Software', 'Transportation & Logistics',
    'Travel & Hospitality'
]

ALL_COUNTRIES = [
    'Canada', 'China', 'France', 'Germany', 'India', 'Israel',
    'Other', 'Singapore', 'Spain', 'United Kingdom', 'United States'
]

ALL_STATUSES = ['acquired', 'closed', 'operating']
ALL_TIERS = ['unfunded', 'seed', 'series_a', 'series_b_plus']

def funding_to_tier(funding: float) -> str:
    if funding <= 0:
        return 'unfunded'
    elif funding < 1_000_000:
        return 'seed'
    elif funding < 10_000_000:
        return 'series_a'
    else:
        return 'series_b_plus'

def build_features(
    founded_year: int,
    funding_total_usd: float,
    funding_rounds: int,
    country: str,
    industry: str,
    # market defaults — median values from training data
    trend_slope: float = 0.0,
    news_volume: float = 30.0,
    news_sentiment: float = 0.6,
    reddit_density: float = 5.0,
    gdp_growth: float = 2.5,
    inflation: float = 3.0,
    startup_density: float = 0.5,
    status: str = 'operating',
) -> pd.DataFrame:
    """Build the full feature vector expected by the trained model."""

    # Load z-score params
    zp = pd.read_csv(ZSCORE_PATH).iloc[0]

    current_year = 2025
    company_age = current_year - founded_year
    tier = funding_to_tier(funding_total_usd)

    # Normalise country: anything not in our list → 'Other'
    if country not in ALL_COUNTRIES:
        country = 'Other'

    row = {
        # Raw features
        'founded_year': founded_year,
        'funding_total_usd': funding_total_usd,
        'funding_rounds': funding_rounds,
        'trend_slope': trend_slope,
        'news_volume': news_volume,
        'news_sentiment': news_sentiment,
        'reddit_density': reddit_density,
        'gdp_growth': gdp_growth,
        'inflation': inflation,
        'startup_density': startup_density,

        # Log transforms
        'log_funding': np.log1p(funding_total_usd),
        'funding_log': np.log1p(funding_total_usd),
        'news_volume_log': np.log1p(news_volume),
        'reddit_density_log': np.log1p(reddit_density),

        # Polynomial / interaction features
        'trend_slope_sq': trend_slope ** 2,
        'gdp_growth_sq': gdp_growth ** 2,
        'funding_sqrt': np.sqrt(max(funding_total_usd, 0)),
        'trend_x_gdp': trend_slope * gdp_growth,
        'trend_x_sentiment': trend_slope * news_sentiment,
        'reddit_x_sentiment': reddit_density * news_sentiment,
        'funding_x_rounds': funding_total_usd * funding_rounds,
        'trend_x_reddit': trend_slope * reddit_density,
        'gdp_x_density': gdp_growth * startup_density,
        'news_x_sentiment': news_volume * news_sentiment,

        # Composite indicators
        'econ_health': gdp_growth - inflation,
        'funding_per_round': funding_total_usd / max(funding_rounds, 1),
        'real_gdp_growth': gdp_growth - inflation,
        'investment_climate': gdp_growth * (1 / max(inflation, 0.1)),
        'ecosystem_strength': startup_density * gdp_growth,
        'market_validation': news_volume * reddit_density,

        # Time features
        'company_age': company_age,
        'pre_2010_flag': int(founded_year < 2010),
        'post_2015_flag': int(founded_year >= 2015),

        # Z-scores
        'trend_zscore': (trend_slope - zp['trend_slope_mean']) / max(zp['trend_slope_std'], 1e-9),
        'gdp_zscore': (gdp_growth - zp['gdp_growth_mean']) / max(zp['gdp_growth_std'], 1e-9),
        'reddit_zscore': (reddit_density - zp['reddit_density_mean']) / max(zp['reddit_density_std'], 1e-9),
    }

    # One-hot encode category
    for cat in ALL_CATEGORIES:
        row[f'cat_{cat}'] = int(industry == cat)

    # One-hot encode status
    for s in ALL_STATUSES:
        row[f'status_{s}'] = int(status == s)

    # One-hot encode country
    for c in ALL_COUNTRIES:
        row[f'country_{c}'] = int(country == c)

    # One-hot encode funding tier
    for t in ALL_TIERS:
        row[f'tier_{t}'] = int(tier == t)

    return pd.DataFrame([row])


def predict_score_and_reason(X: pd.DataFrame, model) -> tuple[float, str]:
    """Run the model and return (score, explanation)."""
    score = float(np.clip(model.predict(X)[0], 0, 100))
    importances = model.feature_importances_
    top_idx = int(np.argmax(X.iloc[0].values * importances))
    top_feature = X.columns[top_idx].replace('_', ' ').replace('cat ', '').title()
    return score, f"Driven by {top_feature}"


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
        model = get_model()
        records = load_store()

        if not records:
            return []

        # Score each saved record fresh with the ML model
        results = []
        for rec in records[-20:]:  # last 20 entries
            X = build_features(
                founded_year=int(rec.get('founded_year', 2015)),
                funding_total_usd=float(rec.get('funding_total_usd', 0)),
                funding_rounds=int(rec.get('funding_rounds', 1)),
                country=str(rec.get('country', 'Other')),
                industry=str(rec.get('industry', 'Technology & Software')),
            )
            score, explanation = predict_score_and_reason(X, model)
            results.append({
                'name': rec.get('name', 'Unknown'),
                'industry': rec.get('industry', ''),
                'funding_total_usd': rec.get('funding_total_usd', 0),
                'ai_predicted_score': round(score, 2),
                'explanation': explanation,
            })

        # Sort by score descending so highest scores appear first
        results.sort(key=lambda x: x['ai_predicted_score'], reverse=True)
        return results

    except Exception as e:
        return {"error": str(e)}


@app.post("/evaluate")
def evaluate_startup(data: StartupEval):
    try:
        model = get_model()
        X = build_features(
            founded_year=data.founded_year,
            funding_total_usd=data.funding_total_usd,
            funding_rounds=data.funding_rounds,
            country=data.country,
            industry=data.industry,
        )
        score, reason = predict_score_and_reason(X, model)

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
        model = get_model()
        X = build_features(
            founded_year=data.founded_year,
            funding_total_usd=data.funding_total_usd,
            funding_rounds=data.funding_rounds,
            country=data.country,
            industry=data.industry,
        )
        score, explanation = predict_score_and_reason(X, model)

        record = {
            'name': data.name,
            'founded_year': data.founded_year,
            'funding_total_usd': data.funding_total_usd,
            'funding_rounds': data.funding_rounds,
            'country': data.country,
            'industry': data.industry,
            'ai_predicted_score': round(score, 2),
            'explanation': explanation,
        }

        records = load_store()
        records.append(record)
        save_store(records)

        return {"status": "saved", "score": round(score, 2)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)