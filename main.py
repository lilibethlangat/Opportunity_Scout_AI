"""
main.py — Opportunity Scout AI
Run with: uvicorn main:app --reload --port 8000

New in this version:
  - user_type field added to all scoring requests ("entrepreneur" | "investor")
  - gdp_context returned in every /evaluate and /compare response
  - founding year no longer penalised — model uses market signals, not vintage age
  - Paths updated for final folder structure (Models/, Data/ subfolders)
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

from predictor import OpportunityPredictor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "Models", "ventures_lightgbm.joblib")
REFERENCE_PATH = os.path.join(BASE_DIR, "Data",   "venture_features.csv")
VENTURES_PATH  = os.path.join(BASE_DIR, "Data",   "ventures.csv")

predictor: OpportunityPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("Loading OpportunityPredictor...")
    predictor = OpportunityPredictor(
        model_path    = MODEL_PATH,
        reference_csv = REFERENCE_PATH,
        ventures_csv  = VENTURES_PATH,
    )
    print("Ready — SHAP explainer active.")
    yield
    print("Shutting down.")


app = FastAPI(title="Opportunity Scout AI", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class StartupEval(BaseModel):
    name:              str   = "My Venture"
    industry:          str
    country:           str
    founded_year:      int
    funding_total_usd: float = 0.0
    funding_rounds:    int   = 1
    user_type:         str   = Field(default="investor",
                                     description="'entrepreneur' or 'investor'")


class WhatIfRequest(BaseModel):
    base:    StartupEval
    changes: dict = {}


def _format(name: str, result: dict) -> dict:
    return {
        "name":              name,
        "score":             result["score"],
        "tier":              result["category"],
        "confidence":        result["confidence"],
        "score_range":       result["score_range"],
        "base_value":        result["base_value"],
        "top_factors":       result["top_factors"],
        "recommendations":   result["recommendations"],
        "benchmark":         result["benchmark"],
        "gdp_context":       result.get("gdp_context", ""),
        "user_type":         result.get("user_type", "investor"),
        "data_quality":      result.get("data_quality",      0.5),
        "quality_label":     result.get("quality_label",     "Medium"),
        "quality_breakdown": result.get("quality_breakdown", {}),
        "live_signals":      result.get("live_signals",      []),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/api/v1/industries")
def get_industries():
    ref = pd.read_csv(REFERENCE_PATH)
    return sorted(ref["category"].dropna().unique().tolist())


@app.get("/api/v1/countries")
def get_countries():
    ref = pd.read_csv(REFERENCE_PATH)
    return sorted(ref["country"].dropna().unique().tolist())


@app.post("/evaluate")
def evaluate(data: StartupEval):
    if predictor is None:
        raise HTTPException(503, "Model not loaded yet.")
    try:
        result = predictor.predict(
            industry       = data.industry,
            country        = data.country,
            year           = data.founded_year,
            funding        = data.funding_total_usd,
            funding_rounds = data.funding_rounds,
            user_type      = data.user_type,
        )
        return _format(data.name, result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/what-if")
def what_if(req: WhatIfRequest):
    if predictor is None:
        raise HTTPException(503, "Model not loaded yet.")
    try:
        b = req.base
        base_r = predictor.predict(b.industry, b.country, b.founded_year,
                                   b.funding_total_usd, b.funding_rounds, b.user_type)
        c = req.changes
        mod_r = predictor.predict(
            c.get("industry", b.industry), c.get("country", b.country),
            c.get("founded_year", b.founded_year),
            c.get("funding_total_usd", b.funding_total_usd),
            c.get("funding_rounds", b.funding_rounds), b.user_type,
        )
        return {
            "original":    _format(b.name, base_r),
            "modified":    _format(b.name + " (modified)", mod_r),
            "score_delta": round(mod_r["score"] - base_r["score"], 2),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/compare")
def compare(opportunities: list[StartupEval]):
    if predictor is None:
        raise HTTPException(503, "Model not loaded yet.")
    try:
        results = []
        for opp in opportunities:
            r = predictor.predict(opp.industry, opp.country, opp.founded_year,
                                  opp.funding_total_usd, opp.funding_rounds, opp.user_type)
            results.append(_format(opp.name or opp.industry, r))
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"results": results, "winner": results[0]["name"]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/top-opportunities")
def top_opportunities():
    try:
        df = pd.read_csv(VENTURES_PATH)
        score_col = next((c for c in ["opportunity_score", "score"] if c in df.columns), None)
        if not score_col:
            return {"error": "No score column found."}
        df = df.dropna(subset=[score_col, "name", "category", "country"]).copy()
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        top = df.dropna(subset=[score_col]).nlargest(20, score_col)
        rows = []
        for _, row in top.iterrows():
            score = round(float(row[score_col]), 2)
            rows.append({
                "name":              str(row.get("name", "—")),
                "industry":          str(row.get("category", "—")),
                "country":           str(row.get("country", "—")),
                "founded_year":      int(row["founded_year"]) if pd.notna(row.get("founded_year")) else "—",
                "funding_total_usd": float(row.get("funding_total_usd", 0)),
                "ai_predicted_score": score,
                "tier": ("High Potential" if score >= 70 else
                         "Medium Potential" if score >= 40 else "Lower Potential"),
                "explanation": "Pre-computed composite opportunity score",
            })
        return rows
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/save-startup")
def save_startup(data: StartupEval):
    if predictor is None:
        raise HTTPException(503, "Model not loaded yet.")
    try:
        result = predictor.predict(data.industry, data.country, data.founded_year,
                                   data.funding_total_usd, data.funding_rounds, data.user_type)
        save_path = os.path.join(BASE_DIR, "saved_startups.csv")
        row = pd.DataFrame([{
            "name": data.name, "industry": data.industry,
            "country": data.country, "founded_year": data.founded_year,
            "funding_total_usd": data.funding_total_usd,
            "user_type": data.user_type,
            "ai_predicted_score": result["score"], "tier": result["category"],
        }])
        if os.path.exists(save_path):
            pd.concat([pd.read_csv(save_path), row], ignore_index=True).to_csv(save_path, index=False)
        else:
            row.to_csv(save_path, index=False)
        return {"status": "saved", "score": result["score"]}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Serve Frontend ─────────────────────────────────────────────────────────────
frontend_path = os.path.join(BASE_DIR, "Frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
