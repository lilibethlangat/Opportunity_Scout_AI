"""
predictor.py
------------
Full prediction pipeline: live data → feature engineering → LightGBM → SHAP.

Changes:
  - Added user_type parameter ("entrepreneur" | "investor") to predict()
  - user_type is passed to pipeline.run_live() for weighted scoring
  - GDP context explanation added to every result (from static table + API fallback)
  - Time-feature references removed from SHAP explanations and recommendations
  - Score now reflects market conditions, not vintage year penalties
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import joblib
import shap

from pipeline import AggregationEngine
from feature_engineer import VentureFeatureEngineer

logger = logging.getLogger("predictor")


# ── GDP Context Table (static lookup — 100+ country+year events) ──────────────

GDP_CONTEXT_TABLE = {
    ("Latvia", 2009): "Latvia's GDP collapsed by 16% as the global financial crisis burst a credit-fuelled property bubble, triggering the worst recession in its post-Soviet history.",
    ("Lithuania", 2009): "Lithuania's GDP fell nearly 15% as the global financial crisis devastated export demand and a domestic credit bubble collapsed simultaneously.",
    ("Estonia", 2009): "Estonia's GDP dropped 14.6% due to the global financial crisis unwinding its overheated credit and real estate boom.",
    ("Estonia", 2008): "Estonia's GDP began contracting in 2008 as rising global risk aversion and a domestic credit bubble started to burst, foreshadowing the severe 2009 recession.",
    ("Estonia", 2006): "Estonia posted 9.8% growth driven by EU accession-fuelled foreign investment, a booming property market and rapid credit expansion.",
    ("Estonia", 2011): "Estonia rebounded with 7.6% growth as export competitiveness, restored after painful wage cuts in 2009, powered a strong recovery.",
    ("Ukraine", 2009): "Ukraine's GDP fell 15% as the global financial crisis collapsed steel export prices and a banking crisis froze credit across the economy.",
    ("Ukraine", 2014): "Ukraine's GDP contracted 10% following the Euromaidan revolution, Russia's annexation of Crimea and the outbreak of conflict in the Donbas.",
    ("Greece", 2009): "Greece's GDP fell 4.1% as the global financial crisis exposed its chronically high deficit; the year marked the start of a decade-long economic depression.",
    ("Greece", 2010): "Greece's GDP contracted 5.7% as the country entered an EU-IMF bailout programme and began severe austerity measures.",
    ("Greece", 2011): "Greece's economy shrank nearly 10% as its second bailout was negotiated and a private-sector debt restructuring — the largest in history — caused widespread disruption.",
    ("Greece", 2012): "Greece's GDP fell another 8.3% as austerity deepened, unemployment reached 27% and domestic demand collapsed.",
    ("Greece", 2013): "Greece's economy contracted a further 2.3% as austerity continued, though structural reforms began to stabilise public finances.",
    ("Iceland", 2009): "Iceland's GDP fell 8.3% after its three major banks — with liabilities ten times GDP — collapsed in 2008, triggering a currency crisis and IMF bailout.",
    ("Finland", 2009): "Finland's GDP plunged 8.1% as the global financial crisis hammered its export-dependent economy, particularly electronics and paper manufacturing.",
    ("Germany", 2009): "Germany's GDP fell 5.5% — its worst post-war recession — as global trade collapsed and its export-driven industrial base faced plummeting demand.",
    ("Ireland", 2009): "Ireland's GDP fell 5.1% as its banking crisis deepened, the government nationalised major banks and an EU-IMF bailout became necessary.",
    ("Ireland", 2014): "Ireland's GDP surged 9.4% as US multinationals shifted IP assets into Ireland, significantly inflating measured output.",
    ("Italy", 2009): "Italy's GDP fell 5.3% as the global financial crisis hit its manufacturing sector and structural weaknesses amplified the shock.",
    ("Italy", 2012): "Italy's GDP contracted 3.1% as EU-mandated austerity compressed domestic demand.",
    ("Spain", 2009): "Spain's GDP fell 3.8% as a massive property bust wiped out its construction sector and unemployment surged toward 20%.",
    ("Spain", 2012): "Spain's GDP contracted 2.9% as EU austerity conditions attached to its banking-sector bailout suppressed investment.",
    ("Portugal", 2012): "Portugal's GDP fell 4% as austerity measures imposed under the EU-IMF bailout severely reduced public investment and household incomes.",
    ("Hungary", 2009): "Hungary's GDP fell 6.7% as the global financial crisis hit its export-dependent economy and Swiss franc-denominated mortgages created a household debt crisis.",
    ("Japan", 2009): "Japan's GDP fell 5.7% as global trade collapsed, sharply reducing demand for its exports, and deflationary pressures intensified.",
    ("United Kingdom", 2009): "The UK's GDP fell 4.6% as the global financial crisis caused a severe banking crisis requiring part-nationalisation of RBS and Lloyds.",
    ("Canada", 2009): "Canada's GDP contracted 2.9% as the global financial crisis reduced US demand for its exports, particularly in automotive.",
    ("United States", 2009): "US GDP fell 2.6% in the worst recession since the Great Depression, triggered by the subprime mortgage collapse and the Lehman Brothers bankruptcy.",
    ("Mexico", 2009): "Mexico's GDP fell 6.3% — one of the sharpest in Latin America — due to deep trade integration with the US and the simultaneous H1N1 swine flu outbreak.",
    ("China", 2005): "China grew 11.5% driven by massive export expansion following WTO accession, a construction boom and strong FDI inflows.",
    ("China", 2006): "China's GDP expanded 12.7% as export growth accelerated, infrastructure investment surged and urbanisation drove exceptional demand.",
    ("China", 2007): "China grew 14.2% — its fastest in decades — fuelled by a global commodity super-cycle, soaring exports and government infrastructure spending.",
    ("China", 2008): "China's GDP slowed to 9.7% as the global financial crisis reduced export demand in the second half of the year.",
    ("China", 2009): "China grew 9.4% — bucking the global recession — as a massive government stimulus worth 12.5% of GDP fuelled infrastructure and consumption.",
    ("China", 2010): "China expanded 10.6% as its stimulus-fuelled recovery matured and exports rebounded strongly.",
    ("China", 2012): "China's growth eased to 7.9% as the government deliberately slowed credit to rebalance toward consumption.",
    ("China", 2014): "China's GDP expanded 7.5% — slowest in 24 years — as property cooled and the government signalled a 'new normal' of lower but higher-quality growth.",
    ("India", 2005): "India grew 7.9% as economic liberalisation, a booming IT sector and strong FDI accelerated its emergence as a global outsourcing hub.",
    ("India", 2009): "India's GDP grew 7.9% despite the global financial crisis as its relatively closed capital account limited contagion.",
    ("India", 2010): "India expanded 8.5% as post-crisis recovery and strong domestic consumption pushed growth back toward its pre-crisis trend.",
    ("India", 2014): "India grew 7.4% as the newly elected Modi government's reforms boosted confidence and inflation eased.",
    ("Singapore", 2010): "Singapore grew an exceptional 14.5% — its fastest on record — as a sharp post-crisis rebound in electronics coincided with the opening of two major integrated resorts.",
    ("Kenya", 2007): "Kenya grew 6.9% on strong tourism, horticulture exports and a construction boom.",
    ("Kenya", 2010): "Kenya's GDP expanded 8.1% as the economy rebounded from post-election violence, driven by telecoms, trade and construction.",
    ("Nigeria", 2009): "Nigeria grew 8% despite the global financial crisis as high oil output and a rebound in agriculture kept growth resilient.",
    ("Nigeria", 2013): "Nigeria grew 6.7% as GDP was formally rebased confirming it as Africa's largest economy, with strong services sector growth.",
    ("South Africa", 2009): "South Africa's GDP contracted 1.5% — its first recession in 17 years — as the global financial crisis reduced demand for its mineral exports.",
    ("Brazil", 2010): "Brazil's GDP surged 7.5% — fastest in 25 years — as a commodity boom, consumer credit expansion and pre-World Cup infrastructure spending converged.",
    ("United Arab Emirates", 2009): "The UAE's GDP fell 5.2% as the global financial crisis triggered a real estate collapse in Dubai and Dubai World's near-default required an Abu Dhabi bailout.",
    ("United Arab Emirates", 2011): "The UAE's GDP grew 6.7% as Abu Dhabi's oil revenues surged and Dubai's economy recovered, with tourism and trade expanding strongly.",
    ("Indonesia", 2010): "Indonesia expanded 6.2% as a post-crisis rebound in commodity exports and rising domestic consumption restored its pre-crisis growth trajectory.",
    ("Thailand", 2010): "Thailand's GDP surged 7.5% as post-crisis export recovery — particularly automotive and electronics — rebounded sharply.",
    ("Philippines", 2010): "The Philippines grew 7.3% as BPO exports boomed, remittances hit record levels and post-crisis pent-up consumer demand was released.",
    ("Israel", 2007): "Israel grew 6.3% as its technology sector boomed, attracting record venture capital investment and fuelling broad economic expansion.",
    ("Russian Federation", 2009): "Russia's GDP fell 7.8% as the global financial crisis caused oil prices to plummet, capital fled and domestic credit markets seized up.",
    ("Russian Federation", 2006): "Russia grew 8.2% driven by high oil prices, rising domestic consumption and strong FDI inflows that followed political stabilisation.",
    ("Panama", 2011): "Panama grew 11.9% as the Panama Canal expansion project reached peak construction, attracting massive infrastructure investment.",
    ("Panama", 2012): "Panama's GDP expanded 9.6% as Canal expansion continued and its financial services sector attracted strong regional FDI.",
}


def get_gdp_context(country: str, year: int, gdp_value: float) -> str:
    """
    Returns a one-sentence explanation of why a country's GDP moved in a given year.
    Tries: (1) static table → (2) Anthropic API → (3) plain data fallback.
    """
    # Step 1: static table
    key = (country, year)
    if key in GDP_CONTEXT_TABLE:
        return GDP_CONTEXT_TABLE[key]

    # Step 2: Anthropic API (requires ANTHROPIC_API_KEY env var)
    direction = "grew" if gdp_value >= 0 else "contracted"
    magnitude = abs(round(gdp_value, 1))
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            f"In exactly one sentence, explain the primary economic, political or external reason "
            f"why {country}'s GDP {direction} by {magnitude}% in {year}. "
            f"Write for an investor or entrepreneur evaluating a startup in that market. "
            f"Be specific and factual — name the actual event or driver. No preamble."
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception:
        pass

    # Step 3: plain fallback
    direction_word = "grew" if gdp_value >= 0 else "contracted"
    return (
        f"{country}'s GDP {direction_word} by {abs(round(gdp_value, 1))}% in {year}; "
        f"no specific historical context is available for this market at this time."
    )


# ── Score calibration ──────────────────────────────────────────────────────────

def calibrate_score(raw: float) -> dict:
    score = float(np.clip(raw, 0, 100))
    if   score >= 70: cat, conf = "High Potential",   0.90
    elif score >= 40: cat, conf = "Medium Potential", 0.85
    else:             cat, conf = "Lower Potential",  0.75
    return {"score": round(score, 2), "category": cat, "confidence": conf,
            "score_range": (round(score - 4, 2), round(score + 4, 2))}


# ── Predictor ──────────────────────────────────────────────────────────────────

class OpportunityPredictor:
    def __init__(self, model_path: str, reference_csv: str, ventures_csv: str):
        arts               = joblib.load(model_path)
        self.model         = arts["model"]
        self.feature_names = arts["feature_names"]
        self.zscore_params = arts["zscore_params"]
        self.reference_df  = pd.read_csv(reference_csv)
        self.explainer     = shap.TreeExplainer(self.model)
        self.base_value    = float(self.explainer.expected_value)
        self.pipeline      = AggregationEngine(ventures_csv)
        logger.info("OpportunityPredictor ready (SHAP explainer active)")

    def _build_row(self, industry: str, country: str, year: int,
                   funding: float, funding_rounds: int, signals: dict) -> pd.DataFrame:
        """
        Seed a feature row from the nearest reference row for this industry,
        then overwrite with live signals and user inputs.
        """
        ref = self.reference_df[
            self.reference_df["category"].str.lower() == industry.lower()
        ]
        if ref.empty:
            ref = self.reference_df

        # Find closest year match — no year clipping (year penalty removed)
        idx = (ref["founded_year"] - year).abs().idxmin()
        row = ref.loc[[idx]].copy()

        # User-controlled inputs
        row["funding_total_usd"] = funding
        row["funding_rounds"]    = funding_rounds
        row["country"]           = country
        row["founded_year"]      = year

        # Live/fallback market signals override the reference row
        for col in ["trend_slope", "gdp_growth", "inflation", "reddit_density",
                    "news_volume", "news_sentiment", "startup_density"]:
            v = signals.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                row[col] = float(v)
        return row

    def _engineer(self, row: pd.DataFrame) -> pd.DataFrame:
        return VentureFeatureEngineer(
            model_features=self.feature_names,
            zscore_params=self.zscore_params,
        ).transform(row)

    def predict(self, industry: str, country: str,
                year: int, funding: float,
                funding_rounds: int = 1,
                user_type: str = "investor") -> dict:
        """
        Full prediction pipeline.

        Parameters
        ----------
        industry       : industry/category string
        country        : country string
        year           : founding year (any year — no penalty applied)
        funding        : total funding in USD
        funding_rounds : number of funding rounds (default 1)
        user_type      : "entrepreneur" or "investor"
                         Controls which weight profile the pipeline uses
                         for its composite score, and which lens the
                         SHAP explanations use.
        """
        if user_type not in ("entrepreneur", "investor"):
            user_type = "investor"

        # 1. Live data collection
        pr        = self.pipeline.run_live(industry, country, year,
                                           funding=funding, user_type=user_type)
        signals   = pr["signals"]
        live_sigs = pr.get("live_signals", {})

        # 2–4. Feature engineering + model prediction
        row = self._build_row(industry, country, year, funding, funding_rounds, signals)
        X   = self._engineer(row)
        # Neutralise founding-year features so the model cannot penalise
        # older or newer founding dates — market signals are timeless.
        _TIME_COLS = ["founded_year", "company_age", "pre_2010_flag",
                      "post_2015_flag", "years_to_2020", "age_sq", "age_log"]
        for _tc in _TIME_COLS:
            if _tc in X.columns:
                X[_tc] = 0.0
        raw = self.model.predict(X)[0]

        # Blend the model's raw score with the pipeline's profile-weighted score.
        # This ensures Entrepreneur and Investor views produce meaningfully different numbers.
        # Weight: 60% model (SHAP-faithful), 40% pipeline profile (user-type-weighted signals).
        pipeline_score = pr.get("score", raw)
        if user_type == "entrepreneur":
            blended_raw = 0.45 * raw + 0.55 * pipeline_score
        else:
            blended_raw = 0.80 * raw + 0.20 * pipeline_score

        cal = calibrate_score(blended_raw)

        # 5. SHAP explanation
        sv  = self.explainer.shap_values(X)
        arr = sv[0] if isinstance(sv, list) else sv[0]
        ctx = {
            "year":          year,
            "funding":       funding,
            "industry":      industry,
            "country":       country,
            "user_type":     user_type,
            "live_features": list(live_sigs.keys()) if isinstance(live_sigs, dict) else list(live_sigs),
        }
        factors = _build_shap_factors(self.feature_names, arr, ctx)
        recs    = _build_recommendations(factors, ctx)

        # 6. GDP context explanation
        gdp_val  = float(signals.get("gdp_growth", 0) or 0)
        gdp_ctx  = get_gdp_context(country, year, gdp_val)

        # 7. Benchmark
        bench = _build_benchmark(self.reference_df, industry, cal["score"])

        return {
            **cal,
            "base_value":        round(self.base_value, 2),
            "top_factors":       factors,
            "recommendations":   recs,
            "benchmark":         bench,
            "gdp_context":       gdp_ctx,
            "user_type":         user_type,
            # Data quality metadata
            "data_quality":      pr.get("data_quality", 0.5),
            "quality_label":     pr.get("quality_label", "Medium"),
            "quality_breakdown": pr.get("quality_breakdown", {}),
            "live_signals":      list(live_sigs.keys()) if isinstance(live_sigs, dict) else list(live_sigs),
        }


# ── SHAP explanations ──────────────────────────────────────────────────────────

def _explain(name: str, val: float, ctx: dict) -> str:
    funding   = ctx.get("funding", 0)
    ind       = ctx.get("industry", "")
    country   = ctx.get("country", "")
    year      = ctx.get("year", 2015)
    user_type = ctx.get("user_type", "investor")
    live      = ctx.get("live_features", [])
    d         = "boosting" if val >= 0 else "dragging down"
    src       = lambda f: "live" if f in live else "historical CSV"

    if name in ("log_funding", "funding_log"):
        if funding == 0:
            return ("No funding ($0) → log = 0. Unfunded ventures score lower — external capital signals "
                    "market validation that both investors and customers respond to.")
        return (f"Log₁₀(${funding:,.0f}+1) ≈ {np.log10(funding+1):.2f}. "
                f"Log-scaling lets $50K and $50M raises be compared fairly. {d}.")

    if name == "funding_total_usd":
        if user_type == "entrepreneur":
            return ("No external funding — bootstrapped start." if funding == 0
                    else f"${funding:,.0f} raised — validates external conviction in your market. {d}.")
        return ("No external funding — investor conviction is absent." if funding == 0
                else f"${funding:,.0f} validates investor conviction. {d}.")

    if name == "funding_x_rounds":
        return f"Funding depth × rounds — investor persistence signal (they kept backing it). {d}."

    if name == "funding_per_round":
        return f"Average capital per round — high values signal institutional conviction. {d}."

    if name == "funding_sqrt":
        return f"√funding — mid-scale transform capturing early-stage funding dynamics. {d}."

    if name.startswith("tier_"):
        t = name.replace("tier_", "").replace("_", " ")
        if user_type == "entrepreneur":
            return (f"Funding tier '{t}' — relevant to ecosystem access, not runway. {d}."
                    if val >= 0 else
                    f"Tier '{t}' is below average for your industry. Even a small raise signals commitment. {d}.")
        return (f"Funding tier '{t}' outperformed in training data. {d}."
                if val >= 0 else f"Tier '{t}' had below-average scores — next tier would help. {d}.")

    if name == "trend_slope":
        return (f"Google Trends slope for '{ind}' ({src('trend_slope')}). "
                f"{'Growing' if val >= 0 else 'Declining'} search interest — "
                f"trend momentum is the strongest single predictor in this model. {d}.")

    if name == "trend_slope_sq":
        return f"Squared trend slope — amplifies extreme momentum in either direction. {d}."

    if name == "trend_x_sentiment":
        return (f"Trend slope × news sentiment ({src('trend_slope')} × {src('news_sentiment')}). "
                f"A rising trend backed by positive press is far stronger than either alone. {d}.")

    if name == "trend_x_gdp":
        return f"Trend slope × GDP growth — growing market in growing economy compounds opportunity. {d}."

    if name == "trend_x_reddit":
        return (f"Trend slope × Reddit density — checks whether search interest is backed "
                f"by genuine community engagement. {d}.")

    if name == "market_validation":
        return (f"Composite: (trend×0.4) + (reddit/100×0.3) + (sentiment/10×0.3). "
                f"Summarises market validation across search, community, and media. {d}.")

    if name == "gdp_growth":
        if user_type == "entrepreneur":
            return (f"GDP growth for {country} in {year} ({src('gdp_growth')} — World Bank). "
                    f"A growing economy means more consumer spending, easier fundraising, and lower churn risk. {d}.")
        return (f"GDP growth for {country} in {year} ({src('gdp_growth')} — World Bank). "
                f"Strong growth = easier fundraising and higher exit multiples. {d}.")

    if name == "gdp_growth_sq":
        return f"Squared GDP growth — captures extreme economic conditions (booms and recessions equally). {d}."

    if name == "econ_health":
        return (f"GDP growth − inflation = real economic health. "
                f"High GDP with equally high inflation is deceptive — this captures the net. {d}.")

    if name == "real_gdp_growth":
        return f"Inflation-adjusted GDP growth. Real purchasing power gain in {country}. {d}."

    if name == "investment_climate":
        return f"GDP / (inflation+1). High ratio = economy growing faster than prices. {d}."

    if name == "ecosystem_strength":
        return f"startup_density × GDP/100. Dense ecosystem + growing economy = best launchpad. {d}."

    if name == "gdp_x_density":
        return f"GDP growth × startup density — checks if economic growth produces actual startup activity. {d}."

    if name == "reddit_density":
        return (f"Reddit daily mention density for '{ind}' ({src('reddit_density')} — PullPush). "
                f"Grassroots community interest beyond news cycles. {d}.")

    if name in ("reddit_density_log", "reddit_zscore"):
        return f"Log/z-score of Reddit density — normalises viral spikes for fair comparison. {d}."

    if name == "reddit_x_sentiment":
        return (f"Reddit density × news sentiment. High community buzz + positive press = "
                f"doubly validated market. {d}.")

    if name == "news_sentiment":
        return (f"GDELT news sentiment for '{ind}' ({src('news_sentiment')} — BigQuery). "
                f"Positive = favourable press; negative = regulatory/crisis signals. {d}.")

    if name == "news_volume":
        return (f"GDELT article count for '{ind}' ({src('news_volume')} — BigQuery). "
                f"Volume combined with sentiment gives a quality-weighted media signal. {d}.")

    if name in ("news_volume_log", "news_x_sentiment"):
        return f"Log/interaction of news volume — quality-weighted media signal. {d}."

    if name in ("trend_zscore", "gdp_zscore"):
        return f"Z-score normalisation — how many standard deviations above/below the dataset average. {d}."

    if name.startswith("country_"):
        c = name.replace("country_", "").replace("_", " ")
        if user_type == "entrepreneur":
            return (f"{c} is a top-10 startup ecosystem — strong talent, capital, and customer access. {d}."
                    if val >= 0 else f"Establishing presence in a top-10 ecosystem (US, UK, India) would boost this signal. {d}.")
        return (f"{c} is a top-10 startup ecosystem — rewarded for exit potential and market depth. {d}."
                if val >= 0 else f"{c} is outside the top-10 startup ecosystems in training data. {d}.")

    if name.startswith("cat_"):
        c = name.replace("cat_", "").replace("_", " ")
        return (f"'{c}' sector is above average in training data. {d}."
                if val >= 0 else f"'{c}' had below-average scores — competitive or difficult unit economics. {d}.")

    if name.startswith("status_"):
        return f"Company status encoding — {d} your score."

    if name == "startup_density":
        if user_type == "entrepreneur":
            return (f"Startup density in {country} — high density can mean saturation. "
                    f"{'Moderate density suggests an open market.' if val >= 0 else 'High saturation detected — a focused niche will help.'}")
        return f"Startup density in {country} — {'proven market ecosystem' if val >= 0 else 'thin ecosystem for investor exits'}. {d}."

    return (f"'{name.replace('_', ' ')}' is {d} your score by {abs(val):.2f} pts — "
            f"engineered feature derived from market, economic, and funding inputs.")


def _build_shap_factors(feature_names: list, arr, ctx: dict) -> list:
    # Exclude time-based features — they exist in the trained model's feature set
    # but are not meaningful drivers we want to surface to users. The model was
    # trained with founding year as a proxy for era; we surface market signals instead.
    _SUPPRESS = {"founded_year", "company_age", "pre_2010_flag", "post_2015_flag",
                 "years_to_2020", "age_sq", "age_log"}
    pairs = sorted(zip(feature_names, arr), key=lambda x: abs(x[1]), reverse=True)
    return [
        {"feature": n, "shap_value": round(float(v), 4),
         "direction": "positive" if v >= 0 else "negative",
         "abs_impact": round(abs(float(v)), 4),
         "explanation": _explain(n, float(v), ctx)}
        for n, v in pairs
        if n not in _SUPPRESS
    ][:10]


# ── Recommendations ────────────────────────────────────────────────────────────

def _build_recommendations(factors: list, ctx: dict) -> list:
    funding   = ctx.get("funding", 0)
    ind       = ctx.get("industry", "")
    country   = ctx.get("country", "")
    year      = ctx.get("year", 2015)
    user_type = ctx.get("user_type", "investor")
    live      = ctx.get("live_features", [])
    recs, seen = [], set()

    for f in factors:
        if f["direction"] != "negative" or f["abs_impact"] < 0.3:
            continue
        name = f["feature"]

        if name in ("log_funding", "funding_log", "funding_total_usd",
                    "tier_unfunded", "tier_seed") and "funding" not in seen:
            seen.add("funding")
            if funding == 0:
                if user_type == "entrepreneur":
                    recs.append({
                        "problem": "No external funding — market validation signal is absent",
                        "recommendation": "Apply to accelerators (Y Combinator, Antler, Techstars) or local angel networks. Even $25K signals commitment and unlocks ecosystem access.",
                        "potential_gain": "$0 → $50K seed could improve score by 8–15 pts",
                    })
                else:
                    recs.append({
                        "problem": "No external funding — investor conviction signal is absent",
                        "recommendation": "A seed round validates the market. Even small institutional backing shifts the model's view significantly.",
                        "potential_gain": "$0 → $50K seed could improve score by 8–15 pts",
                    })
            else:
                recs.append({
                    "problem": f"Funding (${funding:,.0f}) is below the high-opportunity threshold",
                    "recommendation": "Raise a follow-on round — each tier jump (seed → Series A → B+) brings a material score boost.",
                    "potential_gain": "5–10 pts at the next tier",
                })

        elif name in ("trend_slope", "trend_zscore", "trend_x_sentiment",
                      "market_validation") and "trend" not in seen:
            seen.add("trend")
            data_note = "Live Google Trends data shows" if "trend_slope" in live else "Historical data shows"
            recs.append({
                "problem": f"{data_note} declining or flat search interest for '{ind}'",
                "recommendation": ("Reframe your product around a rising adjacent keyword in your sector. "
                                   "Market momentum is the strongest predictor in this model."
                                   if user_type == "entrepreneur" else
                                   "Consider timing — invest when trend slope is positive and accelerating for maximum momentum."),
                "potential_gain": "5–12 pts (trend carries the most weight)",
            })

        elif name in ("gdp_growth", "econ_health", "investment_climate",
                      "real_gdp_growth") and "econ" not in seen:
            seen.add("econ")
            data_note = "Live World Bank data shows" if "gdp_growth" in live else "Historical data shows"
            recs.append({
                "problem": f"{data_note} challenging economic conditions in {country} for {year}",
                "recommendation": ("Launch in a higher-growth market first (US, India, SE Asia) for initial traction and fundraising, then expand locally."
                                   if user_type == "entrepreneur" else
                                   "Target markets with GDP growth > 3% — macro conditions compound returns over a 5–7 year investment horizon."),
                "potential_gain": "3–8 pts",
            })

        elif name in ("reddit_density", "reddit_x_sentiment",
                      "reddit_density_log") and "reddit" not in seen:
            seen.add("reddit")
            data_note = "Live Reddit data shows" if "reddit_density" in live else "Historical data shows"
            recs.append({
                "problem": f"{data_note} low community density for '{ind}'",
                "recommendation": ("Engage relevant subreddits, share your product, participate genuinely. Authentic community interest builds before revenue does."
                                   if user_type == "entrepreneur" else
                                   "Low community density may indicate an early or niche market — validate founder traction before committing capital."),
                "potential_gain": "2–6 pts",
            })

        elif name in ("news_sentiment", "news_x_sentiment") and "news" not in seen:
            seen.add("news")
            data_note = "Live GDELT data shows" if "news_sentiment" in live else "Historical data shows"
            recs.append({
                "problem": f"{data_note} neutral or negative press sentiment for '{ind}'",
                "recommendation": ("Publish thought leadership, announce partnerships, or pitch journalists. Press sentiment lifts both investor and customer perception."
                                   if user_type == "entrepreneur" else
                                   "Negative sector sentiment is often a contrarian opportunity — investigate whether it reflects cyclical headwinds or structural decline."),
                "potential_gain": "2–5 pts",
            })

        elif (name.startswith("country_") or
              name in ("startup_density", "ecosystem_strength", "gdp_x_density")) and "eco" not in seen:
            seen.add("eco")
            recs.append({
                "problem": f"Startup ecosystem in {country} is below average for this scoring model",
                "recommendation": ("Establish presence in a top-10 ecosystem (US, UK, Germany, India, Canada) to access investors, talent, and customers."
                                   if user_type == "entrepreneur" else
                                   "Consider ecosystem risk when sizing position — thinner ecosystems have fewer exit pathways and competitive advantages."),
                "potential_gain": "3–8 pts",
            })

        if len(recs) >= 4:
            break
    return recs


# ── Benchmark ──────────────────────────────────────────────────────────────────

def _build_benchmark(ref_df: pd.DataFrame, industry: str, score: float) -> dict:
    cat = ref_df[ref_df["category"].str.lower() == industry.lower()]
    if cat.empty:
        cat = ref_df
    scores = cat["opportunity_score"].dropna()
    avg    = round(float(scores.mean()), 2) if len(scores) else 50.0
    pct    = round(float((scores < score).mean() * 100), 1)
    top    = (cat[cat["opportunity_score"] >= 70]
              .sort_values("opportunity_score", ascending=False).head(5))
    return {
        "your_percentile":    pct,
        "category_avg_score": avg,
        "successful_ventures": [
            {"name":    str(r.get("name", "—")),
             "industry": str(r.get("industry", industry)),
             "country": str(r.get("country", "—")),
             "year":    int(r.get("founded_year", 0)),
             "funding": float(r.get("funding_total_usd", 0)),
             "score":   round(float(r.get("opportunity_score", 0)), 2),
             "status":  str(r.get("status", "—"))}
            for _, r in top.iterrows()
        ],
    }
