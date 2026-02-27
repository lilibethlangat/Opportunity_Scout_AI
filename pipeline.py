"""
pipeline.py — Live data aggregation engine for Opportunity Scout AI.

"""
from __future__ import annotations
import os, time, threading, logging
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("pipeline")

REQUIRED_FEATURES = [
    "trend_slope", "gdp_growth", "reddit_density", "news_sentiment",
    "news_volume", "startup_density", "funding_total_usd", "inflation",
]
LIVE_FEATURES = [
    "trend_slope", "gdp_growth", "inflation",
    "reddit_density", "news_volume", "news_sentiment",
]

# Weight Profiles (from pipeline_opportunity_score_weights.ipynb) 

# ENTREPRENEUR — "Is this market fertile enough to start in?"
#   trend_slope is the strongest predictor; funding irrelevant at founding.
#   startup_density is INVERTED (high = saturated = bad for new entrant).

# INVESTOR — "Has this company earned its score and is the market behind it?"
#   log_funding first (other smart money already validated the space).
#   startup_density is NOT inverted (high = proven market with exit ecosystem).

WEIGHTS_ENTREPRENEUR = {
    "trend_slope":     0.35,
    "news_sentiment":  0.25,
    "gdp_growth":      0.15,
    "reddit_density":  0.12,
    "news_volume":     0.08,
    "startup_density": 0.03,   # inverted — see compute_opportunity_score()
    "log_funding":     0.00,   # intentionally zero for entrepreneur profile
    "inflation":       0.02,
}

WEIGHTS_INVESTOR = {
    "trend_slope":     0.20,
    "news_sentiment":  0.18,
    "log_funding":     0.22,
    "news_volume":     0.10,
    "reddit_density":  0.10,
    "startup_density": 0.08,   # NOT inverted for investor
    "gdp_growth":      0.10,
    "inflation":       0.02,
}

# Feature bounds (derived from ventures.csv; used for 0-100 normalisation)
# These are conservative dataset-wide min/max values.
_BOUNDS = {
    "trend_slope":     (-0.5,    2.0),
    "gdp_growth":      (-20.0,  20.0),
    "reddit_density":  (0.0,    50.0),
    "news_sentiment":  (0.0,     6.6),
    "news_volume":     (0.0, 500000.0),
    "startup_density": (0.0,    30.0),
    "log_funding":     (0.0,    22.0),
    "inflation":       (-5.0,   50.0),
}


def _normalise(value: float, feature: str) -> float:
    """Scale a raw signal value to 0-100 using stored min/max bounds."""
    lo, hi = _BOUNDS.get(feature, (0.0, 1.0))
    if hi == lo:
        return 50.0
    return float(np.clip(((value - lo) / (hi - lo)) * 100, 0, 100))


def compute_opportunity_score(signals: dict, user_type: str = "investor") -> float:
    """
    Compute a 0-100 pipeline opportunity score from the 8 raw signals.

    Parameters
    ----------
    signals   : validated signal dict (output of validate_signals)
    user_type : "entrepreneur" or "investor"

    Returns
    -------
    float : opportunity score 0-100
    """
    weights = WEIGHTS_INVESTOR if user_type == "investor" else WEIGHTS_ENTREPRENEUR

    log_funding = float(np.log1p(signals.get("funding_total_usd") or 0))
    enriched = dict(signals)
    enriched["log_funding"] = log_funding

    score = 0.0
    for feature, weight in weights.items():
        if weight == 0.0:
            continue
        raw = enriched.get(feature)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            # Impute with midpoint (50 normalised)
            norm = 50.0
        else:
            norm = _normalise(float(raw), feature)

        # Entrepreneur: invert startup_density (high density = saturated = bad)
        if user_type == "entrepreneur" and feature == "startup_density":
            norm = 100.0 - norm

        # Both profiles: invert inflation (high inflation = bad)
        if feature == "inflation":
            norm = 100.0 - norm

        score += weight * norm

    return round(float(np.clip(score, 0, 100)), 2)


#  In-memory cache 
_CACHE: dict = {}


def _cache_key(industry, country, year, user_type):
    return f"{industry}|{country}|{year}|{user_type}"


# Rate limiter 
class RateLimiter:
    def __init__(self, max_calls, period_seconds):
        self.max_calls = max_calls
        self.period    = period_seconds
        self._times    = deque()
        self._lock     = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            while self._times and now - self._times[0] > self.period:
                self._times.popleft()
            if len(self._times) >= self.max_calls:
                sleep = self.period - (now - self._times[0])
                if sleep > 0:
                    time.sleep(sleep)
                now = time.monotonic()
                while self._times and now - self._times[0] > self.period:
                    self._times.popleft()
            self._times.append(time.monotonic())


_RL = {
    "GoogleTrends": RateLimiter(5,  60),
    "WorldBank":    RateLimiter(10, 60),
    "Reddit":       RateLimiter(10, 60),
    "GDELT":        RateLimiter(5,  60),
}


# Base collector 
class BaseCollector(ABC):
    @abstractmethod
    def collect(self, industry: str, country: str, year: int) -> dict:
        pass


# CSV fallback 
class VenturesCollector(BaseCollector):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        logger.info(f"VenturesCollector: {len(self.df)} rows from {filepath}")

    def collect(self, industry, country, year):
        cat_col = "category" if "category" in self.df.columns else "industry"
        mask = self.df[cat_col].str.lower() == industry.lower()
        f    = self.df[mask & (self.df["founded_year"] == year)]
        if f.empty:
            f = self.df[mask & self.df["founded_year"].between(year - 3, year + 3)]
        if f.empty:
            f = self.df[mask]
        if f.empty:
            return {}
        cols = [c for c in ["trend_slope", "gdp_growth", "reddit_density", "news_sentiment",
                             "news_volume", "startup_density", "funding_total_usd", "inflation"]
                if c in f.columns]
        row = f[cols].median()
        return {k: float(v) for k, v in row.items() if not np.isnan(v)}


#  Google Trends 
class GoogleTrendsCollector(BaseCollector):
    CC = {
        "united states": "US", "kenya": "KE", "united kingdom": "GB", "india": "IN",
        "germany": "DE", "france": "FR", "canada": "CA", "australia": "AU",
        "nigeria": "NG", "south africa": "ZA", "china": "CN", "brazil": "BR",
        "chile": "CL", "estonia": "EE", "singapore": "SG", "israel": "IL",
        "spain": "ES", "netherlands": "NL", "sweden": "SE", "japan": "JP",
    }
    KW = {
        "E-Commerce & Retail": "ecommerce", "Technology & Software": "saas software",
        "Healthcare & Life Sciences": "digital health", "Finance & Fintech": "fintech",
        "Media & Entertainment": "streaming media", "Education & Training": "edtech",
        "Real Estate & Construction": "proptech", "Travel & Hospitality": "travel technology",
        "Food & Beverage": "food delivery", "Transportation & Logistics": "logistics technology",
        "Marketing & Advertising": "digital marketing", "Internet & Web Services": "web services",
        "Data & Analytics": "data analytics", "Gaming": "gaming esports",
        "Social & Community": "social media platform", "Consumer & Lifestyle": "consumer technology",
        "Manufacturing & Industrial": "industrial automation", "Sports & Recreation": "fitness technology",
        "Communications & Telecom": "telecommunications 5g", "Energy & Clean Tech": "renewable energy",
        "Enterprise & Business Services": "enterprise software", "Security & Privacy": "cybersecurity",
        "AI & Machine Learning": "artificial intelligence", "Customer Service & CRM": "crm software",
        "HR & Recruiting": "hr technology", "Government & Politics": "govtech",
    }

    def collect(self, industry, country, year):
        try:
            from pytrends.request import TrendReq
            _RL["GoogleTrends"].wait()
            kw  = self.KW.get(industry, industry)
            geo = self.CC.get(country.lower(), "")
            pt  = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            pt.build_payload([kw], cat=0, timeframe=f"{year}-01-01 {year}-12-31", geo=geo)
            data = pt.interest_over_time()
            if data.empty or kw not in data.columns:
                return {}
            series = data[kw].values.astype(float)
            if len(series) < 2:
                return {}
            slope = float(np.polyfit(np.arange(len(series)), series, 1)[0]) / 100.0
            logger.info(f"Trends: '{industry}' {year} slope={slope:.4f}")
            return {"trend_slope": slope}
        except Exception as e:
            logger.error(f"GoogleTrends: {e}")
            return {}


#  World Bank 
class WorldBankCollector(BaseCollector):
    ISO = {
        "united states": "US", "kenya": "KE", "united kingdom": "GB", "india": "IN",
        "germany": "DE", "france": "FR", "canada": "CA", "australia": "AU",
        "nigeria": "NG", "south africa": "ZA", "china": "CN", "brazil": "BR",
        "chile": "CL", "estonia": "EE", "singapore": "SG", "israel": "IL",
        "spain": "ES", "netherlands": "NL", "sweden": "SE", "japan": "JP",
        "indonesia": "ID", "mexico": "MX", "argentina": "AR", "ghana": "GH",
        "rwanda": "RW", "ethiopia": "ET", "united arab emirates": "AE", "pakistan": "PK",
    }

    def _fetch(self, iso, indicator, year):
        _RL["WorldBank"].wait()
        for y in [year, year - 1, year - 2, year - 3]:
            try:
                r = requests.get(
                    f"https://api.worldbank.org/v2/country/{iso}/indicator/{indicator}",
                    params={"date": str(y), "format": "json", "per_page": 1}, timeout=10)
                r.raise_for_status()
                p = r.json()
                if len(p) >= 2 and p[1]:
                    v = p[1][0].get("value")
                    if v is not None:
                        return float(v)
            except Exception:
                pass
        return None

    def collect(self, industry, country, year):
        iso = self.ISO.get(country.lower())
        if not iso:
            return {}
        result = {}
        gdp = self._fetch(iso, "NY.GDP.MKTP.KD.ZG", year)
        if gdp is not None:
            result["gdp_growth"] = gdp
            logger.info(f"WB: {country} gdp={gdp:.2f}")
        inf = self._fetch(iso, "FP.CPI.TOTL.ZG", year)
        if inf is not None:
            result["inflation"] = inf
            logger.info(f"WB: {country} inflation={inf:.2f}")
        return result


#  Reddit / PullPush 
class RedditCollector(BaseCollector):
    URL = "https://api.pullpush.io/reddit/search/submission/"
    HDR = {"User-Agent": "OpportunityScoutAI/1.0"}
    KW  = {
        "E-Commerce & Retail":          ["ecommerce", "online shopping", "retail technology"],
        "Technology & Software":        ["software", "saas", "cloud computing"],
        "Healthcare & Life Sciences":   ["digital health", "telemedicine", "healthcare technology"],
        "Finance & Fintech":            ["fintech", "digital banking", "payment technology"],
        "Media & Entertainment":        ["streaming services", "digital media", "content creation"],
        "Education & Training":         ["edtech", "online learning", "educational technology"],
        "Real Estate & Construction":   ["proptech", "real estate technology"],
        "Travel & Hospitality":         ["travel technology", "online booking"],
        "Food & Beverage":              ["food delivery", "food technology"],
        "Transportation & Logistics":   ["logistics technology", "supply chain"],
        "Marketing & Advertising":      ["digital marketing", "martech"],
        "Internet & Web Services":      ["web services", "internet technology"],
        "Data & Analytics":             ["data analytics", "big data", "business intelligence"],
        "Gaming":                       ["video games", "gaming", "esports"],
        "Social & Community":           ["social media", "social networking"],
        "Consumer & Lifestyle":         ["consumer technology", "lifestyle products"],
        "Manufacturing & Industrial":   ["industrial automation", "manufacturing technology"],
        "Sports & Recreation":          ["fitness technology", "sports technology"],
        "Communications & Telecom":     ["telecommunications", "5g technology"],
        "Energy & Clean Tech":          ["clean energy", "renewable energy", "cleantech"],
        "Enterprise & Business Services": ["enterprise software", "b2b software"],
        "Security & Privacy":           ["cybersecurity", "information security"],
        "AI & Machine Learning":        ["artificial intelligence", "machine learning"],
        "Customer Service & CRM":       ["crm software", "customer service technology"],
        "HR & Recruiting":              ["hr technology", "recruitment technology"],
        "Government & Politics":        ["govtech", "government technology"],
    }

    def _density(self, kw, year):
        s = int(time.mktime(time.strptime(f"{year}-01-01", "%Y-%m-%d")))
        e = int(time.mktime(time.strptime(f"{year}-12-31", "%Y-%m-%d")))
        _RL["Reddit"].wait()
        try:
            r = requests.get(self.URL,
                             params={"q": kw, "after": s, "before": e, "size": 100, "sort": "asc"},
                             headers=self.HDR, timeout=15)
            if r.status_code != 200:
                return 0.0
            data = r.json().get("data", [])
            if len(data) < 2:
                return 0.0
            days = (int(data[-1]["created_utc"]) - int(data[0]["created_utc"])) / 86400
            return round(len(data) / days, 4) if days > 0 else 0.0
        except Exception as e:
            logger.error(f"Reddit '{kw}': {e}")
            return 0.0

    def collect(self, industry, country, year):
        try:
            kws = self.KW.get(industry, [industry.lower()])
            avg = round(sum(self._density(k, year) for k in kws) / len(kws), 4)
            logger.info(f"Reddit: '{industry}' {year} density={avg}")
            return {"reddit_density": avg}
        except Exception as e:
            logger.error(f"RedditCollector: {e}")
            return {}


#  GDELT / BigQuery 
class GDELTCollector(BaseCollector):
    CREDS   = os.getenv("GDELT_CREDENTIALS", "strong_gdelt.json")
    PROJECT = os.getenv("GDELT_PROJECT_ID",  "strong-augury-487515-u7")
    MAP = {
        "Media & Entertainment": "GENERAL_ENTERTAINMENT", "Travel & Hospitality": "TOURISM",
        "Technology & Software": "TECHNOLOGY", "E-Commerce & Retail": "ECONOMY_ECOMMERCE",
        "Real Estate & Construction": "ECONOMY_REALESTATE", "Education & Training": "EDUCATION",
        "Internet & Web Services": "INTERNET", "Food & Beverage": "FOOD_SECURITY",
        "Healthcare & Life Sciences": "HEALTH_SERVICES", "Data & Analytics": "COMPUTING_DATA_MINING",
        "Consumer & Lifestyle": "SOCIETY_LIFESTYLE", "Transportation & Logistics": "TRANSPORTATION",
        "Sports & Recreation": "SPORTS", "Manufacturing & Industrial": "MANUFACTURING",
        "Marketing & Advertising": "ADVERTISING", "Gaming": "GAMES",
        "Finance & Fintech": "ECONOMY_FINTECH", "Communications & Telecom": "TELECOMMUNICATIONS",
        "Social & Community": "SOCIETY", "Energy & Clean Tech": "ENERGY_RENEWABLE",
        "Enterprise & Business Services": "ECONOMY_BUSINESS_SERVICES",
        "AI & Machine Learning": "COMPUTING_ARTIFICIAL_INTELLIGENCE",
        "Customer Service & CRM": "CUSTOMER_SERVICE", "HR & Recruiting": "LABOR_RECRUITMENT",
        "Security & Privacy": "SECURITY_SERVICES", "Government & Politics": "GOVERNMENT",
    }

    def __init__(self):
        self._cache  = {}
        self._client = None

    def _client_ok(self):
        if self._client:
            return self._client
        try:
            from google.cloud import bigquery
            creds = (self.CREDS if os.path.isabs(self.CREDS)
                     else os.path.join(os.path.dirname(__file__), self.CREDS))
            if not os.path.exists(creds):
                logger.warning(f"GDELT creds not found: {creds}")
                return None
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
            self._client = bigquery.Client(project=self.PROJECT, location="US")
            return self._client
        except Exception as e:
            logger.error(f"GDELT client: {e}")
            return None

    def _query(self, theme, year):
        client = self._client_ok()
        if not client:
            return {}
        _RL["GDELT"].wait()
        sql = (
            f"SELECT COUNT(*) AS vol, "
            f"ROUND(AVG(SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64)),4) AS sent "
            f"FROM `gdelt-bq.gdeltv2.gkg_partitioned` "
            f"WHERE _PARTITIONDATE BETWEEN '{year}-01-01' AND '{year}-12-31' "
            f"AND themes LIKE '%{theme}%'"
            if year >= 2015 else
            f"SELECT COUNT(*) AS vol, ROUND(AVG(AvgTone),4) AS sent "
            f"FROM `gdelt-bq.full.events` WHERE Year={year} "
            f"AND (Actor1Type1Code='BUS' OR Actor2Type1Code='BUS')"
        )
        try:
            rows = list(client.query(sql).result())
            if not rows or rows[0].vol == 0:
                return {}
            tone = rows[0].sent
            sent = round(float(np.clip((tone + 10) * (6.6 / 20.0), 0, 6.6)), 4) if tone is not None else None
            r    = {"news_volume": float(rows[0].vol)}
            if sent is not None:
                r["news_sentiment"] = sent
            return r
        except Exception as e:
            logger.error(f"GDELT query: {e}")
            return {}

    def collect(self, industry, country, year):
        theme = self.MAP.get(industry)
        if not theme:
            return {}
        k = f"{theme}|{year}"
        if k in self._cache:
            return self._cache[k]
        result = self._query(theme, year)
        if result:
            logger.info(f"GDELT: '{industry}' {year} → {result}")
        self._cache[k] = result
        return result


#  Signal validation 
def validate_signals(signals: dict):
    validated, issues = {}, {}
    for f in REQUIRED_FEATURES:
        if f not in signals:
            validated[f] = np.nan
            issues[f]    = "missing"
            continue
        try:
            v = float(signals[f])
            validated[f] = v
            if np.isnan(v):
                issues[f] = "nan"
        except Exception:
            validated[f] = np.nan
            issues[f]    = "type_error"
    # pass-through any extra signals
    for k, v in signals.items():
        if k not in validated:
            try:
                validated[k] = float(v)
            except Exception:
                pass
    return validated, issues


def score_data_quality(validated: dict, live_signals: dict) -> dict:
    score = 0.0
    breakdown = {}
    for f in LIVE_FEATURES:
        v = validated.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            breakdown[f] = "missing"
        elif f in live_signals and live_signals[f] is not None:
            breakdown[f] = "live"
            score += 1.0
        else:
            breakdown[f] = "fallback"
            score += 0.5
    q     = round(score / len(LIVE_FEATURES), 4)
    label = "High" if q >= 0.75 else "Medium" if q >= 0.50 else "Low"
    return {"data_quality": q, "quality_label": label, "quality_breakdown": breakdown}


#  Aggregation engine 
class AggregationEngine:
    def __init__(self, ventures_path: str):
        self.fallback   = VenturesCollector(ventures_path)
        self.collectors = [
            GoogleTrendsCollector(),
            WorldBankCollector(),
            RedditCollector(),
            GDELTCollector(),
        ]
        logger.info("AggregationEngine ready")

    def run_live(self, industry: str, country: str, year: int,
                 funding: float = 0.0, user_type: str = "investor") -> dict:
        """
        Full live pipeline.

        Parameters
        ----------
        industry  : industry name string
        country   : country name string
        year      : target founding year (int) — no clipping applied
        funding   : total funding in USD (used for investor score profile)
        user_type : "entrepreneur" or "investor"
                    Controls which weight profile is used for the pipeline score.

        Returns dict with keys:
            signals, live_signals, score, score_profile, user_type,
            data_quality, quality_label, quality_breakdown, issues
        """
        if user_type not in ("entrepreneur", "investor"):
            logger.warning(f"Invalid user_type '{user_type}', defaulting to 'investor'")
            user_type = "investor"

        key = _cache_key(industry, country, year, user_type)
        if key in _CACHE:
            return _CACHE[key]

        # Step 1: Live collectors
        live_sigs = {}
        for collector in self.collectors:
            name = collector.__class__.__name__
            try:
                data = collector.collect(industry, country, year)
                for k, v in data.items():
                    if k not in live_sigs:
                        live_sigs[k] = v
                logger.info(f"{name} → {list(data.keys())}")
            except Exception as e:
                logger.error(f"{name} crashed: {e}")

        # Step 2: Merge live + funding
        merged = dict(live_sigs)
        merged["funding_total_usd"] = funding

        # Step 3: CSV fallback for any missing signals
        hist          = self.fallback.collect(industry, country, year)
        fallback_used = []
        for f in REQUIRED_FEATURES:
            if f not in merged or merged.get(f) is None:
                v = hist.get(f)
                if v is not None:
                    merged[f]    = v
                    fallback_used.append(f)
        if fallback_used:
            logger.info(f"CSV fallback: {fallback_used}")

        # Step 4: Validate
        validated, issues = validate_signals(merged)

        # Step 5: Data quality
        quality = score_data_quality(validated, live_sigs)

        # Step 6: Pipeline opportunity score (weight-profile based)
        pipeline_score = compute_opportunity_score(validated, user_type)
        score_profile  = (WEIGHTS_ENTREPRENEUR if user_type == "entrepreneur"
                          else WEIGHTS_INVESTOR)

        result = {
            "status":      "success",
            "mode":        "live",
            "user_type":   user_type,
            "signals":     validated,
            "live_signals": live_sigs,
            "score":        pipeline_score,
            "score_profile": score_profile,
            **quality,
            "issues":      issues,
        }

        _CACHE[key] = result
        return result
