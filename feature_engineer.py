"""
feature_engineer.py
-------------------
Translates raw user inputs into the exact feature format required
by the LightGBM model. Mirrors feature_engineer.ipynb exactly.

Changes from previous version:
  - REMOVED time features: company_age, pre_2010_flag, post_2015_flag
    (model no longer penalises founding year — market signals are timeless)
  - ADDED funding_tier one-hot: tier_unfunded, tier_seed, tier_series_a, tier_series_b_plus
  - ADDED country_grouped top-10 encoding (same 10 countries as training data)
  - log_funding alias kept alongside funding_log for model compatibility
"""

import numpy as np
import pandas as pd


# Top-10 countries from training data (all others → 'Other')
_TOP_10_COUNTRIES = [
    "United States", "United Kingdom", "Canada", "Germany",
    "India", "Israel", "France", "China", "Spain", "Singapore",
]


class VentureFeatureEngineer:
    """
    Transforms a raw input row into the feature schema the LightGBM model
    was trained on.

    Steps:
      1. Log / polynomial transforms on numeric fields
      2. Interaction terms
      3. Ratio / derived economic metrics
      4. Z-score normalisation using training-set statistics
      5. One-hot encoding: category, status, country_grouped (top-10+Other),
         funding_tier
      6. Strict column alignment — missing dummies filled with 0
    """

    def __init__(self, model_features: list, zscore_params: dict):
        self.model_features = model_features
        self.zscore_params  = zscore_params

    def transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy().reset_index(drop=True)

        # ── Log transforms ──────────────────────────────────────────────────
        df["funding_log"]        = np.log1p(df["funding_total_usd"])
        df["log_funding"]        = np.log1p(df["funding_total_usd"])   # alias
        df["news_volume_log"]    = np.log1p(df.get("news_volume",    pd.Series([0] * len(df))))
        df["reddit_density_log"] = np.log1p(df.get("reddit_density", pd.Series([0] * len(df))))

        # ── Polynomial features ─────────────────────────────────────────────
        df["funding_sqrt"]   = np.sqrt(df["funding_total_usd"])
        df["trend_slope_sq"] = df.get("trend_slope", pd.Series([0] * len(df))) ** 2
        df["gdp_growth_sq"]  = df.get("gdp_growth",  pd.Series([0] * len(df))) ** 2

        # ── Interaction features ────────────────────────────────────────────
        df["trend_x_gdp"]        = df.get("trend_slope",    pd.Series([0]*len(df))) * df.get("gdp_growth",     pd.Series([0]*len(df)))
        df["trend_x_sentiment"]  = df.get("trend_slope",    pd.Series([0]*len(df))) * df.get("news_sentiment",  pd.Series([0]*len(df)))
        df["reddit_x_sentiment"] = df.get("reddit_density", pd.Series([0]*len(df))) * df.get("news_sentiment",  pd.Series([0]*len(df)))
        df["funding_x_rounds"]   = df["funding_log"] * df.get("funding_rounds", pd.Series([0]*len(df)))
        df["trend_x_reddit"]     = df.get("trend_slope",    pd.Series([0]*len(df))) * df["reddit_density_log"]
        df["gdp_x_density"]      = df.get("gdp_growth",     pd.Series([0]*len(df))) * df.get("startup_density", pd.Series([0]*len(df)))
        df["news_x_sentiment"]   = df["news_volume_log"] * df.get("news_sentiment", pd.Series([0]*len(df)))

        # ── Ratio / derived features ────────────────────────────────────────
        df["econ_health"]        = df.get("gdp_growth", pd.Series([0]*len(df))) - df.get("inflation", pd.Series([0]*len(df)))
        df["funding_per_round"]  = df["funding_total_usd"] / (df.get("funding_rounds", pd.Series([0]*len(df))) + 1)
        df["real_gdp_growth"]    = df.get("gdp_growth", pd.Series([0]*len(df))) / (1 + df.get("inflation", pd.Series([0]*len(df))) / 100)
        df["investment_climate"] = df.get("gdp_growth", pd.Series([0]*len(df))) / (df.get("inflation", pd.Series([0]*len(df))) + 1)
        df["ecosystem_strength"] = df.get("startup_density", pd.Series([0]*len(df))) * df.get("gdp_growth", pd.Series([0]*len(df))) / 100
        df["market_validation"]  = (
            df.get("trend_slope",    pd.Series([0]*len(df))) * 0.4 +
            (df.get("reddit_density", pd.Series([0]*len(df))) / 100) * 0.3 +
            (df.get("news_sentiment", pd.Series([0]*len(df))) / 10)  * 0.3
        )

        # NOTE: Time features (company_age, pre_2010_flag, post_2015_flag)
        # have been intentionally REMOVED. The model no longer penalises
        # founding year — market condition signals are timeless.

        # ── Z-scores (training-set statistics) ──────────────────────────────
        zp = self.zscore_params
        df["trend_zscore"]  = (df.get("trend_slope",    pd.Series([0]*len(df))) - zp["trend_slope_mean"])    / zp["trend_slope_std"]
        df["gdp_zscore"]    = (df.get("gdp_growth",     pd.Series([0]*len(df))) - zp["gdp_growth_mean"])     / zp["gdp_growth_std"]
        df["reddit_zscore"] = (df.get("reddit_density", pd.Series([0]*len(df))) - zp["reddit_density_mean"]) / zp["reddit_density_std"]

        # ── Country grouping (top-10 + Other) ──────────────────────────────
        country_col = df.get("country", pd.Series(["Other"] * len(df)))
        df["country_grouped"] = country_col.apply(
            lambda x: x if x in _TOP_10_COUNTRIES else "Other"
        )

        # ── Funding tiers ───────────────────────────────────────────────────
        bins   = [-1, 0, 2_000_000, 10_000_000, float("inf")]
        labels = ["unfunded", "seed", "series_a", "series_b_plus"]
        df["funding_tier"] = pd.cut(df["funding_total_usd"], bins=bins, labels=labels)

        # ── One-hot encoding ─────────────────────────────────────────────────
        df_encoded = pd.get_dummies(df, columns=["category", "status", "country_grouped", "funding_tier"])

        # Rename to match training prefixes
        rename_map = {}
        for col in df_encoded.columns:
            if col.startswith("category_"):
                rename_map[col] = "cat_" + col[len("category_"):]
            elif col.startswith("country_grouped_"):
                rename_map[col] = "country_" + col[len("country_grouped_"):]
            elif col.startswith("funding_tier_"):
                rename_map[col] = "tier_" + col[len("funding_tier_"):]
        df_encoded = df_encoded.rename(columns=rename_map)

        # ── Strict column alignment ──────────────────────────────────────────
        final_df = pd.DataFrame(columns=self.model_features)
        for col in self.model_features:
            if col in df_encoded.columns:
                final_df[col] = df_encoded[col].values
            else:
                final_df[col] = 0

        return final_df[self.model_features].fillna(0).astype(float)
