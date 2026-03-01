import requests
import pandas as pd

# ---------------------------------
# 1. Define Indicators
# ---------------------------------

INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "FP.CPI.TOTL.ZG": "inflation",
    "SL.UEM.TOTL.ZS": "unemployment_rate",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_percent_gdp",
    "SP.POP.TOTL": "population"
}

START_YEAR = 2000
END_YEAR = 2024


# ---------------------------------
# 2. Function to Fetch One Indicator
# ---------------------------------

def fetch_indicator(indicator_code, feature_name):
    print(f"Fetching {feature_name}...")

    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
    params = {
        "format": "json",
        "per_page": 20000
    }

    response = requests.get(url, params=params)
    data = response.json()

    if len(data) < 2:
        print(f"No data returned for {feature_name}")
        return pd.DataFrame()

    records = data[1]

    df = pd.DataFrame(records)

    # Extract relevant columns
    df = df[["country", "date", "value"]]

    df["country"] = df["country"].apply(lambda x: x["value"])
    df.rename(columns={"date": "year", "value": feature_name}, inplace=True)

    df["year"] = df["year"].astype(int)

    # Filter years
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]

    return df


# ---------------------------------
# 3. Fetch All Indicators
# ---------------------------------

macro_df = None

for code, name in INDICATORS.items():
    indicator_df = fetch_indicator(code, name)

    if macro_df is None:
        macro_df = indicator_df
    else:
        macro_df = macro_df.merge(
            indicator_df,
            on=["country", "year"],
            how="outer"
        )

# ---------------------------------
# 4. Remove Aggregated Regions
# ---------------------------------

aggregates = [
    "World",
    "High income",
    "Low income",
    "Middle income"
]

macro_df = macro_df[~macro_df["country"].isin(aggregates)]

# ---------------------------------
# 5. Sort & Forward Fill
# ---------------------------------

macro_df = macro_df.sort_values(["country", "year"])

macro_df = (
    macro_df.groupby("country")
    .apply(lambda x: x.ffill())
    .reset_index(drop=True)
)

# ---------------------------------
# 6. Save
# ---------------------------------

macro_df.to_csv("macro_data.csv", index=False)

print("Macro dataset created successfully.")
print("Shape:", macro_df.shape)
print(macro_df.head())
