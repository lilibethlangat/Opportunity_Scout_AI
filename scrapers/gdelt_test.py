import pandas as pd
from google.cloud import bigquery
import os
import time

# 1. SETUP - Use your NEW JSON key and NEW Project ID
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "strong_gdelt.json"
NEW_PROJECT_ID = 'strong-augury-487515-u7' # Find this in your Google Cloud Dashboard

client = bigquery.Client(project=NEW_PROJECT_ID, location="US")

# The Full Industry Map
INDUSTRY_MAP = {
    'Media & Entertainment': 'GENERAL_ENTERTAINMENT', 'Travel & Hospitality': 'TOURISM',
    'Technology & Software': 'TECHNOLOGY', 'E-Commerce & Retail': 'ECONOMY_ECOMMERCE',
    'Real Estate & Construction': 'ECONOMY_REALESTATE', 'Education & Training': 'EDUCATION',
    'Internet & Web Services': 'INTERNET', 'Food & Beverage': 'FOOD_SECURITY',
    'Healthcare & Life Sciences': 'HEALTH_SERVICES', 'Data & Analytics': 'COMPUTING_DATA_MINING',
    'Consumer & Lifestyle': 'SOCIETY_LIFESTYLE', 'Transportation & Logistics': 'TRANSPORTATION',
    'Sports & Recreation': 'SPORTS', 'Manufacturing & Industrial': 'MANUFACTURING',
    'Marketing & Advertising': 'ADVERTISING', 'Gaming': 'GAMES',
    'Finance & Fintech': 'ECONOMY_FINTECH', 'Communications & Telecom': 'TELECOMMUNICATIONS',
    'Social & Community': 'SOCIETY', 'Energy & Clean Tech': 'ENERGY_RENEWABLE',
    'Enterprise & Business Services': 'ECONOMY_BUSINESS_SERVICES', 'AI & Machine Learning': 'COMPUTING_ARTIFICIAL_INTELLIGENCE',
    'Customer Service & CRM': 'CUSTOMER_SERVICE', 'HR & Recruiting': 'LABOR_RECRUITMENT',
    'Security & Privacy': 'SECURITY_SERVICES', 'Government & Politics': 'GOVERNMENT'
}

# Building the SQL Logic
case_statement = "\n".join([f"WHEN themes LIKE '%{code}%' THEN '{name}'" for name, code in INDUSTRY_MAP.items()])

print(f"🚀 Starting the Full 15-Year Harvest on Project: {NEW_PROJECT_ID}...")

# QUERY 1: Modern Era (2015-2020) - Uses Partitioned Table (Efficient)
sql_modern = f"""
SELECT Year, Industry, COUNT(*) as ArticleVolume, ROUND(AVG(Sentiment), 4) as AvgSentiment
FROM (
    SELECT 
        EXTRACT(YEAR FROM _PARTITIONDATE) as Year,
        CASE {case_statement} ELSE 'Other' END as Industry,
        SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) as Sentiment
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE _PARTITIONDATE BETWEEN '2015-01-01' AND '2020-12-31'
)
WHERE Industry != 'Other'
GROUP BY 1, 2
"""

# QUERY 2: Baseline Era (2005-2014) - Uses Events Table (Very Quota-Safe)
sql_historical = """
SELECT 
    Year, 
    'Business General (Baseline)' as Industry, 
    COUNT(*) as ArticleVolume, 
    ROUND(AVG(AvgTone), 4) as AvgSentiment
FROM `gdelt-bq.full.events`
WHERE Year BETWEEN 2005 AND 2014
  AND (Actor1Type1Code = 'BUS' OR Actor2Type1Code = 'BUS')
  AND EventCode LIKE '03%'
GROUP BY 1, 2
"""

try:
    start_time = time.time()
    
    print("⏳ Pulling 2015-2020 Industry Data (Batch 1/2)...")
    df_modern = client.query(sql_modern).to_dataframe()
    
    print("⏳ Pulling 2005-2014 Historical Baseline (Batch 2/2)...")
    df_history = client.query(sql_historical).to_dataframe()
    
    # Merge and Save
    final_df = pd.concat([df_modern, df_history]).sort_values(['Industry', 'Year'])
    final_df.to_csv("GDELT_Final_Report_2005_2020.csv", index=False)
    
    print(f"\n✅ SUCCESS! File 'GDELT_Final_Report_2005_2020.csv' is ready.")
    print(f"⏱️ Total time: {round((time.time() - start_time)/60, 2)} minutes")

except Exception as e:
    print(f"⚠️ Error: {e}")