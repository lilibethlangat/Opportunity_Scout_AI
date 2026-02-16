import pandas as pd
import numpy as np
import time
from datetime import datetime
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression

# 1. INITIALIZE PYTRENDS (Simplified to avoid library version conflicts)
# We remove 'retries' and 'backoff_factor' from here to avoid the 'method_whitelist' error.
pytrends = TrendReq(hl='en-GB', tz=0)

# 2. CONFIGURATION
MIN_YEAR = 2005 
MAX_YEAR = 2020

# Your provided industry list
INDUSTRY_KEYWORDS = {
    'E-Commerce & Retail': ['ecommerce', 'online shopping', 'retail technology', 'online retail'],
    'Technology & Software': ['software', 'saas', 'cloud computing', 'enterprise software'],
    'Healthcare & Life Sciences': ['digital health', 'telemedicine', 'healthcare technology', 'medical devices', 'biotech'],
    'Finance & Fintech': ['fintech', 'digital banking', 'payment technology', 'cryptocurrency', 'blockchain finance'],
    'Media & Entertainment': ['streaming services', 'digital media', 'content creation', 'entertainment technology'],
    'Education & Training': ['edtech', 'online learning', 'elearning', 'educational technology'],
    'Real Estate & Construction': ['proptech', 'real estate technology', 'construction technology', 'smart buildings'],
    'Travel & Hospitality': ['travel technology', 'online booking', 'hotel technology', 'tourism technology'],
    'Food & Beverage': ['food delivery', 'food technology', 'restaurant technology', 'meal kit'],
    'Transportation & Logistics': ['logistics technology', 'supply chain', 'delivery services', 'transportation technology', 'fleet management'],
    'Marketing & Advertising': ['digital marketing', 'martech', 'advertising technology', 'marketing automation'],
    'Internet & Web Services': ['web services', 'internet technology', 'web applications', 'online services'],
    'Data & Analytics': ['data analytics', 'big data', 'business intelligence', 'data science'],
    'Gaming': ['video games', 'gaming', 'esports', 'mobile gaming'],
    'Social & Community': ['social media', 'social networking', 'online community', 'social platform'],
    'Consumer & Lifestyle': ['consumer technology', 'lifestyle products', 'consumer goods', 'personal technology'],
    'Manufacturing & Industrial': ['industrial automation', 'manufacturing technology', 'industry 4.0', 'smart manufacturing'],
    'Sports & Recreation': ['fitness technology', 'sports technology', 'wellness', 'fitness apps'],
    'Communications & Telecom': ['telecommunications', 'communication technology', '5g technology', 'telecom'],
    'Energy & Clean Tech': ['clean energy', 'renewable energy', 'cleantech', 'solar energy', 'energy technology'],
    'Enterprise & Business Services': ['business software', 'enterprise solutions', 'business services', 'b2b software'],
    'Security & Privacy': ['cybersecurity', 'information security', 'data privacy', 'network security'],
    'AI & Machine Learning': ['artificial intelligence', 'machine learning', 'deep learning', 'ai technology'],
    'Customer Service & CRM': ['crm software', 'customer service', 'customer experience', 'help desk software'],
    'HR & Recruiting': ['recruitment technology', 'hr software', 'talent management', 'recruiting'],
    'Government & Politics': ['govtech', 'government technology', 'civic technology', 'public sector technology']
}

def calculate_slope(y_values):
    if len(y_values) < 2 or np.all(y_values == 0):
        return 0.0
    X = np.arange(len(y_values)).reshape(-1, 1)
    y = y_values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return float(model.coef_[0][0])

def run_collection():
    results = []
    total_industries = len(INDUSTRY_KEYWORDS)
    
    print("="*80)
    print(f"Starting Multi-Keyword Data Collection | Year Range: {MIN_YEAR}-{MAX_YEAR}")
    print(f"Total Industries: {total_industries} | UK Base")
    print("="*80)

    for i, (industry, keywords) in enumerate(INDUSTRY_KEYWORDS.items(), 1):
        print(f"\n[{i}/{total_industries}] Processing: {industry}")
        
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            timeframe = f'{year}-01-01 {year}-12-31'
            
            # Simple manual retry logic
            success = False
            attempts = 0
            while not success and attempts < 3:
                try:
                    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='')
                    data = pytrends.interest_over_time()

                    if not data.empty:
                        if 'isPartial' in data.columns:
                            data = data.drop(columns=['isPartial'])
                        
                        industry_slopes = []
                        for kw in keywords:
                            if kw in data.columns:
                                slope = calculate_slope(data[kw].values)
                                industry_slopes.append(slope)
                                results.append({
                                    'industry': industry, 'year': year, 'keyword': kw, 'slope': slope
                                })
                        
                        avg_slope = np.mean(industry_slopes) if industry_slopes else 0
                        print(f"  {year}: Success (Avg Slope: {avg_slope:+.4f})")
                        success = True
                    else:
                        print(f"  {year}: No data.")
                        success = True # Don't retry if Google explicitly says 'no data'
                    
                    time.sleep(7) # Respectful delay

                except Exception as e:
                    attempts += 1
                    print(f"  {year} Attempt {attempts} failed: {e}")
                    time.sleep(10 * attempts) # Wait longer each time it fails

    if results:
        df_results = pd.DataFrame(results)
        # Fixed filename as discussed
        filename = "industry_trends_final.csv"
        df_results.to_csv(filename, index=False)
        print(f"\nDONE! Data saved to {filename}")

if __name__ == "__main__":
    run_collection()