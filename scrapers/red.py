import requests
import pandas as pd
import time
from datetime import datetime

# 1. CONFIGURATION
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

def get_popularity_level(density):
    if density >= 10: return "Viral"
    if density >= 1: return "Mainstream"
    if density >= 0.1: return "Emerging"
    return "Niche"

def get_mention_density(keyword, year):
    url = "https://api.pullpush.io/reddit/search/submission/"
    start_time = int(time.mktime(time.strptime(f"{year}-01-01", "%Y-%m-%d")))
    
    params = {
        'q': keyword,
        'after': start_time,
        'size': 100,
        'sort': 'asc'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200: return 0.0
        
        data = response.json().get('data', [])
        if len(data) < 2: return 0.0
            
        first_time = data[0]['created_utc']
        last_time = data[-1]['created_utc']
        days_elapsed = (last_time - first_time) / 86400
        
        if days_elapsed <= 0: return 0.0
        return round(len(data) / days_elapsed, 4)
    except:
        return 0.0

def run_reddit_collection():
    results = []
    total_industries = len(INDUSTRY_KEYWORDS)
    
    print(f"Starting Reddit Industry Analysis (2005-2020)...")
    
    for i, (industry, keywords) in enumerate(INDUSTRY_KEYWORDS.items(), 1):
        print(f"[{i}/{total_industries}] Processing {industry}...")
        
        for year in range(MIN_YEAR, MAX_YEAR + 1):
            # To be thorough, we average the density across the industry's keywords
            keyword_densities = []
            for kw in keywords:
                density = get_mention_density(kw, year)
                keyword_densities.append(density)
                # Small sleep to be kind to the API
                time.sleep(0.5)
            
            avg_density = round(sum(keyword_densities) / len(keyword_densities), 4)
            pop_level = get_popularity_level(avg_density)
            
            results.append({
                'Industry': industry,
                'Year': year,
                'Daily_Mention_Density': avg_density,
                'Popularity_Level': pop_level
            })
            
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('reddit.csv', index=False)
    print("\nSUCCESS: 'reddit.csv' has been created.")

if __name__ == "__main__":
    run_reddit_collection()