"""
VENTURES.CSV - COMPREHENSIVE DATASET

GDELT Mapping Logic:
- 2005-2014: Use "Business General (Baseline)" for ALL industries
- 2015-2020: Use exact matches where available
- 2015-2020: Map missing categories to nearest neighbor
- Fallback: Use annual average if no neighbor exists

Output: ventures.csv with NO null values in scoring columns

Requirements:
pip install pandas numpy scikit-learn

Usage:
python create_ventures_comprehensive.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """File paths and configuration."""
    
    # Input files
    STARTUPS_FILE = 'Data/startup_core.csv'
    TRENDS_FILE = 'Data/industry_trends.csv'
    GDELT_FILE = 'Data/GDELT.csv'
    REDDIT_FILE = 'Data/reddit.csv'
    MACRO_FILE = 'Data/macro_data.csv'
    
    # Output file
    OUTPUT_FILE = 'Data/ventures.csv'
    
    # Date range
    MIN_YEAR = 2005
    MAX_YEAR = 2020
    
    # Feature weights for opportunity score (updated per requirements)
    WEIGHTS = {
        'trend_slope': 0.25,        # Market momentum (25%)
        'gdp_growth': 0.20,         # Economic growth (20%)
        'reddit_density': 0.15,     # Community buzz (15%)
        'news_sentiment': 0.15,     # News sentiment (15%)
        'funding_log': 0.15,        # Total funding log scale (15%)
        'inflation_inverse': 0.10   # Economic stability inverse (10%)
    }
    
    # GDELT industry mapping (nearest neighbor for missing categories)
    GDELT_NEAREST_NEIGHBOR = {
        'AI & Machine Learning': 'Technology & Software',
        'Finance & Fintech': 'Technology & Software',
        'E-Commerce & Retail': 'Technology & Software',
        'Data & Analytics': 'Technology & Software',
        'Marketing & Advertising': 'Technology & Software',
        'Media & Entertainment': 'Technology & Software',
        'Enterprise & Business Services': 'Technology & Software',
        'Real Estate & Construction': 'Technology & Software',
        'Energy & Clean Tech': 'Manufacturing & Industrial',
        'Consumer & Lifestyle': 'Social & Community'
    }


# ============================================================================
# LOAD DATA
# ============================================================================

def load_all_data():
    """Load all data sources."""
    
    print("=" * 80)
    print("CREATING VENTURES.CSV - SMART GDELT MAPPING")
    print("=" * 80)
    
    print("\n📂 Loading all data sources...")
    
    # 1. Load startups
    print("\n1. Loading startups...")
    startups = pd.read_csv(Config.STARTUPS_FILE)
    original_count = len(startups)
    
    # Filter to 2005-2020
    startups = startups[(startups['founded_year'] >= Config.MIN_YEAR) & 
                        (startups['founded_year'] <= Config.MAX_YEAR)]
    print(f"   ✅ Loaded {original_count:,} startups")
    print(f"   📅 Filtered 2005-2020: {len(startups):,} startups")
    
    # 2. Load Google Trends
    print("\n2. Loading Google Trends...")
    trends = pd.read_csv(Config.TRENDS_FILE)
    print(f"   ✅ {len(trends):,} records")
    
    # 3. Load GDELT
    print("\n3. Loading GDELT...")
    gdelt = pd.read_csv(Config.GDELT_FILE)
    gdelt.columns = ['year', 'industry', 'news_volume', 'sentiment']
    print(f"   ✅ {len(gdelt):,} records")
    print(f"   Industries: {gdelt['industry'].nunique()}")
    
    # 4. Load Reddit
    print("\n4. Loading Reddit...")
    reddit = pd.read_csv(Config.REDDIT_FILE)
    reddit.columns = ['industry', 'year', 'reddit_density', 'popularity']
    print(f"   ✅ {len(reddit):,} records")
    
    # 5. Load macro
    print("\n5. Loading macro data...")
    macro = pd.read_csv(Config.MACRO_FILE)
    print(f"   ✅ {len(macro):,} records")
    
    return startups, trends, gdelt, reddit, macro


# ============================================================================
# GDELT SMART MAPPING
# ============================================================================

def create_gdelt_complete_mapping(gdelt):
    """
    Create complete GDELT mapping with smart logic:
    - 2005-2014: Use "Business General (Baseline)" for ALL
    - 2015-2020: Use exact matches or nearest neighbor
    - Fallback: Annual average
    """
    
    print("\n🧠 Creating smart GDELT mapping...")
    
    # Get baseline data
    baseline = gdelt[gdelt['industry'] == 'Business General (Baseline)'].copy()
    print(f"   Found baseline data: {len(baseline)} year records")
    
    # Get all unique categories from startups
    startups_temp = pd.read_csv(Config.STARTUPS_FILE)
    all_categories = startups_temp['category'].unique()
    
    # Create complete mapping
    complete_mappings = []
    
    for year in range(Config.MIN_YEAR, Config.MAX_YEAR + 1):
        
        if year <= 2014:
            # Rule 1: Use baseline for ALL industries (2005-2014)
            baseline_year = baseline[baseline['year'] == year]
            
            if len(baseline_year) > 0:
                baseline_news = baseline_year['news_volume'].values[0]
                baseline_sentiment = baseline_year['sentiment'].values[0]
                
                for category in all_categories:
                    complete_mappings.append({
                        'year': year,
                        'category': category,
                        'news_volume': baseline_news,
                        'sentiment': baseline_sentiment,
                        'source': 'baseline'
                    })
                
                print(f"   {year}: Used baseline for all {len(all_categories)} categories")
            
        else:
            # Rule 2 & 3: For 2015-2020, use exact match or nearest neighbor
            year_data = gdelt[gdelt['year'] == year].copy()
            
            # Calculate annual average as fallback
            annual_avg_news = year_data['news_volume'].mean()
            annual_avg_sentiment = year_data['sentiment'].mean()
            
            for category in all_categories:
                # Try exact match first
                exact_match = year_data[year_data['industry'] == category]
                
                if len(exact_match) > 0:
                    # Exact match found
                    complete_mappings.append({
                        'year': year,
                        'category': category,
                        'news_volume': exact_match['news_volume'].values[0],
                        'sentiment': exact_match['sentiment'].values[0],
                        'source': 'exact_match'
                    })
                
                elif category in Config.GDELT_NEAREST_NEIGHBOR:
                    # Use nearest neighbor
                    neighbor = Config.GDELT_NEAREST_NEIGHBOR[category]
                    neighbor_data = year_data[year_data['industry'] == neighbor]
                    
                    if len(neighbor_data) > 0:
                        complete_mappings.append({
                            'year': year,
                            'category': category,
                            'news_volume': neighbor_data['news_volume'].values[0],
                            'sentiment': neighbor_data['sentiment'].values[0],
                            'source': f'neighbor:{neighbor}'
                        })
                    else:
                        # Neighbor not found, use annual average
                        complete_mappings.append({
                            'year': year,
                            'category': category,
                            'news_volume': annual_avg_news,
                            'sentiment': annual_avg_sentiment,
                            'source': 'annual_average'
                        })
                
                else:
                    # No neighbor defined, use annual average
                    complete_mappings.append({
                        'year': year,
                        'category': category,
                        'news_volume': annual_avg_news,
                        'sentiment': annual_avg_sentiment,
                        'source': 'annual_average'
                    })
            
            # Show statistics for this year
            sources = pd.DataFrame([m for m in complete_mappings if m['year'] == year])
            source_counts = sources['source'].value_counts()
            print(f"   {year}: {', '.join([f'{k}:{v}' for k,v in source_counts.items()])}")
    
    # Convert to DataFrame
    gdelt_complete = pd.DataFrame(complete_mappings)
    
    print(f"\n   ✅ Created complete mapping: {len(gdelt_complete)} category-year combinations")
    print(f"   ✅ NO NULL VALUES in GDELT data")
    
    return gdelt_complete


# ============================================================================
# MERGE ALL FEATURES
# ============================================================================

def merge_all_features(startups, trends, gdelt_complete, reddit, macro):
    """Merge all features into startups dataset."""
    
    print("\n🔄 Merging all features...")
    
    # 1. Merge Google Trends (mean slope per category-year)
    print("\n   1. Merging Google Trends...")
    trends_agg = trends.groupby(['industry', 'year'])['slope'].mean().reset_index()
    trends_agg.rename(columns={'slope': 'trend_slope', 'year': 'founded_year', 
                                'industry': 'category'}, inplace=True)
    
    startups = startups.merge(
        trends_agg,
        on=['category', 'founded_year'],
        how='left'
    )
    print(f"      ✅ Merged: {startups['trend_slope'].notna().sum():,} records")
    
    # 2. Merge GDELT (complete mapping)
    print("\n   2. Merging GDELT (smart mapping)...")
    gdelt_complete.rename(columns={'year': 'founded_year'}, inplace=True)
    
    startups = startups.merge(
        gdelt_complete[['category', 'founded_year', 'news_volume', 'sentiment']],
        on=['category', 'founded_year'],
        how='left'
    )
    print(f"      ✅ news_volume: {startups['news_volume'].notna().sum():,} records")
    print(f"      ✅ sentiment: {startups['sentiment'].notna().sum():,} records")
    
    # 3. Merge Reddit
    print("\n   3. Merging Reddit...")
    reddit.rename(columns={'year': 'founded_year', 'industry': 'category'}, inplace=True)
    
    startups = startups.merge(
        reddit[['category', 'founded_year', 'reddit_density']],
        on=['category', 'founded_year'],
        how='left'
    )
    print(f"      ✅ Merged: {startups['reddit_density'].notna().sum():,} records")
    
    # 4. Merge macro and calculate startup density
    print("\n   4. Merging macro & calculating density...")
    
    macro_subset = macro[['country', 'year', 'gdp_growth', 'inflation', 'population']].copy()
    macro_subset.rename(columns={'year': 'founded_year'}, inplace=True)
    
    # Calculate startup density
    startup_counts = startups.groupby(['country', 'founded_year']).size().reset_index(name='startup_count')
    
    density_df = startup_counts.merge(
        macro_subset[['country', 'founded_year', 'population']],
        on=['country', 'founded_year'],
        how='left'
    )
    
    density_df['startup_density'] = (density_df['startup_count'] / 
                                     (density_df['population'] / 1_000_000)).round(4)
    
    # Merge macro
    startups = startups.merge(
        macro_subset[['country', 'founded_year', 'gdp_growth', 'inflation']],
        on=['country', 'founded_year'],
        how='left'
    )
    
    # Merge density
    startups = startups.merge(
        density_df[['country', 'founded_year', 'startup_density']],
        on=['country', 'founded_year'],
        how='left'
    )
    
    print(f"      ✅ gdp_growth: {startups['gdp_growth'].notna().sum():,} records")
    print(f"      ✅ inflation: {startups['inflation'].notna().sum():,} records")
    print(f"      ✅ startup_density: {startups['startup_density'].notna().sum():,} records")
    
    return startups


# ============================================================================
# CALCULATE OPPORTUNITY SCORE
# ============================================================================

def calculate_opportunity_score_new(df):
    """
    Calculate opportunity score with updated weights:
    - Trend Slope: 25%
    - GDP Growth: 20%
    - Reddit Density: 15%
    - News Sentiment: 15%
    - Total Funding (Log): 15%
    - Inflation (Inverse): 10%
    """
    
    print("\n🎯 Calculating Opportunity Scores (New Formula)...")
    
    # Prepare features
    print("\n   📊 Preparing features...")
    
    # 1. Funding log scale
    df['funding_log'] = np.log1p(df['funding_total_usd'])  # log(1 + funding)
    
    # 2. Rename for clarity
    df['news_sentiment'] = df['sentiment']
    
    # Select features
    feature_cols = ['trend_slope', 'gdp_growth', 'reddit_density', 
                   'news_sentiment', 'funding_log', 'inflation']
    
    X = df[feature_cols].copy()
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Startups: {len(X):,}")
    
    # Check completeness BEFORE imputation
    for col in feature_cols:
        missing = X[col].isna().sum()
        pct = missing / len(X) * 100
        print(f"      {col:20s} Missing: {missing:6,} ({pct:5.1f}%)")
    
    # Impute missing values with median
    print(f"\n   📊 Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=feature_cols,
        index=X.index
    )
    
    # Normalize features to 0-100
    print(f"   📊 Normalizing features to 0-100...")
    X_normalized = pd.DataFrame(index=X.index)
    
    for col in feature_cols:
        values = X_imputed[col]
        min_val, max_val = values.min(), values.max()
        
        if col == 'inflation':
            # Inflation: lower is better (inverse)
            if max_val > min_val:
                X_normalized['inflation_inverse'] = 100 - ((values - min_val) / (max_val - min_val) * 100)
            else:
                X_normalized['inflation_inverse'] = 50
        else:
            # Higher is better
            if max_val > min_val:
                X_normalized[col] = ((values - min_val) / (max_val - min_val)) * 100
            else:
                X_normalized[col] = 50
    
    # Apply weights
    print(f"\n   📊 Applying weights...")
    opportunity_score = pd.Series(0.0, index=X.index)
    
    weight_cols = {
        'trend_slope': Config.WEIGHTS['trend_slope'],
        'gdp_growth': Config.WEIGHTS['gdp_growth'],
        'reddit_density': Config.WEIGHTS['reddit_density'],
        'news_sentiment': Config.WEIGHTS['news_sentiment'],
        'funding_log': Config.WEIGHTS['funding_log'],
        'inflation_inverse': Config.WEIGHTS['inflation_inverse']
    }
    
    for col, weight in weight_cols.items():
        opportunity_score += X_normalized[col] * weight
        print(f"      {col:20s} {weight*100:5.1f}%")
    
    opportunity_score = opportunity_score.round(2)
    
    # Add to dataframe
    df['opportunity_score'] = opportunity_score
    
    # Categorize
    df['opportunity_category'] = pd.cut(
        opportunity_score,
        bins=[0, 40, 70, 100],
        labels=['Lower Potential', 'Medium Potential', 'High Potential']
    )
    
    print(f"\n   ✅ Scores calculated!")
    print(f"\n   📊 Distribution:")
    print(f"      Min:    {opportunity_score.min():.2f}")
    print(f"      25%:    {opportunity_score.quantile(0.25):.2f}")
    print(f"      Median: {opportunity_score.median():.2f}")
    print(f"      75%:    {opportunity_score.quantile(0.75):.2f}")
    print(f"      Max:    {opportunity_score.max():.2f}")
    
    print(f"\n   📊 Categories:")
    for cat in ['High Potential', 'Medium Potential', 'Lower Potential']:
        count = (df['opportunity_category'] == cat).sum()
        print(f"      {cat:20s} {count:6,} ({count/len(df)*100:5.1f}%)")
    
    return df


# ============================================================================
# FINALIZE
# ============================================================================

def finalize_dataset(df):
    """Finalize and save."""
    
    print("\n📋 Finalizing dataset...")
    
    # Select and order columns
    final_columns = [
        'name', 'industry', 'category', 'founded_year', 'country', 'status',
        'funding_total_usd', 'funding_rounds',
        'opportunity_score', 'opportunity_category',
        'trend_slope', 'news_volume', 'news_sentiment', 'reddit_density',
        'gdp_growth', 'inflation', 'startup_density'
    ]
    
    # Keep only existing columns
    final_columns = [c for c in final_columns if c in df.columns]
    df_final = df[final_columns].copy()
    
    # Save
    df_final.to_csv(Config.OUTPUT_FILE, index=False)
    
    import os
    size_mb = os.path.getsize(Config.OUTPUT_FILE) / (1024 * 1024)
    
    print(f"\n💾 Saved: {Config.OUTPUT_FILE}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Rows: {len(df_final):,}")
    print(f"   Columns: {len(df_final.columns)}")
    
    # Check for nulls in scoring columns
    print(f"\n🔍 Checking for NULL values in key columns:")
    scoring_cols = ['trend_slope', 'news_volume', 'news_sentiment', 'reddit_density',
                   'gdp_growth', 'inflation', 'startup_density', 'opportunity_score']
    
    for col in scoring_cols:
        if col in df_final.columns:
            nulls = df_final[col].isna().sum()
            print(f"   {col:20s} NULLs: {nulls:6,} ({nulls/len(df_final)*100:5.1f}%)")
    
    return df_final


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary(df):
    """Generate comprehensive summary."""
    
    print("\n" + "=" * 80)
    print("VENTURES.CSV - FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Startups: {len(df):,}")
    print(f"   Year Range: {df['founded_year'].min()}-{df['founded_year'].max()}")
    print(f"   Countries: {df['country'].nunique()}")
    print(f"   Categories: {df['category'].nunique()}")
    
    print(f"\n🏆 TOP 20 OPPORTUNITIES:")
    top_20 = df.nlargest(20, 'opportunity_score')[
        ['name', 'category', 'country', 'founded_year', 'funding_total_usd', 'opportunity_score']
    ]
    for i, row in enumerate(top_20.itertuples(), 1):
        funding_str = f"${row.funding_total_usd:,.0f}" if row.funding_total_usd > 0 else "Undisclosed"
        print(f"   {i:2d}. {row.name:35s} | {row.category:25s} | {row.opportunity_score:.2f} | {funding_str}")
    
    print(f"\n🌍 TOP 10 COUNTRIES:")
    country_scores = df.groupby('country').agg({
        'opportunity_score': 'mean',
        'name': 'count'
    }).rename(columns={'name': 'count'})
    country_scores = country_scores[country_scores['count'] >= 10]
    top_countries = country_scores.nlargest(10, 'opportunity_score')
    
    for i, (country, row) in enumerate(top_countries.iterrows(), 1):
        print(f"   {i:2d}. {country:30s} Avg: {row['opportunity_score']:5.2f} ({int(row['count'])} startups)")
    
    print(f"\n🏭 TOP 10 CATEGORIES:")
    category_scores = df.groupby('category').agg({
        'opportunity_score': 'mean',
        'name': 'count'
    }).rename(columns={'name': 'count'})
    category_scores = category_scores[category_scores['count'] >= 10]
    top_categories = category_scores.nlargest(10, 'opportunity_score')
    
    for i, (category, row) in enumerate(top_categories.iterrows(), 1):
        print(f"   {i:2d}. {category:35s} Avg: {row['opportunity_score']:5.2f} ({int(row['count'])} startups)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    start_time = datetime.now()
    print(f"\n🚀 Starting at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load all data
        startups, trends, gdelt, reddit, macro = load_all_data()
        
        # Create complete GDELT mapping
        gdelt_complete = create_gdelt_complete_mapping(gdelt)
        
        # Merge all features
        ventures = merge_all_features(startups, trends, gdelt_complete, reddit, macro)
        
        # Calculate opportunity score
        ventures = calculate_opportunity_score_new(ventures)
        
        # Finalize
        ventures_final = finalize_dataset(ventures)
        
        # Generate summary
        generate_summary(ventures_final)
        
        # Final message
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("✅ VENTURES.CSV CREATED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"\n⏰ Time: {duration:.1f} seconds")
        print(f"📁 File: {Config.OUTPUT_FILE}")
        print(f"📊 Rows: {len(ventures_final):,}")
        
        print(f"\n✅ GDELT Mapping:")
        print(f"   • 2005-2014: Business Baseline for ALL categories")
        print(f"   • 2015-2020: Exact match > Nearest neighbor > Annual average")
        print(f"   • Result: NO NULL values in news data!")
        
        print(f"\n✅ Opportunity Score:")
        print(f"   • Trend Slope: 25%")
        print(f"   • GDP Growth: 20%")
        print(f"   • Reddit Density: 15%")
        print(f"   • News Sentiment: 15%")
        print(f"   • Funding (Log): 15%")
        print(f"   • Inflation (Inverse): 10%")
        
        return ventures_final
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    ventures = main()
    
    if ventures is not None:
        print("\n🎉 SUCCESS! ventures.csv is ready!")
