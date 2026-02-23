import pandas as pd
import joblib
from sqlalchemy import create_engine
import google.generativeai as genai

# 1. SETUP: Database & Model
# Replace 'YOUR_PASSWORD' with your actual PostgreSQL password
DB_URL = "postgresql://postgres:password@localhost:5432/opportunity_scout"
engine = create_engine(DB_URL)
model = joblib.load('opportunity_model.joblib')

# 2. SETUP: AI Voice (Gemini)
# Replace 'YOUR_API_KEY' with your key from Google AI Studio
genai.configure(api_key="YOUR_API_KEY")
ai_scout = genai.GenerativeModel('gemini-1.5-flash')

def get_predictions_from_db():
    # Pull startups that are still operating
    query = "SELECT * FROM startups WHERE status = 'operating' LIMIT 500;"
    df = pd.read_sql(query, engine)
    
    # Select the exact features your model was trained on
    # (Based on your notebook: founded_year, funding_total_usd, funding_rounds, etc.)
    features = ['founded_year', 'funding_total_usd', 'funding_rounds', 'trend_slope', 
                'news_volume', 'news_sentiment', 'reddit_density', 'gdp_growth', 
                'inflation', 'startup_density']
    
    X = df[features].fillna(0) # Fill missing values just in case
    
    # Use your XGBoost model to predict scores
    df['ai_predicted_score'] = model.predict(X)
    
    # Sort by the best predicted opportunities
    return df.sort_values(by='ai_predicted_score', ascending=False).head(3)

def generate_report(top_startups):
    print("🤖 AI Scout is analyzing the data...")
    
    # Clean up the data for the AI to read easily
    data_summary = top_startups[['name', 'industry', 'ai_predicted_score']].to_string()
    
    prompt = f"""
    You are a professional Venture Capital Scout. 
    Here are the top 3 startups predicted by our XGBoost model:
    {data_summary}
    
    Provide a brief, 2-sentence investment pitch for why these industries/companies 
    are currently showing high potential based on their scores.
    """
    
    response = ai_scout.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    top_3 = get_predictions_from_db()
    print("\n--- TOP 3 OPPORTUNITIES FOUND IN DATABASE ---")
    print(top_3[['name', 'industry', 'ai_predicted_score']])
    
    report = generate_report(top_3)
    print("\n--- AI SCOUT INVESTMENT REPORT ---")
    print(report)