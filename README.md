### Opportunity Scout AI: Comprehensive Intelligence Platform

Opportunity Scout AI is a decision-support system designed to evaluate the viability of startup ventures. While traditional models focus strictly on internal financials, this platform introduces a context-aware evaluation engine. It merges venture-specific data (funding, industry, age) with real-time global signals (GDP growth, news sentiment, and social hype) to provide a 360-degree "Opportunity Score."

### Technical Architecture
The system operates as a full-stack Machine Learning application divided into three distinct layers:

#### 1. The Data & Feature Engineering Pipeline
The project processes a hybrid dataset comprised of:

Venture Core: A database of 25,000+ historical startups (from ventures.csv and startup_core.csv), tracking their funding history and ultimate status (operating, acquired, or closed).

Macro Signals: Global economic indicators from the World Bank API, including annual GDP growth and inflation rates for over 50 countries.

Momentum Signals: * Google Trends: Industry-specific keyword growth (e.g., "AI," "Fintech") to calculate the "Trend Slope."

GDELT Project: Global news volume and sentiment analysis to gauge media coverage.

Reddit API: Community density metrics to measure grassroots interest.

Normalization: All live inputs are standardized using pre-calculated Z-scores (zscore_params.csv) and Min-Max bounds (feature_bounds.csv) to ensure consistent model performance.

#### 2. The Machine Learning Engine
The "brain" of the platform is a lightGBM model optimized for high-dimensional tabular data.

Model Performance: Trained to minimize RMSE (Root Mean Square Error) against historical success outcomes.

Feature Engineering: Includes derived features like funding_log, trend_x_gdp (interaction term), and ecosystem_strength.

Explainable AI (XAI): Integrated with SHAP. The model doesn't just give a score; it calculates the "Shapley Values" for each prediction. The backend identifies the feature with the highest positive contribution to explain why a startup is high-potential (e.g., "Market momentum is the primary driver").

#### 3. API & Interactive Dashboard
FastAPI Backend: A high-performance Python API that handles asynchronous requests. When a user runs an analysis, the backend triggers the AggregationEngine to fetch live data before running the prediction.

Responsive UI: A dashboard built with modern CSS (Grid/Flexbox) that allows users to:

    1. Simulate: Test "What-If" scenarios by adjusting funding or location.

    2. Monitor: View aggregate portfolio stats like "Average Viability."

    3. Database Integration: Save real-time evaluations to a historical log for team review.


### How It Works (The Pipeline)
The project is divided into three distinct phases that work in a continuous loop:

#### 1. Data Fusion & Engineering
We don't just use one list of startups. We merged historical performance data (25,000+ entries) with live external signals.

Dynamic Scaling: We use Z-Score Normalization. This means if a startup has $1M in funding, the AI compares that $1M against the average for its specific country and year to see if it’s actually "well-funded" or just average.

Trend Analysis: We calculate the "Slope" of industry growth. A positive slope means an industry is emerging; a negative one means the hype is dying down.

#### 2. The Predictive Brain
We trained two high-performance models: XGBoost and LightGBM.

Explainable AI (SHAP): Every time the model gives a score, it uses SHAP values to explain the result. If a score is high, the AI will explicitly tell you: "Primary Driver: Strong Market Trend" or "Primary Driver: High Investment Climate."

Standardization: Using zscore_params.csv and feature_bounds.csv, the model ensures that a startup from 2005 and a startup from 2026 are evaluated fairly based on the economic context of their respective times.

#### 3. The Interactive Dashboard
A full-stack interface that allows users to move from "Data Science" to "Decision Making."

Portfolio View: See the average viability of all startups you've tracked.

Live Evaluation: A "What-If" simulator where you can input a new company, and the backend fetches live GDP and Trend data to give you an instant score.

#### Key Components
Backend & Logic (main.py): The FastAPI server that orchestrates the live data collectors and executes the prediction pipeline.

ML Models (opportunity_model.json): The production-ready model optimized for real-time web inference.

Research Lab (modeling_and_explainability.ipynb): The development environment where the models were trained, validated, and tested for accuracy.

Live Data Signals: Datasets like GDELT.csv, macro_data.csv, and reddit.csv provide the global news, economic, and social signals that feed the AI.

User Interface: A modern, professional dashboard built using index.html, style.css, and script.js.

#### Summary of Value
Opportunity Scout AI transforms venture analysis into a rigorous, data-driven process. By combining gradient boosting for prediction and SHAP for transparency, it provides a unique window into which startups are truly positioned to lead the market.