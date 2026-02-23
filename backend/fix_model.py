import joblib
import xgboost as xgb

# 1. Load the "broken" joblib file
model = joblib.load('opportunity_model.joblib')

# 2. Extract the actual XGBoost booster and save it properly
if hasattr(model, 'get_booster'):
    model.get_booster().save_model('opportunity_model.json')
    print("✅ Model converted to JSON format!")
else:
    # If it's already a booster
    model.save_model('opportunity_model.json')
    print("✅ Model saved to JSON format!")