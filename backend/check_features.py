import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'opportunity_model.joblib')

model = joblib.load(MODEL_PATH)

if hasattr(model, 'feature_names_in_'):
    print("Feature Names (In):")
    print(list(model.feature_names_in_))
elif hasattr(model, 'feature_name'):
    print("Feature Names (LightGBM/XGBoost):")
    # For LightGBM
    try:
        print(model.feature_name())
    except:
        # For XGBoost
        try:
            print(model.get_booster().feature_names)
        except:
            print("Could not find feature names.")
else:
    print("Model does not expose feature names directly.")
