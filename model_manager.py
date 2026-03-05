import os
import joblib

MODEL_5M = "btc_model_5m.pkl"
MODEL_15M = "btc_model_15m.pkl"

models = {}

def load_models():
    global models
    
    if os.path.exists(MODEL_5M):
        models["5m"] = joblib.load(MODEL_5M)
        print("ML model loaded for 5m")
    else:
        models["5m"] = None
        print("No ML model for 5m yet — skipping ML filter")

    if os.path.exists(MODEL_15M):
        models["15m"] = joblib.load(MODEL_15M)
        print("ML model loaded for 15m")
    else:
        models["15m"] = None
        print("No ML model for 15m yet — skipping ML filter")

def predict_ml(tf, features):
    model = models.get(tf)
    if model is None:
        return None
        
    try:
        pred = model.predict([features])[0]
        return "UP" if pred == 1 else "DOWN"
    except Exception as e:
        print("ML prediction skipped:", e)
        return None
