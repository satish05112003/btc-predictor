import os
import joblib

MODEL_5M = "btc_model_5m.pkl"
MODEL_15M = "btc_model_15m.pkl"

models = {}

def load_models():
    global models
    
    if os.path.exists(MODEL_5M):
        models["5m"] = joblib.load(MODEL_5M)
        print("5m ML model loaded")
    else:
        print("No ML model for 5m yet — running without ML")

    if os.path.exists(MODEL_15M):
        models["15m"] = joblib.load(MODEL_15M)
        print("15m ML model loaded")
    else:
        print("No ML model for 15m yet — running without ML")

def predict_ml(tf, features):
    if tf not in models:
        return None
        
    model = models[tf]
    pred = model.predict([features])[0]
    return "UP" if pred == 1 else "DOWN"
