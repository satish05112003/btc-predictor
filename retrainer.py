import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import time
import os

def retrain_model():
    print("[RETRAINER] Checking if retrain is possible...")
    if not os.path.exists("mistakes_dataset.csv"):
        print("[RETRAINER] No mistakes dataset found. Skipping.")
        return "No mistakes dataset found."
        
    df = pd.read_csv("mistakes_dataset.csv")
    if len(df) < 50:
        msg = f"Not enough data to retrain (Need 50, got {len(df)})."
        print("[RETRAINER]", msg)
        return msg
        
    features = ["RSI", "volume", "orderflow_buy", "orderflow_sell", "imbalance", "pressure", "volatility"]
    
    # Remove NaN values in necessary columns
    df = df.dropna(subset=features + ["actual_direction"])
    if len(df) < 50:
        msg = "Not enough valid data after cleaning."
        print("[RETRAINER]", msg)
        return msg
        
    # Process both timeframes
    messages = []
    for tf in ["5m", "15m"]:
        df_tf = df[df["timeframe"] == tf]
        if len(df_tf) >= 50:
            X = df_tf[features]
            y = (df_tf["actual_direction"] == "UP").astype(int)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)
            
            joblib.dump(clf, f"btc_model_{tf}.pkl")
            msg = f"{tf} model retrained successfully with {len(df_tf)} samples."
            print(f"[RETRAINER] {msg}")
            messages.append(msg)
        else:
            msg = f"Not enough data to retrain {tf} (Need 50, got {len(df_tf)})."
            print(f"[RETRAINER] {msg}")
            messages.append(msg)
            
    return "\n".join(messages)

def retrain_loop():
    print("[RETRAINER] Loop started. Retraining every 6 hours.")
    while True:
        try:
            retrain_model()
        except Exception as e:
            print("[RETRAINER] Error:", e)
        time.sleep(60 * 60 * 6) # Sleep for 6 hours
