import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()
BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID=os.getenv("TELEGRAM_CHAT_ID")

def send_accuracy_chart():
    if not os.path.exists("prediction_performance.csv"):
        print("[CHARTS] No performance data to chart.")
        return
        
    df = pd.read_csv("prediction_performance.csv")
    if len(df) < 5:
        print("[CHARTS] Not enough data for chart.")
        return
        
    df["correct"] = df["correct"].astype(str).str.lower() == "true"
    df["rolling_acc"] = df["correct"].rolling(window=10, min_periods=1).mean() * 100

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(df) + 1), df["rolling_acc"], marker='o', label="Rolling Accuracy", color="blue")
    plt.title("BTC Model Accuracy Trend")
    plt.xlabel("Predictions")
    plt.ylabel("Accuracy %")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    payload = {"chat_id": CHAT_ID, "caption": "📊 Live Performance Chart"}
    files = {"photo": ("chart.png", buf, "image/png")}
    
    try:
        if BOT_TOKEN and BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            requests.post(url, data=payload, files=files, timeout=10)
            print("[CHARTS] Accuracy chart sent to Telegram.")
    except Exception as e:
        print("[CHARTS] Telegram error:", e)
