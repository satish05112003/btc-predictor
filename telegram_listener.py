import requests
import time
import os
import csv
import threading
from dotenv import load_dotenv
from performance_tracker import get_stats, get_today_stats, get_mistakes
from retrainer import retrain_model

load_dotenv()

BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_IDS = [x.strip() for x in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if x.strip()]

URL=f"https://api.telegram.org/bot{BOT_TOKEN}"

def send(chat_id,msg):
    url=URL+"/sendMessage"
    requests.post(url,json={"chat_id":chat_id,"text":msg})

def is_admin(user_id):
    return str(user_id) in ADMIN_IDS

def get_last_predictions(limit=5):
    if not os.path.exists("last_predictions.csv"):
        return "No predictions yet."
    
    with open("last_predictions.csv") as f:
        reader = list(csv.DictReader(f))
        
    if not reader:
        return "No predictions yet."
        
    recent = reader[-limit:]
    msg = "Recent Predictions:\n\n"
    for r in recent:
        msg += f"{r['tf']} | {r['signal']} @ {r['price']} (Conf: {r['confidence']}%)\n"
    return msg

def start_listener():
    print("Telegram listener running...")
    offset=None

    while True:
        try:
            r=requests.get(URL+"/getUpdates",params={"offset":offset})
            data=r.json()

            for update in data.get("result", []):
                offset=update["update_id"]+1

                if "message" not in update:
                    continue

                chat_id=update["message"]["chat"]["id"]
                user_id=update["message"]["from"]["id"]
                text=update["message"].get("text","")

                print("Telegram command received:", text)

                # --- PUBLIC COMMANDS ---
                if text=="/health" or text=="/system":
                    send(chat_id,"System Status : RUNNING 🟢")

                elif text=="/stats" or text=="/accuracy":
                    total,correct,accuracy=get_stats()
                    msg=f"BTC Prediction Arena\n\nTotal Predictions : {total}\nCorrect Predictions : {correct}\nAccuracy : {accuracy:.2f}%"
                    send(chat_id,msg)

                elif text=="/stats_today":
                    total,acc=get_today_stats()
                    send(chat_id,f"Today Predictions: {total}\nAccuracy: {acc:.2f}%")

                elif text=="/mistakes":
                    m=get_mistakes()
                    send(chat_id,f"Total Mistakes Logged for AI: {m}")
                    
                elif text=="/dashboard":
                    total,correct,accuracy=get_stats()
                    losses = total - correct
                    msg=f"BTC MODEL DASHBOARD\n\nSignals: {total}\nWins: {correct}\nLosses: {losses}\n\nAccuracy: {accuracy:.2f}%"
                    send(chat_id,msg)

                elif text=="/last" or text=="/predictions":
                    msg = get_last_predictions()
                    send(chat_id, msg)
                    
                elif text=="/help":
                    msg="""
Available Commands:
/health - System status
/stats - Total statistics
/stats_today - Today's stats
/mistakes - Mistake records count
/dashboard - Full overview
/accuracy - Same as stats
/last - Last 5 predictions
/help - This message

Admins:
/retrain - Force AI model retain
/reset_stats - Wipe performance file
/model - Check AI file status
"""
                    send(chat_id, msg)

                # --- ADMIN COMMANDS ---
                elif text=="/retrain":
                    if is_admin(user_id):
                        send(chat_id, "Triggering retrain...")
                        res = retrain_model()
                        send(chat_id, res)
                    else:
                        send(chat_id, "Unauthorized. You are not an admin.")

                elif text=="/reset_stats":
                    if is_admin(user_id):
                        if os.path.exists("prediction_performance.csv"):
                            os.remove("prediction_performance.csv")
                        send(chat_id, "Performance stats have been wiped.")
                    else:
                        send(chat_id, "Unauthorized.")
                        
                elif text=="/model":
                    if is_admin(user_id):
                        msg = ""
                        for tf in ["5m", "15m"]:
                            f = f"btc_model_{tf}.pkl"
                            if os.path.exists(f):
                                sz = os.path.getsize(f) / 1024
                                msg += f"Model {tf}: Active ({sz:.1f} KB)\n"
                            else:
                                msg += f"Model {tf}: Not found. Retrain needed.\n"
                        send(chat_id, msg)
                    else:
                        send(chat_id, "Unauthorized.")

        except Exception as e:
            # print("Telegram listener error:", e)
            pass
            
        time.sleep(2)
