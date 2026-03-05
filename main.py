import websocket
import json
import time
import datetime
import zoneinfo
from datetime import UTC
import threading

from btc_predictor_all_in_one import predict_now, dispatch
from modules.orderflow_engine import process_trade, get_features, reset
from telegram_listener import start_listener
from telegram_notifier import send_telegram
from prediction_logger import log_prediction
from performance_tracker import log_result
from mistake_logger import log_mistake
from charts import send_accuracy_chart
from retrainer import retrain_loop
from keep_alive import start_keep_alive
from model_manager import load_models


WS_URL = "wss://ws-feed.exchange.coinbase.com"

last_5m = None
last_15m = None

# prevent duplicate predictions
last_prediction_time = {"5m": None, "15m": None}

# pending predictions to evaluate accuracy when next candle closes
pending_predictions = {"5m": None, "15m": None}

prediction_count = 0



def run_prediction(tf):

    global last_prediction_time, pending_predictions, prediction_count

    now = datetime.datetime.now(UTC)
    candle = now.replace(second=0, microsecond=0)

    # skip duplicate predictions
    if last_prediction_time[tf] == candle:
        return

    last_prediction_time[tf] = candle

    print("\nRunning", tf, "prediction...\n")

    features = get_features()

    print("Order Flow Stats")
    print("----------------")
    print("Buy Volume :", features["buy_volume"])
    print("Sell Volume:", features["sell_volume"])
    print("Imbalance  :", features["imbalance"])
    print("Pressure   :", features["pressure"])
    print("Trades     :", features["trade_count"])
    print()

    r = predict_now(tf, of_features=features)
    current_close = r["current_close"]

    # 1. EVALUATE PAST PREDICTION (Did the previous candle close matching prediction?)
    prev = pending_predictions.get(tf)
    if prev and prev["signal"] in ["UP", "DOWN"]:
        actual_direction = "UP" if current_close > prev["price"] else "DOWN"
        
        # Track accuracy
        log_result(tf, prev["signal"], actual_direction)

        # Log mistake if it was wrong
        if prev["signal"] != actual_direction:
            feature_row = {
                "tf": tf,
                "prediction": prev["signal"],
                "actual": actual_direction,
                "RSI": prev["rsi"],
                "volume": prev["volume"],
                "buy": prev["buy_volume"],
                "sell": prev["sell_volume"],
                "imbalance": prev["imbalance"],
                "pressure": prev["pressure"],
                "volatility": prev["volatility"]
            }
            log_mistake(feature_row)

    # 2. STORE NEW PREDICTION FOR EVALUATION ON NEXT CANDLE CLOSE
    pending_predictions[tf] = {
        "signal": r["signal"],
        "price": current_close,
        "rsi": r.get("rsi", 0),
        "volume": r.get("volume", 0),
        "volatility": r.get("volatility", 0),
        "buy_volume": features["buy_volume"],
        "sell_volume": features["sell_volume"],
        "imbalance": features["imbalance"],
        "pressure": features["pressure"],
    }

    # Execute original dispatch logic (logs to CSV, sends webhook)
    dispatch(r)

    # 3. LOG LATEST PREDICTION
    log_prediction(
        tf,
        r["signal"],
        current_close,
        round(r["prob_up"] * 100, 1),
        round(r["prob_down"] * 100, 1),
        round(r["confidence"] * 100, 1)
    )

    # 4. SEND TELEGRAM ALERT FOR NEW PREDICTION
    try:
        dt_utc = datetime.datetime.fromisoformat(r['timestamp'])
        tz_ist = zoneinfo.ZoneInfo("Asia/Kolkata")
        dt_ist = dt_utc.astimezone(tz_ist)
        tf_mins = 5 if r['timeframe'] == "5m" else 15
        start_dt = dt_ist + datetime.timedelta(minutes=tf_mins)
        end_dt = start_dt + datetime.timedelta(minutes=tf_mins)
        t1 = start_dt.strftime("%I:%M").lstrip("0")
        t2 = end_dt.strftime("%I:%M %p").lower().lstrip("0")
        time_str = f"{t1}-{t2}"
    except Exception:
        time_str = r['timestamp']

    tf_str = "5 MIN" if r["timeframe"] == "5m" else "15 MIN"
    signal = r["signal"]
    signal_str = ""
    if signal == "UP":
        signal_str = "🟢 Direction: LONG (UP)"
    elif signal == "DOWN":
        signal_str = "🔴 Direction: SHORT (DOWN)"
    else:
        signal_str = "⏸️ Direction: SKIP"
        
    price = f"{r['current_close']:,.2f}"
    p_up = round(r["prob_up"]*100, 1)
    p_down = round(r["prob_down"]*100, 1)
    conf = round(r["confidence"]*100, 1)

    msg = f"🚨 BTC SIGNAL ( {tf_str} )\n\n"
    msg += f"{signal_str}\n\n"
    msg += f"💰 Price: ${price}\n\n"
    msg += f"📊 Probabilities\n"
    msg += f"⬆️ UP: {p_up}%\n"
    msg += f"⬇️ DOWN: {p_down}%\n\n"
    msg += f"💪 Confidence: {conf}%\n\n"
    msg += f"⏱ Time: {time_str}"

    if signal in ["UP", "DOWN", "SKIP"]:
        send_telegram(msg)

    prediction_count += 1
    if prediction_count % 20 == 0:
        send_accuracy_chart()

    if tf == "15m":
        reset()


def on_open(ws):

    print("Connected to Coinbase WebSocket\n")

    subscribe = {
        "type": "subscribe",
        "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}]
    }

    ws.send(json.dumps(subscribe))


def on_message(ws, message):

    global last_5m, last_15m

    data = json.loads(message)

    if data.get("type") == "ticker":

        time.sleep(0.05)

        process_trade(
            price=data["price"],
            size=data["last_size"],
            side=data["side"]
        )

        now = datetime.datetime.now(UTC)

        if now.minute % 5 == 0 and now.second < 2:

            candle = now.replace(second=0, microsecond=0)

            if candle != last_5m:

                last_5m = candle

                print("\n5m candle closed -> running prediction")

                run_prediction("5m")

        if now.minute % 15 == 0 and now.second < 2:

            candle = now.replace(second=0, microsecond=0)

            if candle != last_15m:

                last_15m = candle

                print("\n15m candle closed -> running prediction")

                run_prediction("15m")


def on_error(ws, error):
    print("Error:", error)


def on_close(ws, close_status_code, close_msg):

    print("\nConnection closed. Reconnecting in 5 seconds...\n")

    time.sleep(5)

    start_ws()


def start_ws():

    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()

def run_prediction_engine():
    start_ws()

def start_system():
    print("Starting BTC Prediction Arena...")

    start_keep_alive()
    load_models()

    # Start Telegram listener
    threading.Thread(
        target=start_listener,
        daemon=True
    ).start()

    print("Telegram listener started")

    # Start AI Retrainer loop
    threading.Thread(
        target=retrain_loop,
        daemon=True
    ).start()

    print("AI Retrainer started")

    # Continue running prediction engine
    run_prediction_engine()

def main():
    while True:
        try:
            start_system()
        except Exception as e:
            print("System crashed:", e)
            print("Restarting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()
