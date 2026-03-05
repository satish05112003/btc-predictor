import websocket
import json

WS_URL = "wss://ws-feed.exchange.coinbase.com"

def on_open(ws):
    print("Connected to Coinbase WebSocket")

    subscribe = {
        "type": "subscribe",
        "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}]
    }

    ws.send(json.dumps(subscribe))


def on_message(ws, message):
    data = json.loads(message)

    if data.get("type") == "ticker":
        price = data["price"]
        print("BTC Price:", price)


def on_error(ws, error):
    print("Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("Connection closed")


ws = websocket.WebSocketApp(
    WS_URL,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

ws.run_forever()
