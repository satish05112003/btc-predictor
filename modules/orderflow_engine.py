buy_volume = 0.0
sell_volume = 0.0
trade_count = 0


def process_trade(price, size, side):
    global buy_volume, sell_volume, trade_count

    size = float(size)

    if side == "buy":
        buy_volume += size
    else:
        sell_volume += size

    trade_count += 1


def get_features():
    global buy_volume, sell_volume, trade_count

    imbalance = buy_volume - sell_volume
    total_volume = buy_volume + sell_volume

    if total_volume == 0:
        pressure = 0
    else:
        pressure = imbalance / total_volume

    return {
        "buy_volume": round(buy_volume, 2),
        "sell_volume": round(sell_volume, 2),
        "imbalance": round(imbalance, 2),
        "pressure": round(pressure, 4),
        "trade_count": trade_count
    }


def reset():
    global buy_volume, sell_volume, trade_count

    buy_volume = 0
    sell_volume = 0
    trade_count = 0
