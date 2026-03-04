"""
==============================================================================
  BTC CANDLE DIRECTION PREDICTOR
  Module 1: Data Collector
  Exchange : Coinbase (BTC/USD)
  Timeframes: 5m, 15m

  Coinbase API notes:
  - No API key required for public OHLCV data
  - Max 300 candles per request (vs Binance's 1000)
  - Symbol format: "BTC/USD" (not BTC/USDT)
  - Granularities supported: 1m, 5m, 15m, 1h, 6h, 1d
==============================================================================
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data"
SYMBOL      = "BTC/USD"            # Coinbase uses USD not USDT
TIMEFRAMES  = ["5m", "15m"]
MONTHS_BACK = 12
CANDLES_PER_CALL = 300             # Coinbase max per request is 300

DATA_DIR.mkdir(exist_ok=True)


# ── EXCHANGE SETUP ────────────────────────────────────────────────────────────
def get_exchange() -> ccxt.coinbase:
    """
    Returns a ccxt Coinbase exchange instance.
    No API key needed for public market data (OHLCV, ticker, orderbook).
    """
    exchange = ccxt.coinbase({
        "enableRateLimit": True,
    })
    return exchange


# ── FETCH FULL HISTORY ────────────────────────────────────────────────────────
def fetch_ohlcv_full(
    timeframe: str = "1h",
    months_back: int = 12,
    symbol: str = SYMBOL,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch full OHLCV history by paginating through Coinbase.
    Coinbase max 300 candles/request — loops ~350x for 12m of 5m data.
    Returns a clean DataFrame with timestamp index.
    """
    exchange = get_exchange()

    since_dt  = datetime.utcnow() - timedelta(days=30 * months_back)
    since_ms  = int(since_dt.timestamp() * 1000)

    print(f"[DataCollector] Fetching {symbol} {timeframe} from {since_dt.date()} ...")

    all_candles = []
    request_count = 0

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=CANDLES_PER_CALL
            )
        except ccxt.NetworkError as e:
            print(f"  ⚠  Network error: {e} — retrying in 10s")
            time.sleep(10)
            continue
        except ccxt.RateLimitExceeded:
            print("  ⚠  Rate limit — waiting 30s")
            time.sleep(30)
            continue
        except ccxt.ExchangeError as e:
            print(f"  ✗  Exchange error: {e}")
            break

        if not candles:
            break

        all_candles.extend(candles)
        since_ms      = candles[-1][0] + 1
        request_count += 1

        # Stop if we've caught up to now
        last_ts = pd.to_datetime(candles[-1][0], unit="ms", utc=True)
        if last_ts >= pd.Timestamp.utcnow() - pd.Timedelta(timeframe):
            break

        # Coinbase rate limit — more conservative than Binance
        time.sleep(max(exchange.rateLimit / 1000, 0.5))

        if request_count % 10 == 0:
            print(f"  ... {len(all_candles):,} candles fetched so far")

    df = _candles_to_df(all_candles)
    print(f"[DataCollector] ✓  {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")

    if save:
        path = DATA_DIR / f"btc_{timeframe}_raw.parquet"
        df.to_parquet(path)
        print(f"[DataCollector]    Saved → {path}")

    return df


def _candles_to_df(candles: list) -> pd.DataFrame:
    df = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    # Cast to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


# ── LOAD CACHED DATA ──────────────────────────────────────────────────────────
def load_data(timeframe: str = "5m", refresh: bool = False) -> pd.DataFrame:
    """
    Load cached parquet if available, otherwise fetch from Binance.
    Set refresh=True to force re-download.
    """
    path = DATA_DIR / f"btc_{timeframe}_raw.parquet"

    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        print(f"[DataCollector] Loaded cached {timeframe} data: {len(df):,} candles")
        return df

    return fetch_ohlcv_full(timeframe=timeframe, months_back=MONTHS_BACK)


# ── FETCH LATEST N CANDLES (for live use) ─────────────────────────────────────
def fetch_latest(timeframe: str = "5m", n: int = 300) -> pd.DataFrame:
    """
    Fetch the most recent N candles.
    Used in live prediction — needs 250+ for indicator warm-up.
    """
    exchange = get_exchange()
    candles  = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=n)
    df       = _candles_to_df(candles)
    # Drop the last (unclosed) candle
    df       = df.iloc[:-1]
    return df


# ── APPEND NEW CANDLES TO EXISTING DATA ───────────────────────────────────────
def update_data(timeframe: str = "5m") -> pd.DataFrame:
    """
    Load existing data and append any new candles since last entry.
    Efficient for keeping data fresh without full re-download.
    """
    path = DATA_DIR / f"btc_{timeframe}_raw.parquet"

    if not path.exists():
        return fetch_ohlcv_full(timeframe=timeframe)

    existing = pd.read_parquet(path)
    last_ts  = existing.index[-1]
    since_ms = int(last_ts.timestamp() * 1000) + 1

    exchange = get_exchange()
    candles  = exchange.fetch_ohlcv(SYMBOL, timeframe, since=since_ms, limit=CANDLES_PER_CALL)

    if not candles:
        print(f"[DataCollector] Data already up to date ({timeframe})")
        return existing

    new_df   = _candles_to_df(candles)
    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(path)

    print(f"[DataCollector] Updated {timeframe}: +{len(new_df)} new candles → {len(combined):,} total")
    return combined


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for tf in TIMEFRAMES:
        fetch_ohlcv_full(timeframe=tf, months_back=MONTHS_BACK)
