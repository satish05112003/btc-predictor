"""
==============================================================================
  BTC CANDLE DIRECTION PREDICTOR
  Module 4: Live Predictor + Alerting
  Runs at each candle close and outputs probability-based signals.

  Alert channels:
    ✅ Console (always)
    ✅ CSV log file
    ✅ Telegram bot
    ✅ Discord / Slack webhook
==============================================================================
"""

import os
import json
import time
import requests
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ── CONFIG (edit these or use environment variables) ──────────────────────────

CONFIDENCE_THRESHOLD = 0.60   # Only signal when max(P_up, P_down) > this
TIMEFRAMES           = ["5m", "15m"]

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")   # e.g. "123456:ABC-DEF..."
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")   # e.g. "-1001234567890"

# Discord / Slack webhook URL
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")         # or SLACK_WEBHOOK_URL

LOGS_DIR  = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "predictor.log"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_next_candle(
    timeframe: str = "5m",
    models: dict   = None,
    feature_cols: list = None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Fetch latest candles, build features, run all models, return signal dict.
    Uses ensemble soft-voting (average of all loaded models).
    """
    from data_collector   import fetch_latest
    from feature_engineer import build_features, get_feature_columns, inject_mtf_features

    if models is None or feature_cols is None:
        from model_trainer import load_latest_models
        models, feature_cols = load_latest_models(timeframe)

    # ── Fetch data ────────────────────────────────────────────────────────────
    log.info(f"Fetching latest candles for {timeframe} ...")
    df = fetch_latest(timeframe=timeframe, n=300)   # 300 candles for warmup

    # For 5m: also fetch 15m for MTF features
    if timeframe == "5m":
        df_15m = fetch_latest(timeframe="15m", n=100)
        df_15m_feat = build_features(df_15m)
        df_feat = build_features(df)
        df_feat = inject_mtf_features(df_feat, df_15m_feat, prefix="h1_")
    else:
        df_feat = build_features(df)

    df_feat = df_feat.dropna()

    # ── Extract last row (most recent CLOSED candle) ───────────────────────────
    # Align to available features
    available = [c for c in feature_cols if c in df_feat.columns]
    missing   = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        log.warning(f"{len(missing)} feature(s) missing: {missing[:5]} ...")

    last_row   = df_feat[available].iloc[[-1]]
    candle_ts  = df_feat.index[-1]
    current_close = df.iloc[-1]["close"]

    # ── Run ensemble ──────────────────────────────────────────────────────────
    probas = []
    for name, model in models.items():
        try:
            p = model.predict_proba(last_row)[0][1]
            probas.append(p)
            log.debug(f"  {name}: P(UP)={p:.4f}")
        except Exception as e:
            log.warning(f"  Model {name} failed: {e}")

    if not probas:
        log.error("All models failed — no prediction")
        return {}

    prob_up   = float(np.mean(probas))
    prob_down = 1.0 - prob_up
    confidence = max(prob_up, prob_down)

    # ── Signal determination ──────────────────────────────────────────────────
    if prob_up > threshold:
        signal = "UP"
        emoji  = "🟢"
    elif prob_down > threshold:
        signal = "DOWN"
        emoji  = "🔴"
    else:
        signal = "SKIP"
        emoji  = "⏸️"

    result = {
        "timestamp"    : candle_ts.isoformat(),
        "timeframe"    : timeframe,
        "current_close": round(current_close, 2),
        "prob_up"      : round(prob_up,    4),
        "prob_down"    : round(prob_down,  4),
        "confidence"   : round(confidence, 4),
        "signal"       : signal,
        "emoji"        : emoji,
        "n_models"     : len(probas),
    }

    _print_signal(result)
    return result


def _print_signal(r: dict):
    """Console output."""
    print(f"\n{'═'*55}")
    print(f"  BTC/USD {r['timeframe']} — Candle Close Signal")
    print(f"{'─'*55}")
    print(f"  Candle close : {r['timestamp']}")
    print(f"  Close price  : ${r['current_close']:,.2f}")
    print(f"  P(UP)        : {r['prob_up']:.4f}  ({r['prob_up']*100:.1f}%)")
    print(f"  P(DOWN)      : {r['prob_down']:.4f}  ({r['prob_down']*100:.1f}%)")
    print(f"  Confidence   : {r['confidence']:.4f}")
    print(f"  Signal       : {r['emoji']}  {r['signal']}")
    print(f"{'═'*55}")


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT CHANNELS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. CSV LOG ────────────────────────────────────────────────────────────────

def log_to_csv(result: dict, timeframe: str = "5m"):
    """Append prediction result to a daily CSV file."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    path     = LOGS_DIR / f"signals_{timeframe}_{date_str}.csv"
    is_new   = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if is_new:
            writer.writeheader()
        writer.writerow({k: v for k, v in result.items() if k != "emoji"})

    log.info(f"Logged to CSV → {path.name}")


# ── 2. TELEGRAM ───────────────────────────────────────────────────────────────

def send_telegram(result: dict, bot_token: str = None, chat_id: str = None):
    """
    Send signal to Telegram channel/group.

    Setup:
      1. Create a bot via @BotFather → copy token
      2. Add bot to your channel/group
      3. Get chat_id via https://api.telegram.org/bot<TOKEN>/getUpdates
      4. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    """
    token   = bot_token or TELEGRAM_BOT_TOKEN
    chat    = chat_id   or TELEGRAM_CHAT_ID

    if not token or not chat:
        log.debug("Telegram not configured — skipping")
        return False

    if result.get("signal") == "SKIP":
        return False   # Only alert on actual signals

    msg = _format_telegram_message(result)

    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat, "text": msg, "parse_mode": "Markdown"}

    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            log.info("Telegram alert sent ✓")
            return True
        else:
            log.warning(f"Telegram failed: {resp.status_code} {resp.text}")
            return False
    except requests.RequestException as e:
        log.error(f"Telegram error: {e}")
        return False


def _format_telegram_message(r: dict) -> str:
    signal_line = {
        "UP":   "🟢 *LONG SIGNAL* — Next candle predicted UP",
        "DOWN": "🔴 *SHORT SIGNAL* — Next candle predicted DOWN",
        "SKIP": "⏸️ No signal (low confidence)",
    }.get(r["signal"], "")

    return (
        f"*BTC/USD {r['timeframe']} Signal*\n"
        f"{'─'*28}\n"
        f"{signal_line}\n\n"
        f"📊 Close: `${r['current_close']:,.2f}`\n"
        f"⬆️ P(UP):    `{r['prob_up']*100:.1f}%`\n"
        f"⬇️ P(DOWN):  `{r['prob_down']*100:.1f}%`\n"
        f"💪 Confidence: `{r['confidence']*100:.1f}%`\n"
        f"⏱ Time: `{r['timestamp']}`"
    )


# ── 3. DISCORD / SLACK WEBHOOK ────────────────────────────────────────────────

def send_webhook(result: dict, url: str = None):
    """
    Send signal to Discord or Slack via webhook.

    Discord setup:
      Server Settings → Integrations → Webhooks → New Webhook → Copy URL

    Slack setup:
      api.slack.com/apps → Create App → Incoming Webhooks → Activate → Copy URL
    """
    webhook = url or WEBHOOK_URL

    if not webhook:
        log.debug("Webhook not configured — skipping")
        return False

    if result.get("signal") == "SKIP":
        return False

    # Detect Discord vs Slack by URL pattern
    is_discord = "discord.com" in webhook

    if is_discord:
        payload = _discord_payload(result)
    else:
        payload = _slack_payload(result)

    try:
        resp = requests.post(webhook, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            log.info(f"{'Discord' if is_discord else 'Slack'} alert sent ✓")
            return True
        else:
            log.warning(f"Webhook failed: {resp.status_code} {resp.text[:200]}")
            return False
    except requests.RequestException as e:
        log.error(f"Webhook error: {e}")
        return False


def _discord_payload(r: dict) -> dict:
    color = 0x00C851 if r["signal"] == "UP" else 0xFF4444  # green / red

    return {
        "embeds": [{
            "title"      : f"BTC/USD {r['timeframe']} Signal {r['emoji']}",
            "description": (
                f"**{'🟢 LONG' if r['signal'] == 'UP' else '🔴 SHORT'} — "
                f"Next candle predicted {r['signal']}**"
            ),
            "color"  : color,
            "fields" : [
                {"name": "Close Price",  "value": f"${r['current_close']:,.2f}", "inline": True},
                {"name": "P(UP)",        "value": f"{r['prob_up']*100:.1f}%",   "inline": True},
                {"name": "P(DOWN)",      "value": f"{r['prob_down']*100:.1f}%", "inline": True},
                {"name": "Confidence",   "value": f"{r['confidence']*100:.1f}%","inline": True},
                {"name": "Models Used",  "value": str(r["n_models"]),           "inline": True},
                {"name": "Candle Time",  "value": r["timestamp"],               "inline": False},
            ],
            "footer" : {"text": "BTC Candle Predictor"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    }


def _slack_payload(r: dict) -> dict:
    color = "good" if r["signal"] == "UP" else "danger"

    return {
        "attachments": [{
            "color"    : color,
            "title"    : f"BTC/USD {r['timeframe']} {r['emoji']} {r['signal']} Signal",
            "fields"   : [
                {"title": "Close Price",  "value": f"${r['current_close']:,.2f}", "short": True},
                {"title": "P(UP)",        "value": f"{r['prob_up']*100:.1f}%",   "short": True},
                {"title": "P(DOWN)",      "value": f"{r['prob_down']*100:.1f}%", "short": True},
                {"title": "Confidence",   "value": f"{r['confidence']*100:.1f}%","short": True},
            ],
            "footer": "BTC Candle Predictor",
            "ts"    : int(datetime.now(timezone.utc).timestamp()),
        }]
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FULL SIGNAL DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def dispatch_signal(result: dict):
    """
    Send signal to ALL configured channels.
    Channels are silently skipped if not configured.
    """
    if not result:
        return

    tf = result.get("timeframe", "5m")

    log_to_csv(result, tf)          # always log
    send_telegram(result)           # only if TELEGRAM_* env vars set
    send_webhook(result)            # only if DISCORD/SLACK webhook set


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEDULER — runs at each candle close
# ══════════════════════════════════════════════════════════════════════════════

def _seconds_until_close(timeframe: str) -> float:
    """Seconds remaining until current candle closes + 5s buffer."""
    now = datetime.now(timezone.utc)
    tf_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
    mins  = tf_minutes.get(timeframe, 5)
    secs  = mins * 60
    elapsed = (now.minute % mins) * 60 + now.second
    remaining = secs - elapsed + 5   # +5s to ensure candle is closed
    return remaining


def run_live(
    timeframes: list = TIMEFRAMES,
    threshold: float = CONFIDENCE_THRESHOLD,
):
    """
    Main live loop. Sleeps until each candle close, then predicts.
    Runs both 5m and 15m in a single loop — uses the shorter timeframe
    (5m) as the base tick, and fires 15m predictions when the 15m candle closes.
    """
    log.info("=" * 55)
    log.info("  BTC Candle Predictor — LIVE MODE")
    log.info(f"  Timeframes     : {timeframes}")
    log.info(f"  Confidence thr : {threshold}")
    log.info(f"  Alert channels : CSV + Telegram + Webhook")
    log.info("=" * 55)

    # Pre-load models
    models_cache = {}
    feats_cache  = {}

    for tf in timeframes:
        try:
            from model_trainer import load_latest_models
            models_cache[tf], feats_cache[tf] = load_latest_models(tf)
        except FileNotFoundError:
            log.warning(f"No saved model for {tf} — train first (python model_trainer.py)")

    while True:
        # Always base loop on 5m
        wait = _seconds_until_close("5m")
        log.info(f"Next 5m candle close in {wait:.0f}s ...")
        time.sleep(max(wait, 1))

        now = datetime.now(timezone.utc)

        for tf in timeframes:
            # Only fire 15m prediction when the 15m candle closes
            tf_minutes = {"5m": 5, "15m": 15}[tf]
            if now.minute % tf_minutes != 0:
                continue

            log.info(f"─── {tf} candle closed at {now.strftime('%H:%M')} UTC ───")

            if tf not in models_cache:
                log.warning(f"Skipping {tf} — no model loaded")
                continue

            try:
                result = predict_next_candle(
                    timeframe    = tf,
                    models       = models_cache[tf],
                    feature_cols = feats_cache[tf],
                    threshold    = threshold,
                )
                dispatch_signal(result)
            except Exception as e:
                log.error(f"Prediction error ({tf}): {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST SIGNAL REVIEW
# ══════════════════════════════════════════════════════════════════════════════

def review_past_signals(timeframe: str = "5m", days: int = 7) -> pd.DataFrame:
    """
    Load CSV logs from the past N days, join with actual close prices,
    and compute accuracy of past signals.
    """
    from data_collector import fetch_latest

    records = []
    for i in range(days):
        from datetime import timedelta
        date_str = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d")
        path     = LOGS_DIR / f"signals_{timeframe}_{date_str}.csv"
        if path.exists():
            records.append(pd.read_csv(path))

    if not records:
        print("No log files found for review.")
        return pd.DataFrame()

    df_log = pd.concat(records).reset_index(drop=True)
    df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], utc=True)

    # Fetch recent OHLCV to compute actual outcomes
    df_price = fetch_latest(timeframe=timeframe, n=2000)

    # For each logged signal, find the actual outcome
    outcomes = []
    for _, row in df_log.iterrows():
        ts = row["timestamp"]
        # Find current candle
        try:
            curr_close = df_price.loc[ts]["close"]
            # Find next candle
            next_ts    = df_price.index[df_price.index.get_loc(ts) + 1]
            next_close = df_price.loc[next_ts]["close"]
            actual     = "UP" if next_close > curr_close else "DOWN"
            correct    = int(actual == row["signal"]) if row["signal"] != "SKIP" else None
            outcomes.append({"actual": actual, "correct": correct})
        except Exception:
            outcomes.append({"actual": None, "correct": None})

    df_log = pd.concat([df_log, pd.DataFrame(outcomes)], axis=1)

    # Summary stats
    taken  = df_log[df_log["signal"] != "SKIP"]
    if len(taken) > 0:
        acc = taken["correct"].mean()
        print(f"\nSignal Review ({timeframe}, last {days}d)")
        print(f"  Signals taken : {len(taken)}")
        print(f"  Skipped       : {len(df_log) - len(taken)}")
        print(f"  Accuracy      : {acc:.3f}  ({acc*100:.1f}%)")
        print(f"  UP signals    : {(taken['signal']=='UP').sum()}")
        print(f"  DOWN signals  : {(taken['signal']=='DOWN').sum()}")

    return df_log


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "once"

    if cmd == "once":
        # Run a single prediction right now (for testing)
        for tf in TIMEFRAMES:
            result = predict_next_candle(timeframe=tf)
            dispatch_signal(result)

    elif cmd == "live":
        # Continuous live loop
        run_live()

    elif cmd == "review":
        # Review accuracy of past signals
        review_past_signals("5m",  days=7)
        review_past_signals("15m", days=7)

    else:
        print("Usage: python live_predictor.py [once|live|review]")
