# BTC Candle Direction Predictor

Binary classification model predicting whether the next BTC/USD candle closes UP or DOWN.

**Exchange:** Coinbase | **Timeframes:** 5m + 15m | **Model:** LightGBM ensemble

---

## Quick Start

> **Exchange:** Coinbase Advanced Trade API (public endpoints — no API key required for market data)


```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download 12 months of data
python btc_predictor_all_in_one.py fetch

# 3. Train models (5m first, then 15m)
python btc_predictor_all_in_one.py train 5m
python btc_predictor_all_in_one.py train 15m

# 4. Single prediction now
python btc_predictor_all_in_one.py predict 5m

# 5. Live loop (fires at each candle close)
python btc_predictor_all_in_one.py live
```

## Modular Version

```bash
# Step-by-step using individual modules
cd modules

# 1. Download data
python data_collector.py

# 2. Train
python model_trainer.py

# 3. Live
python live_predictor.py live

# 4. Review past signal accuracy
python live_predictor.py review
```

## Jupyter Notebook

```bash
jupyter notebook notebooks/BTC_Candle_Predictor.ipynb
```

---

## Alert Configuration

Set environment variables before running:

```bash
# Telegram
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
export TELEGRAM_CHAT_ID="-1001234567890"

# Discord webhook
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

# Slack webhook
export DISCORD_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

### Telegram Setup
1. Message @BotFather → `/newbot` → copy token
2. Add bot to your channel/group
3. Get chat_id: `https://api.telegram.org/bot<TOKEN>/getUpdates`

### Discord Setup
Server Settings → Integrations → Webhooks → New Webhook → Copy URL

---

## Project Structure

```
btc_predictor/
├── btc_predictor_all_in_one.py   # Single-file version
├── requirements.txt
├── modules/
│   ├── data_collector.py          # Binance OHLCV downloader
│   ├── feature_engineer.py        # 60+ indicators + MTF injection
│   ├── model_trainer.py           # LightGBM + XGBoost + LR + validation
│   └── live_predictor.py          # Scheduler + all alert channels
├── notebooks/
│   └── BTC_Candle_Predictor.ipynb # Full interactive walkthrough
├── data/                          # Parquet candle files
├── models/                        # Saved model .pkl files
└── logs/                          # Signal CSVs + predictor.log
```

---

## Features Used (60+)

| Category        | Features |
|-----------------|----------|
| Price           | Multi-period returns, candle body/wick ratios, gap |
| EMA             | 20/50/200 distances, slopes, bull/bear stack |
| Momentum        | RSI (7/14/21), MACD, Stochastics, Williams %R, CCI, ROC |
| Volatility      | ATR%, Bollinger Bands, rolling std, vol ratio |
| Volume          | Spike ratio, OBV slope, VWAP distance |
| Structure       | Swing hi/lo distance, streak, BOS signals, candle patterns |
| Statistical     | Skew, kurtosis, autocorr, Hurst exponent |
| Time            | Cyclical hour/DOW encoding, session flags |
| Multi-timeframe | 15m features injected into 5m rows (as_of merge) |

---

## Signal Output Example

```
════════════════════════════════════════════════════════
  BTC/USD 5m  |  2024-11-15T14:25:00+00:00
────────────────────────────────────────────────────────
  Close      : $91,234.00
  P(UP)      : 67.3%   P(DOWN): 32.7%
  Confidence : 67.3%
  Signal     : 🟢  UP
════════════════════════════════════════════════════════
```

---

## Realistic Performance Expectations

| Metric       | Random | Good Model |
|--------------|--------|------------|
| Accuracy     | 50%    | 52–56%     |
| ROC-AUC      | 0.500  | 0.52–0.57  |

> Even 53% accuracy with confidence filtering (≥0.60) on ~30% of candles delivers
> a meaningful edge. Financial ML doesn't need to be "accurate" — it needs to be
> *slightly better than random, consistently*.
