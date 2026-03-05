"""
==============================================================================
  BTC CANDLE DIRECTION PREDICTOR — ALL-IN-ONE SCRIPT
  Exchange  : Coinbase (BTC/USD)
  Timeframes: 5m + 15m (multi-timeframe model)
  Target    : Binary UP/DOWN classification per candle

  USAGE:
    python btc_predictor_all_in_one.py fetch     # Download 12m of data
    python btc_predictor_all_in_one.py train     # Train all models
    python btc_predictor_all_in_one.py predict   # Single prediction now
    python btc_predictor_all_in_one.py live      # Continuous live loop
    python btc_predictor_all_in_one.py review    # Review past signal accuracy

  ALERT CONFIG (set as environment variables):
    TELEGRAM_BOT_TOKEN   = "123456:ABC..."
    TELEGRAM_CHAT_ID     = "-1001234567890"
    DISCORD_WEBHOOK_URL  = "https://discord.com/api/webhooks/..."
==============================================================================
"""

import os, sys, csv, json, time, logging, warnings

import sys
sys.stdout.reconfigure(encoding="utf-8")
import zoneinfo

import numpy as np
import pandas as pd
import requests
import joblib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

# ── DIRECTORIES ───────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOGS_DIR  = ROOT / "logs"
for d in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "predictor.log"),
    ],
)
log = logging.getLogger("BTC")

# ── SETTINGS ──────────────────────────────────────────────────────────────────
SYMBOL             = "BTC/USD"              # Coinbase uses USD not USDT
TIMEFRAMES         = ["5m", "15m"]
MONTHS_BACK        = 12
CONFIDENCE_THRESH  = 0.60
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT      = os.getenv("TELEGRAM_CHAT_ID",   "")
WEBHOOK_URL        = os.getenv("DISCORD_WEBHOOK_URL", "")

# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(timeframe="5m", months=12, save=True):
    import ccxt
    exchange = ccxt.coinbase({"enableRateLimit": True})
    since    = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30*months)
    since_ms = int(since.timestamp() * 1000)
    all_c    = []
    print(f"Fetching {timeframe} data from Coinbase from {since.date()} ...")
    print(f"Note: Coinbase allows 300 candles/request — may take a few minutes")

    while True:
        try:
            batch = exchange.fetch_ohlcv(SYMBOL, timeframe, since=since_ms, limit=1000)
        except ccxt.RateLimitExceeded:
            log.warning("Rate limit hit — waiting 30s")
            time.sleep(30); continue
        except Exception as e:
            log.warning(f"Fetch error: {e} — retrying in 10s")
            time.sleep(10); continue

        if not batch: break
        all_c.extend(batch)
        since_ms = batch[-1][0] + 1
        last_ts  = pd.to_datetime(batch[-1][0], unit="ms", utc=True)
        if last_ts >= pd.Timestamp.utcnow() - pd.Timedelta(timeframe): break
        time.sleep(max(exchange.rateLimit / 1000, 0.5))  # Coinbase rate limit

    df = pd.DataFrame(all_c, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index().astype(float)

    if save:
        df.to_parquet(DATA_DIR / f"btc_{timeframe}.parquet")
        print(f"Saved {len(df):,} candles → data/btc_{timeframe}.parquet")
    return df


def load_data(timeframe="5m", refresh=False):
    path = DATA_DIR / f"btc_{timeframe}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        print(f"Loaded {timeframe}: {len(df):,} candles")
        return df
    return fetch_ohlcv(timeframe)


def fetch_latest(timeframe="5m", n=300):
    import ccxt
    exchange = ccxt.coinbase({"enableRateLimit": True})

    candles = None

    for i in range(5):
        try:
            candles = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=n)
            break
        except Exception as e:
            log.warning(f"Retrying Coinbase fetch ({i+1}/5)...")
            time.sleep(3)

    if candles is None:
        log.error("Failed to fetch candles from Coinbase")
        return None

    df = pd.DataFrame(
        candles,
        columns=["timestamp","open","high","low","close","volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    return df.iloc[:-1].astype(float)  # drop unclosed candle


# ══════════════════════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df):
    from ta import momentum as tm, trend as tt, volatility as tv, volume as tvol
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    ret = c.pct_change()

    # Price
    for n in [1,2,3,5,8,13]:
        df[f"ret_{n}"] = c.pct_change(n)
    df["body_ratio"] = (c - df["open"]).abs() / (h - l + 1e-9)
    df["upper_wick"] = (h - pd.concat([df["open"],c],axis=1).max(axis=1)) / (h-l+1e-9)
    df["lower_wick"] = (pd.concat([df["open"],c],axis=1).min(axis=1) - l) / (h-l+1e-9)
    df["hl_pct"]     = (h - l) / c
    df["gap"]        = (df["open"] - c.shift(1)) / (c.shift(1) + 1e-9)

    # EMAs
    for span in [20, 50, 200]:
        df[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()
    df["dist_ema20"]   = (c - df["ema_20"])  / df["ema_20"]
    df["dist_ema50"]   = (c - df["ema_50"])  / df["ema_50"]
    df["dist_ema200"]  = (c - df["ema_200"]) / df["ema_200"]
    df["spread_20_50"] = (df["ema_20"] - df["ema_50"])  / df["ema_50"]
    df["spread_50_200"]= (df["ema_50"] - df["ema_200"]) / df["ema_200"]
    df["slope_ema20"]  = df["ema_20"].pct_change(3)
    df["slope_ema50"]  = df["ema_50"].pct_change(3)
    df["bull_stack"]   = ((df["ema_20"]>df["ema_50"]) & (df["ema_50"]>df["ema_200"])).astype(int)
    df["above_ema20"]  = (c > df["ema_20"]).astype(int)
    df["above_ema50"]  = (c > df["ema_50"]).astype(int)

    # Momentum
    df["rsi_7"]     = tm.RSIIndicator(c, window=7).rsi()
    df["rsi_14"]    = tm.RSIIndicator(c, window=14).rsi()
    df["rsi_slope"] = df["rsi_14"].diff(3)
    macd = tt.MACD(c)
    df["macd_hist"]       = macd.macd_diff()
    df["macd_hist_slope"] = df["macd_hist"].diff(1)
    df["macd_cross"]      = ((macd.macd() > macd.macd_signal()) &
                              (macd.macd().shift(1) <= macd.macd_signal().shift(1))).astype(int)
    stoch = tm.StochasticOscillator(h, l, c)
    df["stoch_kd"] = stoch.stoch() - stoch.stoch_signal()
    df["willr"]    = tm.WilliamsRIndicator(h, l, c).williams_r()
    df["cci"]      = tt.CCIIndicator(h, l, c).cci()
    df["roc_5"]    = tm.ROCIndicator(c, window=5).roc()

    # Volatility
    df["atr_pct"]    = tv.AverageTrueRange(h, l, c).average_true_range() / c
    bb = tv.BollingerBands(c)
    df["bb_width"]   = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg()+1e-9)
    df["bb_pos"]     = (c - bb.bollinger_lband()) / (bb.bollinger_hband()-bb.bollinger_lband()+1e-9)
    df["rvol_10"]    = ret.rolling(10).std()
    df["rvol_20"]    = ret.rolling(20).std()
    df["vol_ratio"]  = df["rvol_10"] / (df["rvol_20"] + 1e-9)

    # Volume
    df["vol_ma20"]   = v.rolling(20).mean()
    df["vol_spike"]  = v / (df["vol_ma20"] + 1e-9)
    df["vol_chg3"]   = v.pct_change(3)
    obv = (np.sign(c.diff()) * v).cumsum()
    df["obv_slope"]  = obv.pct_change(5)
    vwap = (c*v).rolling(24).sum() / (v.rolling(24).sum()+1e-9)
    df["vwap_dist"]  = (c - vwap) / (vwap + 1e-9)

    # Market structure
    df["high10"]     = h.rolling(10).max()
    df["low10"]      = l.rolling(10).min()
    df["dhigh10"]    = (df["high10"] - c) / c
    df["dlow10"]     = (c - df["low10"]) / c
    df["streak_up"]  = (c > c.shift(1)).astype(int).rolling(5).sum()
    df["streak_dn"]  = (c < c.shift(1)).astype(int).rolling(5).sum()
    df["bos_up"]     = (c > df["high10"].shift(1)).astype(int)
    df["bos_dn"]     = (c < df["low10"].shift(1)).astype(int)
    df["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    df["doji"]       = ((c-df["open"]).abs()/(h-l+1e-9) < 0.1).astype(int)

    # Statistical
    df["roll_mean"]  = ret.rolling(20).mean()
    df["roll_var"]   = ret.rolling(20).var()
    df["roll_skew"]  = ret.rolling(20).apply(lambda x: skew(x), raw=True)
    df["roll_kurt"]  = ret.rolling(20).apply(lambda x: kurtosis(x), raw=True)
    df["autocorr5"]  = ret.rolling(30).apply(
        lambda x: pd.Series(x).autocorr(lag=5) if len(x)>5 else 0.0, raw=False)

    # Time
    df["hour_sin"]   = np.sin(2*np.pi*df.index.hour/24)
    df["hour_cos"]   = np.cos(2*np.pi*df.index.hour/24)
    df["dow_sin"]    = np.sin(2*np.pi*df.index.dayofweek/7)
    df["dow_cos"]    = np.cos(2*np.pi*df.index.dayofweek/7)
    df["is_weekend"] = (df.index.dayofweek>=5).astype(int)
    df["london"]     = df.index.hour.isin(range(7,16)).astype(int)
    df["ny"]         = df.index.hour.isin(range(13,21)).astype(int)

    return df


def inject_mtf(df_5m, df_15m):
    """Forward-fill 15m features into 5m rows (no lookahead)."""
    cols = ["rsi_14","macd_hist","dist_ema20","dist_ema50","bull_stack",
            "bb_pos","atr_pct","vol_spike","streak_up","streak_dn","slope_ema20"]
    cols = [c for c in cols if c in df_15m.columns]
    renamed = df_15m[cols].rename(columns={c: f"h1_{c}" for c in cols})
    return pd.merge_asof(df_5m.sort_index(), renamed.sort_index(),
                         left_index=True, right_index=True, direction="backward")


def add_target(df):
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
    return df


RAW = {"open","high","low","close","volume","target","future_return",
       "ema_20","ema_50","ema_200","vol_ma20","high10","low10"}

def feature_cols(df):
    return [c for c in df.columns if c not in RAW]


def prepare_dataset(df_5m, df_15m=None):
    df = build_features(df_5m)
    if df_15m is not None:
        df_15m_feat = build_features(df_15m)
        df = inject_mtf(df, df_15m_feat)
    df = add_target(df)
    n  = len(df)
    df = df.dropna()
    print(f"Dataset: {len(df):,} rows  (dropped {n-len(df)} NaN rows)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  3. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(timeframe="5m"):
    import lightgbm as lgb, xgboost as xgb
    from sklearn.linear_model    import LogisticRegression
    from sklearn.preprocessing   import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics         import roc_auc_score, accuracy_score, brier_score_loss

    df_5m  = load_data("5m")
    df_15m = load_data("15m")
    df     = prepare_dataset(df_5m, df_15m if timeframe=="5m" else None)
    fc     = feature_cols(df)

    if timeframe == "15m":
        df_15m_feat = build_features(df_15m)
        df_15m_feat = add_target(df_15m_feat)
        df_15m_feat = inject_mtf(df_15m_feat, build_features(df_5m)
                                  .rename(columns=lambda c: f"lo_{c}"))
        df = df_15m_feat.dropna()
        fc = feature_cols(df)

    X, y   = df[fc], df["target"]
    n      = len(X)
    n_tr   = int(n * 0.70)
    n_val  = int(n * 0.85)

    X_tr, y_tr   = X.iloc[:n_tr],      y.iloc[:n_tr]
    X_val, y_val = X.iloc[n_tr:n_val], y.iloc[n_tr:n_val]
    X_te, y_te   = X.iloc[n_val:],     y.iloc[n_val:]

    print(f"\nTrain:{len(X_tr):,}  Val:{len(X_val):,}  Test:{len(X_te):,}")

    # LightGBM
    lgbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.01,
                                max_depth=5, num_leaves=31,
                                min_child_samples=80, subsample=0.8,
                                colsample_bytree=0.7, reg_alpha=0.1,
                                reg_lambda=0.2, random_state=42,
                                verbose=-1, n_jobs=-1)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val,y_val)],
             callbacks=[lgb.early_stopping(80,verbose=False), lgb.log_evaluation(200)])

    # XGBoost
    xgbm = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.01,
                               max_depth=5, subsample=0.8,
                               colsample_bytree=0.7, reg_alpha=0.1,
                               early_stopping_rounds=80, eval_metric="logloss",
                               random_state=42, verbosity=0, n_jobs=-1)
    xgbm.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=200)

    # Logistic Regression
    sc = StandardScaler()
    lr = LogisticRegression(C=0.05, max_iter=2000, random_state=42)
    lr.fit(sc.fit_transform(X_tr), y_tr)

    # Evaluate
    models = {"lgbm": lgbm, "xgb": xgbm}
    print("\n── Test Set Evaluation ──")
    for name, m in models.items():
        p = m.predict_proba(X_te)[:,1]
        print(f"  {name}: AUC={roc_auc_score(y_te,p):.4f}  "
              f"Acc={accuracy_score(y_te,(p>0.5).astype(int)):.4f}  "
              f"Brier={brier_score_loss(y_te,p):.4f}")
        for thr in [0.60, 0.65]:
            mask = (p>thr)|(p<1-thr)
            if mask.sum()>10:
                print(f"    conf>{thr}: Acc={accuracy_score(y_te[mask],(p[mask]>0.5).astype(int)):.4f} "
                      f"({mask.sum()} signals)")

    # Walk-forward CV
    print("\n── Walk-Forward CV (5 folds) ──")
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    for i,(tr_i,va_i) in enumerate(tscv.split(X)):
        m2 = lgb.LGBMClassifier(n_estimators=500,learning_rate=0.02,max_depth=5,
                                  random_state=42,verbose=-1)
        m2.fit(X.iloc[tr_i], y.iloc[tr_i])
        p2 = m2.predict_proba(X.iloc[va_i])[:,1]
        print(f"  Fold {i+1}: AUC={roc_auc_score(y.iloc[va_i],p2):.4f}  "
              f"Acc={accuracy_score(y.iloc[va_i],(p2>0.5).astype(int)):.4f}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    for name, m in [("lgbm",lgbm),("xgb",xgbm),("lr_scaler",sc),("lr",lr)]:
        joblib.dump(m, MODEL_DIR / f"{timeframe}_{name}_{ts}.pkl")
    json.dump(fc, open(MODEL_DIR / f"{timeframe}_features_{ts}.json","w"))

    print(f"\n✓ Models saved  (timestamp: {ts})")
    return models, fc, ts


# ══════════════════════════════════════════════════════════════════════════════
#  4. LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def load_models(timeframe="5m"):
    pkls = sorted(MODEL_DIR.glob(f"{timeframe}_lgbm_*.pkl"), reverse=True)
    if not pkls:
        raise FileNotFoundError(f"No saved model for {timeframe}. Run: python ... train")
    ts   = "_".join(pkls[0].stem.split("_")[-2:])
    lgbm = joblib.load(MODEL_DIR / f"{timeframe}_lgbm_{ts}.pkl")
    xgbm = joblib.load(MODEL_DIR / f"{timeframe}_xgb_{ts}.pkl")
    fc   = json.load(open(MODEL_DIR / f"{timeframe}_features_{ts}.json"))
    return {"lgbm": lgbm, "xgb": xgbm}, fc


def predict_now(timeframe="5m", models=None, fc=None, threshold=CONFIDENCE_THRESH, of_features=None):
    if models is None:
        models, fc = load_models(timeframe)

    df = fetch_latest(timeframe, n=300)
    if timeframe == "5m":
        df15 = fetch_latest("15m", n=100)
        feat = inject_mtf(build_features(df), build_features(df15))
    else:
        feat = build_features(df)
    feat = feat.dropna()

    avail = [c for c in fc if c in feat.columns]
    row   = feat[avail].iloc[[-1]]

    probas = [m.predict_proba(row)[0][1] for m in models.values()]
    p_up   = float(np.mean(probas))
    p_dn   = 1 - p_up
    
    # --- AI MODEL FILTER ---
    if of_features:
        try:
            from model_manager import predict_ml
            
            rsi_val = feat["rsi_14"].iloc[-1] if "rsi_14" in feat.columns else 50
            vol_val = df.iloc[-1]["volume"]
            buy_val = of_features.get("buy_volume", 0)
            sell_val = of_features.get("sell_volume", 0)
            imb_val = of_features.get("imbalance", 0)
            pres_val = of_features.get("pressure", 0)
            atr_val = feat["atr_pct"].iloc[-1] if "atr_pct" in feat.columns else 0
            
            X_new = [rsi_val, vol_val, buy_val, sell_val, imb_val, pres_val, atr_val]
            
            ml_pred = predict_ml(timeframe, X_new)
            
            if ml_pred is not None:
                # Increase probability in predicted direction
                if ml_pred == "UP":
                    p_up += 0.05
                    p_dn -= 0.05
                elif ml_pred == "DOWN":
                    p_dn += 0.05
                    p_up -= 0.05
                    
                p_up = max(min(p_up, 1.0), 0.0)
                p_dn = max(min(p_dn, 1.0), 0.0)
                
        except Exception as e:
            print("[AI FILTER ERROR]", e)

    conf   = max(p_up, p_dn)

    signal = "UP" if p_up > threshold else "DOWN" if p_dn > threshold else "SKIP"
    emoji  = "🟢" if signal=="UP" else "🔴" if signal=="DOWN" else "⏸️"

    result = {
        "timestamp": feat.index[-1].isoformat(),
        "timeframe": timeframe,
        "current_close": round(df.iloc[-1]["close"], 2),
        "prob_up": round(p_up,4), "prob_down": round(p_dn,4),
        "confidence": round(conf,4), "signal": signal, "emoji": emoji,
        "rsi": round(feat["rsi_14"].iloc[-1], 2) if "rsi_14" in feat.columns else 0,
        "volume": df.iloc[-1]["volume"],
        "volatility": round(feat["atr_pct"].iloc[-1], 4) if "atr_pct" in feat.columns else 0
    }

    # Console print
    print(f"\n{'═'*52}")
    print(f"  BTC/USD {timeframe}  |  {result['timestamp']}")
    print(f"{'─'*52}")
    print(f"  Close      : ${result['current_close']:,.2f}")
    print(f"  P(UP)      : {p_up*100:.1f}%   P(DOWN): {p_dn*100:.1f}%")
    print(f"  Confidence : {conf*100:.1f}%")
    print(f"  Signal     : {emoji}  {signal}")
    print(f"{'═'*52}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  5. ALERTING
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(r):
    path   = LOGS_DIR / f"signals_{r['timeframe']}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in r if k!="emoji"])
        if is_new: w.writeheader()
        w.writerow({k:v for k,v in r.items() if k!="emoji"})
    log.info(f"Logged -> {path.name}")


def send_telegram(r):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT: return
    
    try:
        dt_utc = datetime.fromisoformat(r['timestamp'])
        tz_ist = zoneinfo.ZoneInfo("Asia/Kolkata")
        dt_ist = dt_utc.astimezone(tz_ist)
        tf_mins = 5 if r['timeframe'] == "5m" else 15
        start_dt = dt_ist + timedelta(minutes=tf_mins)
        end_dt = start_dt + timedelta(minutes=tf_mins)
        t1 = start_dt.strftime("%I:%M").lstrip("0")
        t2 = end_dt.strftime("%I:%M %p").lower().lstrip("0")
        time_ist = f"{t1}-{t2}"
    except Exception:
        time_ist = r['timestamp']

    tf_str = "5 MIN" if r['timeframe'] == "5m" else "15 MIN"
    dir_str = "🟢 Direction: LONG (UP)" if r['signal'] == "UP" else "🔴 Direction: SHORT (DOWN)"

    msg = (f"🚨 BTC SIGNAL ( {tf_str} )\n\n"
           f"{dir_str}\n\n"
           f"\U0001F4B0 Price: ${r['current_close']:,.2f}\n\n"
           f"📊 Probabilities\n"
           f"⬆️ UP: {r['prob_up']*100:.1f}%\n"
           f"⬇️ DOWN: {r['prob_down']*100:.1f}%\n\n"
           f"💪 Confidence: {r['confidence']*100:.1f}%\n\n"
           f"⏱ Time: {time_ist}")

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
        if resp.status_code == 200: log.info("Telegram OK")
        else: log.warning(f"Telegram failed: {resp.status_code}")
    except Exception as e:
        log.error(f"Telegram error: {e}")


def send_webhook(r):
    if not WEBHOOK_URL or r["signal"] == "SKIP": return
    is_discord = "discord.com" in WEBHOOK_URL
    color = 0x00C851 if r["signal"]=="UP" else 0xFF4444

    if is_discord:
        payload = {"embeds":[{
            "title": f"BTC/USD {r['timeframe']} {r['emoji']} {r['signal']}",
            "color": color,
            "fields":[
                {"name":"Close",       "value":f"${r['current_close']:,.2f}", "inline":True},
                {"name":"P(UP)",       "value":f"{r['prob_up']*100:.1f}%",   "inline":True},
                {"name":"P(DOWN)",     "value":f"{r['prob_down']*100:.1f}%", "inline":True},
                {"name":"Confidence",  "value":f"{r['confidence']*100:.1f}%","inline":True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]}
    else:
        payload = {"attachments":[{
            "color": "good" if r["signal"]=="UP" else "danger",
            "title": f"BTC/USD {r['timeframe']} {r['emoji']} {r['signal']}",
            "fields":[
                {"title":"Close","value":f"${r['current_close']:,.2f}","short":True},
                {"title":"Confidence","value":f"{r['confidence']*100:.1f}%","short":True},
            ]
        }]}
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code in (200,204): log.info("Webhook OK")
        else: log.warning(f"Webhook failed: {resp.status_code}")
    except Exception as e:
        log.error(f"Webhook error: {e}")


def dispatch(r):
    if not r: return
    save_csv(r)
    send_webhook(r)


# ══════════════════════════════════════════════════════════════════════════════
#  6. LIVE SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

def run_live():
    log.info("Starting live loop — predicting at each candle close ...")
    cache = {}
    for tf in TIMEFRAMES:
        try:
            cache[tf] = load_models(tf)
        except FileNotFoundError:
            log.warning(f"No model for {tf} — run 'train' first")

    while True:
        now     = datetime.now(timezone.utc)
        # Wait until next 5m close
        elapsed = (now.minute % 5)*60 + now.second
        wait    = max(5*60 - elapsed + 5, 1)
        log.info(f"Next 5m close in {wait:.0f}s ...")
        time.sleep(wait)

        now = datetime.now(timezone.utc)
        for tf in TIMEFRAMES:
            tf_min = {"5m":5,"15m":15}[tf]
            if now.minute % tf_min != 0: continue
            if tf not in cache: continue
            try:
                m, fc = cache[tf]
                r = predict_now(tf, m, fc)
                dispatch(r)
            except Exception as e:
                log.error(f"Prediction error ({tf}): {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    tf  = sys.argv[2] if len(sys.argv) > 2 else "5m"

    if cmd == "fetch":
        for t in TIMEFRAMES:
            fetch_ohlcv(t, months=MONTHS_BACK)

    elif cmd == "train":
        train(tf)

    elif cmd == "predict":
        r = predict_now(tf)
        dispatch(r)

    elif cmd == "live":
        run_live()

    else:
        print(__doc__)
