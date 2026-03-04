"""
==============================================================================
  BTC CANDLE DIRECTION PREDICTOR
  Module 2: Feature Engineering
  Builds 60+ technical + statistical features from OHLCV data
  Supports multi-timeframe feature injection (5m features into 15m rows)
==============================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings("ignore")

# ── FEATURE GROUPS (toggle on/off for ablation studies) ───────────────────────
FEATURE_CONFIG = {
    "price":       True,
    "ema":         True,
    "momentum":    True,
    "volatility":  True,
    "volume":      True,
    "structure":   True,
    "statistical": True,
    "time":        True,
    "mtf":         True,   # multi-timeframe (5m features injected into 15m)
}

# Raw OHLCV columns — NEVER include these as model features
RAW_COLS = ["open", "high", "low", "close", "volume"]


# ══════════════════════════════════════════════════════════════════════════════
#  PRIMARY FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, cfg: dict = FEATURE_CONFIG) -> pd.DataFrame:
    """
    Main entry point. Takes raw OHLCV DataFrame, returns feature-rich DataFrame.
    All features are normalized/ratio-based — no raw price levels included.
    """
    df = df.copy()
    ret = df["close"].pct_change()

    if cfg.get("price"):       df = _add_price_features(df)
    if cfg.get("ema"):         df = _add_ema_features(df)
    if cfg.get("momentum"):    df = _add_momentum_features(df)
    if cfg.get("volatility"):  df = _add_volatility_features(df)
    if cfg.get("volume"):      df = _add_volume_features(df)
    if cfg.get("structure"):   df = _add_structure_features(df)
    if cfg.get("statistical"): df = _add_statistical_features(df, ret)
    if cfg.get("time"):        df = _add_time_features(df)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns list of feature columns (excludes raw OHLCV and target variables).
    Call this AFTER build_features() to get the exact list to feed the model.
    """
    exclude = set(RAW_COLS + ["target", "future_return",
                               # Intermediate columns used for ratios
                               "ema_20", "ema_50", "ema_200",
                               "vol_ma_20", "vol_ma_50",
                               "recent_high_10", "recent_low_10",
                               "obv", "bb_upper", "bb_lower", "bb_mid"])
    return [c for c in df.columns if c not in exclude]


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE GROUPS
# ══════════════════════════════════════════════════════════════════════════════

def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized price-based momentum and candle structure."""
    c = df["close"]

    # Returns over multiple lookbacks
    for n in [1, 2, 3, 5, 8, 13]:
        df[f"ret_{n}"] = c.pct_change(n)

    # Candle body & wick ratios
    df["body_ratio"]   = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-9)
    df["upper_wick"]   = (df["high"]  - df[["open","close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["lower_wick"]   = (df[["open","close"]].min(axis=1) - df["low"])   / (df["high"] - df["low"] + 1e-9)
    df["candle_dir"]   = np.sign(df["close"] - df["open"])   # +1 bullish, -1 bearish

    # Intrabar range relative to price
    df["hl_pct"]       = (df["high"] - df["low"]) / df["close"]

    # Gap from previous close (overnight / session gaps)
    df["gap"]          = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    return df


def _add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    """EMA levels, distances, slopes, and alignment."""
    c = df["close"]

    df["ema_20"]  = c.ewm(span=20, adjust=False).mean()
    df["ema_50"]  = c.ewm(span=50, adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    # Normalized distance from price to each EMA
    df["dist_ema20"]   = (c - df["ema_20"])  / df["ema_20"]
    df["dist_ema50"]   = (c - df["ema_50"])  / df["ema_50"]
    df["dist_ema200"]  = (c - df["ema_200"]) / df["ema_200"]

    # EMA vs EMA spreads (trend structure)
    df["ema20_50_spread"]  = (df["ema_20"] - df["ema_50"])  / df["ema_50"]
    df["ema50_200_spread"] = (df["ema_50"] - df["ema_200"]) / df["ema_200"]

    # EMA slopes (rate of change over 3 periods)
    df["slope_ema20"]  = df["ema_20"].pct_change(3)
    df["slope_ema50"]  = df["ema_50"].pct_change(3)

    # Bullish alignment: ema20 > ema50 > ema200
    df["ema_bull_stack"] = (
        (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])
    ).astype(int)
    df["ema_bear_stack"] = (
        (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"])
    ).astype(int)

    # Price vs EMAs (above/below flags)
    df["above_ema20"]  = (c > df["ema_20"]).astype(int)
    df["above_ema50"]  = (c > df["ema_50"]).astype(int)
    df["above_ema200"] = (c > df["ema_200"]).astype(int)

    return df


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, MACD, Stochastics, Williams %R, CCI."""
    try:
        from ta import momentum as ta_mom, trend as ta_trend
    except ImportError:
        print("  ⚠  `ta` not installed. Run: pip install ta")
        return df

    c, h, l = df["close"], df["high"], df["low"]

    # RSI at multiple lengths
    df["rsi_7"]       = ta_mom.RSIIndicator(c, window=7).rsi()
    df["rsi_14"]      = ta_mom.RSIIndicator(c, window=14).rsi()
    df["rsi_21"]      = ta_mom.RSIIndicator(c, window=21).rsi()
    df["rsi_slope"]   = df["rsi_14"].diff(3)
    df["rsi_overbought"]  = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]    = (df["rsi_14"] < 30).astype(int)

    # MACD (12/26/9 default)
    macd = ta_trend.MACD(c)
    df["macd_line"]   = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["macd_hist_slope"] = df["macd_hist"].diff(1)
    df["macd_cross_up"]   = (
        (df["macd_line"] > df["macd_signal"]) &
        (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)

    # Stochastic
    stoch = ta_mom.StochasticOscillator(h, l, c, window=14)
    df["stoch_k"]     = stoch.stoch()
    df["stoch_d"]     = stoch.stoch_signal()
    df["stoch_kd_diff"] = df["stoch_k"] - df["stoch_d"]

    # Williams %R
    df["willr"]       = ta_mom.WilliamsRIndicator(h, l, c).williams_r()

    # CCI
    df["cci"]         = ta_trend.CCIIndicator(h, l, c).cci()

    # ROC (Rate of Change)
    df["roc_5"]       = ta_mom.ROCIndicator(c, window=5).roc()
    df["roc_10"]      = ta_mom.ROCIndicator(c, window=10).roc()

    return df


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """ATR, Bollinger Bands, rolling volatility."""
    try:
        from ta import volatility as ta_vol
    except ImportError:
        return df

    c, h, l = df["close"], df["high"], df["low"]

    # ATR (normalized by price)
    atr = ta_vol.AverageTrueRange(h, l, c, window=14)
    df["atr_pct"] = atr.average_true_range() / c

    # Bollinger Bands
    bb = ta_vol.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_mid"]      = bb.bollinger_mavg()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)
    df["bb_position"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["bb_squeeze"]  = (df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)).astype(int)

    # Rolling std of returns
    ret = c.pct_change()
    df["rvol_10"]  = ret.rolling(10).std()
    df["rvol_20"]  = ret.rolling(20).std()
    df["rvol_50"]  = ret.rolling(50).std()

    # Volatility ratio (current vs longer-term)
    df["vol_ratio"] = df["rvol_10"] / (df["rvol_50"] + 1e-9)

    # True Range (single period, normalized)
    df["tr_pct"]   = (
        pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1) / c
    )

    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume spikes, OBV, VWAP distance."""
    v, c = df["volume"], df["close"]

    df["vol_ma_20"]    = v.rolling(20).mean()
    df["vol_ma_50"]    = v.rolling(50).mean()
    df["vol_spike_20"] = v / (df["vol_ma_20"] + 1e-9)
    df["vol_spike_50"] = v / (df["vol_ma_50"] + 1e-9)
    df["vol_change_3"] = v.pct_change(3)
    df["vol_trend"]    = (df["vol_ma_20"] > df["vol_ma_50"]).astype(int)

    # OBV slope (direction of accumulation)
    obv = (np.sign(c.diff()) * v).cumsum()
    df["obv"]          = obv
    df["obv_slope"]    = obv.pct_change(5)
    df["obv_trend"]    = (obv > obv.ewm(span=10).mean()).astype(int)

    # Rolling VWAP (24-period) distance
    vwap = (c * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-9)
    df["vwap_dist"]    = (c - vwap) / (vwap + 1e-9)

    # Volume-price alignment (price up + vol up = bullish confirmation)
    df["vol_price_align"] = (
        np.sign(c.pct_change()) == np.sign(v.pct_change())
    ).astype(int)

    return df


def _add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market structure: swing highs/lows, HH/LL, break of structure, streaks."""
    c, h, l = df["close"], df["high"], df["low"]

    # Swing high/low over rolling windows
    for w in [10, 20]:
        df[f"recent_high_{w}"]      = h.rolling(w).max()
        df[f"recent_low_{w}"]       = l.rolling(min_periods=1, window=w).min()
        df[f"dist_swing_high_{w}"]  = (df[f"recent_high_{w}"] - c) / c
        df[f"dist_swing_low_{w}"]   = (c - df[f"recent_low_{w}"])  / c

    # Drop intermediate cols later — keep just the distance ratios as features
    df["recent_high_10"] = df["recent_high_10"]  # kept for internal use
    df["recent_low_10"]  = df["recent_low_10"]

    # Higher High / Lower Low
    df["hh"]  = (h > h.shift(1)).astype(int)
    df["ll"]  = (l < l.shift(1)).astype(int)
    df["hl"]  = (l > l.shift(1)).astype(int)   # higher low
    df["lh"]  = (h < h.shift(1)).astype(int)   # lower high

    # Consecutive candle streaks
    up_candle = (c > c.shift(1)).astype(int)
    df["streak_up"]   = up_candle.rolling(5).sum()
    df["streak_down"] = (1 - up_candle).rolling(5).sum()

    # Break of structure signals (close above/below recent high/low)
    df["bos_up"]   = (c > df["recent_high_10"].shift(1)).astype(int)
    df["bos_down"] = (c < df["recent_low_10"].shift(1)).astype(int)

    # Inside bar (current range within previous range)
    df["inside_bar"] = (
        (h < h.shift(1)) & (l > l.shift(1))
    ).astype(int)

    # Candle pattern: Doji, Hammer, Shooting Star
    body  = (c - df["open"]).abs()
    range_ = h - l + 1e-9
    df["doji"]          = (body / range_ < 0.1).astype(int)
    df["hammer"]        = (
        (c > df["open"]) & ((df["open"] - l) > 2 * body)
    ).astype(int)
    df["shooting_star"] = (
        (c < df["open"]) & ((h - df["open"]) > 2 * body)
    ).astype(int)

    return df


def _add_statistical_features(df: pd.DataFrame, ret: pd.Series) -> pd.DataFrame:
    """Rolling statistics: mean, variance, skew, kurtosis, autocorrelation, Hurst."""

    df["roll_mean_20"]  = ret.rolling(20).mean()
    df["roll_var_20"]   = ret.rolling(20).var()
    df["roll_skew_20"]  = ret.rolling(20).apply(lambda x: skew(x), raw=True)
    df["roll_kurt_20"]  = ret.rolling(20).apply(lambda x: kurtosis(x), raw=True)
    df["autocorr_5"]    = ret.rolling(30).apply(
        lambda x: pd.Series(x).autocorr(lag=5) if len(x) > 5 else 0.0, raw=False
    )

    # Hurst exponent (simplified RS method — trend=0.5+, mean-rev<0.5)
    def hurst_exp(ts):
        if len(ts) < 20 or np.std(ts) < 1e-10:
            return 0.5
        lags    = range(2, min(15, len(ts) // 2))
        rs_vals = []
        for lag in lags:
            chunks = [ts[i : i + lag] for i in range(0, len(ts) - lag, lag)]
            rs_c   = []
            for ch in chunks:
                if len(ch) < 2:
                    continue
                dev = (ch - ch.mean()).cumsum()
                s   = ch.std() + 1e-10
                rs_c.append((dev.max() - dev.min()) / s)
            if rs_c:
                rs_vals.append(np.mean(rs_c))
        if len(rs_vals) < 2:
            return 0.5
        try:
            return float(np.polyfit(np.log(list(lags)[: len(rs_vals)]), np.log(rs_vals), 1)[0])
        except Exception:
            return 0.5

    df["hurst"] = ret.rolling(60).apply(hurst_exp, raw=False)

    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encoding — avoids ordinality artifacts."""
    idx = df.index

    df["hour"]       = idx.hour
    df["dow"]        = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"]  / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"]  / 7)

    # High-activity session flags (UTC)
    df["london_session"]   = idx.hour.isin(range(7, 16)).astype(int)
    df["ny_session"]       = idx.hour.isin(range(13, 21)).astype(int)
    df["overlap_session"]  = idx.hour.isin(range(13, 16)).astype(int)
    df["asia_session"]     = idx.hour.isin(range(0, 7)).astype(int)

    # Drop raw columns used for encoding
    df.drop(columns=["hour", "dow"], inplace=True, errors="ignore")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-TIMEFRAME FEATURE INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def inject_mtf_features(
    df_target: pd.DataFrame,
    df_higher: pd.DataFrame,
    prefix: str = "mtf_",
    cols: list = None,
) -> pd.DataFrame:
    """
    Merge features from a higher timeframe into target timeframe rows.
    Uses forward-fill (as_of merge) — NEVER introduces lookahead.

    Example: inject 15m features into 5m rows.
    df_target : 5m feature DataFrame
    df_higher : 15m feature DataFrame (already processed by build_features)
    """
    if cols is None:
        cols = [
            "rsi_14", "macd_hist", "dist_ema20", "dist_ema50",
            "bb_position", "atr_pct", "vol_spike_20",
            "ema_bull_stack", "streak_up", "streak_down",
            "hurst", "slope_ema20",
        ]
        cols = [c for c in cols if c in df_higher.columns]

    # Rename columns with prefix
    higher_renamed = df_higher[cols].rename(columns={c: f"{prefix}{c}" for c in cols})

    # Merge — forward fill so each 5m row gets the most recent 15m value
    df_merged = pd.merge_asof(
        df_target.sort_index(),
        higher_renamed.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    print(f"[Features] MTF injection: added {len(cols)} {prefix} columns")
    return df_merged


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET VARIABLE
# ══════════════════════════════════════════════════════════════════════════════

def add_target(df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
    """
    Target = 1 if next candle's close > current close, else 0.

    ⚠️  CRITICAL: shift(-lookahead) uses FUTURE data.
    This column must ONLY be used as the label — never as a feature.
    Drop the last `lookahead` rows before training (NaN targets).
    """
    df = df.copy()
    df["target"]        = (df["close"].shift(-lookahead) > df["close"]).astype(float)
    df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
    # NaN on last row(s) — drop before training
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE (convenience wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def full_pipeline(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame = None,
    inject_15m: bool = True,
) -> pd.DataFrame:
    """
    Build complete feature set for 5m data, optionally injecting 15m context.
    Returns clean DataFrame ready for modelling (NaN rows dropped).
    """
    print("[Features] Building 5m features ...")
    df = build_features(df_5m)
    df = add_target(df)

    if inject_15m is not None and df_15m is not None:
        print("[Features] Building 15m features for MTF injection ...")
        df_15m_feat = build_features(df_15m)
        df = inject_mtf_features(df, df_15m_feat, prefix="h1_")

    n_before = len(df)
    df = df.dropna()
    print(f"[Features] Dropped {n_before - len(df)} rows with NaN → {len(df):,} clean rows")

    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_collector import load_data

    df_5m  = load_data("5m")
    df_15m = load_data("15m")
    df_out = full_pipeline(df_5m, df_15m)

    print(f"\nFinal dataset shape : {df_out.shape}")
    print(f"Target balance      : {df_out['target'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Sample features     : {get_feature_columns(df_out)[:10]}")
