"""
==============================================================================
  BTC CANDLE DIRECTION PREDICTOR
  Module 3: Model Trainer & Validator
  - Walk-forward time-series cross validation
  - LightGBM (primary) + XGBoost + Logistic Regression baseline
  - SHAP feature importance
  - Calibration curve
  - Confidence-filtered evaluation
==============================================================================
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

# ML
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration     import CalibratedClassifierCV, calibration_curve
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss, confusion_matrix
)
import lightgbm as lgb
import xgboost  as xgb

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_splits(
    df: pd.DataFrame,
    feature_cols: list,
    train_pct: float = 0.70,
    val_pct:   float = 0.15,
):
    """
    Chronological 70 / 15 / 15 split — NO shuffling.
    Returns X_train, X_val, X_test, y_train, y_val, y_test
    with a DatetimeIndex preserved for diagnostics.
    """
    n         = len(df)
    train_end = int(n * train_pct)
    val_end   = int(n * (train_pct + val_pct))

    X = df[feature_cols]
    y = df["target"]

    X_train, y_train = X.iloc[:train_end],  y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:],    y.iloc[val_end:]

    print(f"[Trainer] Split sizes → Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"          Date range  → Train: {X_train.index[0].date()} → {X_train.index[-1].date()}")
    print(f"                        Test : {X_test.index[0].date()}  → {X_test.index[-1].date()}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_logistic(X_train, y_train, X_val=None, y_val=None):
    """Baseline logistic regression with standard scaling."""
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)

    model  = LogisticRegression(C=0.05, max_iter=2000, solver="lbfgs",
                                 class_weight="balanced", random_state=42)
    model.fit(X_tr, y_train)

    # Wrap so predict_proba works on raw (unscaled) input
    return _ScaledClassifier(scaler, model)


def train_lightgbm(X_train, y_train, X_val, y_val, timeframe: str = "5m"):
    """Primary model — LightGBM with early stopping on validation set."""
    params = dict(
        n_estimators      = 2000,
        learning_rate     = 0.01,
        max_depth         = 5,
        num_leaves        = 31,
        min_child_samples = 80,      # higher = less overfit for short candles
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        reg_alpha         = 0.1,
        reg_lambda        = 0.2,
        class_weight      = "balanced",
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set   = [(X_val, y_val)],
        callbacks  = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    print(f"[Trainer] LightGBM best iteration: {model.best_iteration_}")
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Secondary model — XGBoost."""
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators          = 2000,
        learning_rate         = 0.01,
        max_depth             = 5,
        subsample             = 0.8,
        colsample_bytree      = 0.7,
        reg_alpha             = 0.1,
        reg_lambda            = 0.2,
        scale_pos_weight      = scale_pos,
        eval_metric           = "logloss",
        early_stopping_rounds = 80,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
    )
    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose  = 200,
    )
    print(f"[Trainer] XGBoost best iteration: {model.best_iteration}")
    return model


def train_ensemble(X_train, y_train, X_val, y_val, timeframe: str = "5m"):
    """Train all three models and return a soft-voting ensemble."""
    print("\n[Trainer] ── Training Logistic Regression (baseline) ──")
    lr   = train_logistic(X_train, y_train)

    print("\n[Trainer] ── Training LightGBM ──")
    lgbm = train_lightgbm(X_train, y_train, X_val, y_val, timeframe)

    print("\n[Trainer] ── Training XGBoost ──")
    xgbm = train_xgboost(X_train, y_train, X_val, y_val)

    return {"logistic": lr, "lightgbm": lgbm, "xgboost": xgbm}


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X, y, name: str = "Model", threshold: float = 0.5) -> dict:
    """Full evaluation suite including confidence-filtered metrics."""
    proba  = model.predict_proba(X)[:, 1]
    pred   = (proba >= threshold).astype(int)

    metrics = {
        "name"        : name,
        "n_samples"   : len(y),
        "accuracy"    : round(accuracy_score(y, pred), 4),
        "precision"   : round(precision_score(y, pred, zero_division=0), 4),
        "recall"      : round(recall_score(y, pred, zero_division=0), 4),
        "f1"          : round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc"     : round(roc_auc_score(y, proba), 4),
        "brier_score" : round(brier_score_loss(y, proba), 4),
    }

    print(f"\n{'═'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    for k, v in metrics.items():
        if k not in ("name", "n_samples"):
            print(f"  {k:<15}: {v}")

    # Confidence-filtered evaluation
    for thr in [0.55, 0.60, 0.65]:
        mask    = (proba > thr) | (proba < (1 - thr))
        n_taken = mask.sum()
        if n_taken < 20:
            continue
        acc_f   = accuracy_score(y[mask], pred[mask])
        pct     = n_taken / len(y) * 100
        print(f"  conf>{thr}  → acc {acc_f:.4f}  ({n_taken} signals, {pct:.1f}% of candles)")

    metrics["proba"] = proba
    return metrics


def walk_forward_cv(df: pd.DataFrame, feature_cols: list, n_splits: int = 5) -> pd.DataFrame:
    """
    Walk-forward cross-validation with gap=1 between train/val to prevent leakage.
    Returns per-fold metrics as DataFrame.
    """
    print(f"\n[Trainer] Walk-forward CV ({n_splits} folds) ...")
    tscv    = TimeSeriesSplit(n_splits=n_splits, gap=1)
    X       = df[feature_cols]
    y       = df["target"]
    results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_v = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_v = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.02, max_depth=5,
            min_child_samples=50, random_state=42, verbose=-1
        )
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_v)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        results.append({
            "fold"       : fold + 1,
            "train_size" : len(X_tr),
            "val_size"   : len(X_v),
            "accuracy"   : round(accuracy_score(y_v, pred),       4),
            "roc_auc"    : round(roc_auc_score(y_v, proba),       4),
            "brier"      : round(brier_score_loss(y_v, proba),    4),
            "val_start"  : X_v.index[0].date(),
            "val_end"    : X_v.index[-1].date(),
        })
        print(f"  Fold {fold+1}: AUC={results[-1]['roc_auc']:.4f}  Acc={results[-1]['accuracy']:.4f}  "
              f"({results[-1]['val_start']} → {results[-1]['val_end']})")

    cv_df = pd.DataFrame(results)
    print(f"\n  Mean AUC : {cv_df['roc_auc'].mean():.4f} ± {cv_df['roc_auc'].std():.4f}")
    print(f"  Mean Acc : {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
    return cv_df


def ensemble_predict_proba(models: dict, X: pd.DataFrame) -> np.ndarray:
    """Soft-vote ensemble: average P(UP) across all models."""
    probas = np.stack([m.predict_proba(X)[:, 1] for m in models.values()])
    return probas.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE (SHAP)
# ══════════════════════════════════════════════════════════════════════════════

def get_top_features(model, feature_cols: list, top_n: int = 30) -> list:
    """
    Return top N features by LightGBM gain importance.
    Falls back to split importance if gain not available.
    """
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=False)
        top = imp[imp > 0].head(top_n).index.tolist()
        print(f"[Trainer] Top {len(top)} features selected (of {len(feature_cols)})")
        return top
    return feature_cols


def shap_analysis(model, X_sample: pd.DataFrame):
    """Print SHAP summary — requires shap package."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        vals      = explainer.shap_values(X_sample)
        if isinstance(vals, list):
            vals = vals[1]   # class 1 values
        imp = pd.DataFrame({
            "feature"      : X_sample.columns,
            "mean_abs_shap": np.abs(vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        print("\n[Trainer] Top 15 features by SHAP:")
        print(imp.head(15).to_string(index=False))
        return imp
    except ImportError:
        print("[Trainer] Install shap for importance analysis: pip install shap")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_models(models: dict, feature_cols: list, timeframe: str = "5m", metrics: dict = None):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    for name, model in models.items():
        path = MODELS_DIR / f"{timeframe}_{name}_{ts}.pkl"
        joblib.dump(model, path)
        print(f"[Trainer] Saved {name} → {path}")

    # Save feature list
    feat_path = MODELS_DIR / f"{timeframe}_features_{ts}.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)

    # Save metrics summary
    if metrics:
        meta = {
            "timeframe"   : timeframe,
            "timestamp"   : ts,
            "feature_cols": feature_cols,
            "metrics"     : {k: {mk: mv for mk, mv in v.items() if mk != "proba"}
                             for k, v in metrics.items()},
        }
        meta_path = MODELS_DIR / f"{timeframe}_meta_{ts}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    return ts


def load_latest_models(timeframe: str = "5m"):
    """Load the most recent saved models and feature list."""
    # Find latest timestamp
    pkls  = sorted(MODELS_DIR.glob(f"{timeframe}_lightgbm_*.pkl"), reverse=True)
    if not pkls:
        raise FileNotFoundError(f"No saved models found for {timeframe}")

    ts    = pkls[0].stem.split("_")[-2] + "_" + pkls[0].stem.split("_")[-1]
    models = {}
    for name in ["logistic", "lightgbm", "xgboost"]:
        p = MODELS_DIR / f"{timeframe}_{name}_{ts}.pkl"
        if p.exists():
            models[name] = joblib.load(p)
            print(f"[Trainer] Loaded {name} ({timeframe})")

    feat_path = MODELS_DIR / f"{timeframe}_features_{ts}.json"
    feature_cols = json.load(open(feat_path)) if feat_path.exists() else None

    return models, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: SCALED CLASSIFIER WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class _ScaledClassifier:
    """Wraps a scaler + classifier so predict_proba accepts raw features."""
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model  = model

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    @property
    def feature_importances_(self):
        return getattr(self.model, "coef_", None)


# ══════════════════════════════════════════════════════════════════════════════
#  FULL TRAINING PIPELINE (convenience wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def full_training_pipeline(df: pd.DataFrame, feature_cols: list, timeframe: str = "5m"):
    """
    End-to-end:  split → train all models → evaluate → walk-forward CV → save
    Returns trained models dict and evaluation metrics.
    """
    print(f"\n{'═'*60}")
    print(f"  BTC Candle Predictor — Training ({timeframe})")
    print(f"{'═'*60}")

    # 1. Split
    X_tr, X_val, X_te, y_tr, y_val, y_te = prepare_splits(df, feature_cols)

    # 2. Train
    models = train_ensemble(X_tr, y_tr, X_val, y_val, timeframe)

    # 3. Evaluate on test set
    print(f"\n[Trainer] ── Test Set Evaluation ──")
    ensemble_proba = ensemble_predict_proba(models, X_te)

    # Wrap ensemble as a pseudo-model for evaluate()
    class EnsembleMock:
        def predict_proba(self, X):
            p = ensemble_predict_proba(models, X)
            return np.column_stack([1 - p, p])
    ens = EnsembleMock()

    metrics = {}
    for name, model in list(models.items()) + [("ensemble", ens)]:
        metrics[name] = evaluate(model, X_te, y_te, name=name.upper())

    # 4. Walk-forward CV
    cv_results = walk_forward_cv(df, feature_cols)

    # 5. SHAP importance on LightGBM
    sample = X_tr.sample(min(500, len(X_tr)), random_state=42)
    shap_analysis(models["lightgbm"], sample)

    # 6. Save
    ts = save_models(models, feature_cols, timeframe, metrics)

    print(f"\n[Trainer] ✓ Training complete. Timestamp: {ts}")
    return models, metrics, cv_results


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_collector   import load_data
    from feature_engineer import full_pipeline, get_feature_columns

    for tf in ["5m", "15m"]:
        print(f"\n{'#'*60}")
        print(f"#  Training for {tf}")
        print(f"{'#'*60}")

        df_5m  = load_data("5m")
        df_15m = load_data("15m")
        df     = full_pipeline(df_5m, df_15m)
        fc     = get_feature_columns(df)

        # Use appropriate target timeframe
        if tf == "15m":
            df_15m_feat = __import__("feature_engineer").build_features(df_15m)
            df_15m_feat = __import__("feature_engineer").add_target(df_15m_feat)
            df_15m_feat = df_15m_feat.dropna()
            fc_15m      = get_feature_columns(df_15m_feat)
            full_training_pipeline(df_15m_feat, fc_15m, timeframe="15m")
        else:
            full_training_pipeline(df, fc, timeframe="5m")
