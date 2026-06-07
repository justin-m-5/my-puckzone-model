# models/xg.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import numpy as np
import pandas as pd

# For xG, AUC / log-loss / Brier score are the right metrics — not accuracy.
# We want a well-calibrated probability, not a hard classifier.

XG_FEATURE_COLS = [
    "shot_distance",
    "shot_angle",
    "is_behind_net",
    "shot_type_code",
    "is_rebound",
    "period",
    "is_pp",
    "is_sh",
    "is_en",
    "seconds_since_last_event",
    "event_gap",
    "prev_event_same_team",
    "possession_change",
    "is_rush_proxy",
    "is_one_timer_proxy",
    "is_5v5",
    "strength_state_code",
    "score_state",
    "x_abs_norm",
    "y_abs_norm",
    "x_sq",
    "y_sq",
    "xy_interaction",
    "lane_center_low_angle",
    "lane_center_mid_angle",
    "lane_wide_high_angle",
]


def build_xg_feature_matrix(df: pd.DataFrame, feature_cols=None, fill_value=0.0) -> pd.DataFrame:
    """
    Return xG feature matrix with backward-compatible reindexing.
    Missing columns are filled with neutral values.
    """
    cols = list(feature_cols or XG_FEATURE_COLS)
    return df.reindex(columns=cols, fill_value=fill_value).fillna(fill_value)


def calibration_bins(y_true, proba, n_bins=10):
    """Return predicted-vs-actual calibration bins for reporting/saving."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    max_bin_idx = n_bins - 1
    idx = np.clip(np.digitize(proba, bins) - 1, 0, max_bin_idx)
    out = []
    for b in range(len(bins) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "low": float(bins[b]),
            "high": float(bins[b + 1]),
            "n": n,
            "pred_mean": float(proba[mask].mean()),
            "actual_rate": float(y_true[mask].mean()),
        })
    return out


def get_xg_models():
    """Returns models for expected goals (shot → goal probability)."""
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, C=1.0),
            "scale": True,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
            "scale": False,
        },
    }


def train_xg_model(name, model_cfg, X_train, y_train, X_test, y_test):
    """Train a single xG model and return probability-based metrics."""
    model = model_cfg["model"]
    X_tr = X_train.copy()
    X_te = X_test.copy()

    scaler = None
    if model_cfg["scale"]:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    model.fit(X_tr, y_train)
    proba = model.predict_proba(X_te)[:, 1]

    return {
        "name": name,
        "model": model,
        "scaler": scaler,
        "auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "proba": proba,
    }