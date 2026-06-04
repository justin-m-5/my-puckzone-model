# models/xg.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

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
]


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