# models/player.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Features used by the player scoring model
PLAYER_FEATURE_COLS = [
    "rolling_goals",
    "rolling_assists",
    "rolling_points",
    "rolling_sog",
    "rolling_plus_minus",
    "rolling_hits",
    "rolling_blocked_shots",
    "rolling_toi_sec",
    "is_home",
    "position_code",
]


def get_player_models():
    """Returns models for player scoring prediction (will a player record a point?)."""
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
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


def train_player_model(name, model_cfg, X_train, y_train, X_test, y_test):
    """Train a single player model and return results."""
    model = model_cfg["model"]
    X_tr = X_train.copy()
    X_te = X_test.copy()

    scaler = None
    if model_cfg["scale"]:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=["No Point", "Scored Point"])

    return {
        "name": name,
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report,
        "preds": preds,
    }