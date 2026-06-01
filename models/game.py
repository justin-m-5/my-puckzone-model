# models/game.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

FEATURE_COLS = [
    "home_point_pctg", "home_win_pctg", "home_reg_win_pctg",
    "home_goal_diff", "home_l10_points",
    "home_goalie_sv_pctg", "home_rest_days", "home_is_b2b",
    "home_pp_pctg", "home_faceoff_pctg", "home_sog", "home_hits", "home_blocked_shots",
    "away_point_pctg", "away_win_pctg", "away_reg_win_pctg",
    "away_goal_diff", "away_l10_points",
    "away_goalie_sv_pctg", "away_rest_days", "away_is_b2b",
    "away_pp_pctg", "away_faceoff_pctg", "away_sog", "away_hits", "away_blocked_shots",
    "diff_point_pctg", "diff_goal_diff", "diff_l10_points",
    "diff_points", "diff_goalie_sv_pctg", "rest_advantage",
    "diff_pp_pctg", "diff_faceoff_pctg", "diff_sog",
    "home_home_win_pctg", "away_road_win_pctg", "diff_home_road_pctg",
    "h2h_home_win_pctg", "home_elo", "away_elo", "elo_diff"
]

def get_models():
    """Returns all models we want to compare."""
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "scale": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42),
            "scale": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42),
            "scale": False,
        },
    }

def train_model(name, model_cfg, X_train, y_train, X_test, y_test):
    """Train a single model and return results."""
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
    report = classification_report(y_test, preds, target_names=["Away Win", "Home Win"])

    return {
        "name": name,
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "report": report,
        "preds": preds,
    }