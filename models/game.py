# models/game.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

FEATURE_COLS = [
    "home_point_pctg", "home_win_pctg", "home_reg_win_pctg",
    "home_goal_diff", "home_l10_points",
    "home_goalie_sv_pctg", "home_goalie_gsax", "home_rest_days", "home_is_b2b",
    "home_pp_pctg", "home_faceoff_pctg", "home_sog", "home_hits", "home_blocked_shots",
    "away_point_pctg", "away_win_pctg", "away_reg_win_pctg",
    "away_goal_diff", "away_l10_points",
    "away_goalie_sv_pctg", "away_goalie_gsax", "away_rest_days", "away_is_b2b",
    "away_pp_pctg", "away_faceoff_pctg", "away_sog", "away_hits", "away_blocked_shots",
    "diff_point_pctg", "diff_goal_diff", "diff_l10_points",
    "diff_points", "diff_goalie_sv_pctg", "diff_goalie_gsax", "rest_advantage",
    "diff_pp_pctg", "diff_faceoff_pctg", "diff_sog",
    # --- advanced shot-share metrics (rolling, from features/advanced.py) ---
    "diff_cf_pct",     # Corsi: shot-attempt share
    "diff_xgf_pct",    # Expected-goals share
    "diff_hdcf_pct",   # High-danger chance share
    "diff_cf_pct_5v5",   # 5v5 Corsi share
    "diff_xgf_pct_5v5",  # 5v5 Expected-goals share
    "diff_hdcf_pct_5v5", # 5v5 High-danger chance share
    "home_team_off_strength", "away_team_off_strength", "diff_team_off_strength",
    "home_team_def_strength", "away_team_def_strength", "diff_team_def_strength",
    "home_team_schedule_strength", "away_team_schedule_strength", "diff_team_schedule_strength",
    "home_team_split_strength", "away_team_split_strength", "diff_team_split_strength",
    "home_team_form_blend", "away_team_form_blend", "diff_team_form_blend",
    "home_goalie_talent_strength", "away_goalie_talent_strength", "diff_goalie_talent_strength",
    "home_goalie_workload", "away_goalie_workload", "diff_goalie_workload",
    "home_goalie_fatigue", "away_goalie_fatigue", "diff_goalie_fatigue",
    "home_goalie_team_adj_strength", "away_goalie_team_adj_strength", "diff_goalie_team_adj_strength",
    "home_lineup_availability", "away_lineup_availability", "diff_lineup_availability",
    "home_top_skater_impact", "away_top_skater_impact", "diff_top_skater_impact",
    "home_deployment_concentration", "away_deployment_concentration", "diff_deployment_concentration",
    "home_home_win_pctg", "away_road_win_pctg", "diff_home_road_pctg",
    "h2h_home_win_pctg", "home_elo", "away_elo", "elo_diff"
]

# A leaner, decorrelated subset for experimentation. Prefers matchup diffs over
# raw home/away pairs, drops redundant Corsi (xGF/HDCF dominate), and removes the
# near-zero rest / back-to-back / last-10 tail. Compare against the full set with
# scripts.train.compare; nothing in production reads this unless you wire it in.
FEATURE_COLS_LEAN = [
    "elo_diff",
    "diff_goal_diff",
    "diff_point_pctg",
    "diff_points",
    "diff_home_road_pctg",
    "h2h_home_win_pctg",
    "diff_goalie_sv_pctg",
    "diff_goalie_gsax",
    "diff_pp_pctg",
    "diff_faceoff_pctg",
    "diff_sog",
    "diff_xgf_pct",
    "diff_hdcf_pct",
    "diff_xgf_pct_5v5",
    "diff_hdcf_pct_5v5",
]
assert set(FEATURE_COLS_LEAN).issubset(set(FEATURE_COLS))

# Per-column neutral fill values for missing features. 0.5 is correct only for
# share/percentage features (Corsi%, xGF%, win%, etc.), so list every NON-0.5
# column explicitly here. Anything not listed falls back to 0.5.
NEUTRAL_FILLS = {
    # goalie rolling GSAx: league-average GSAx is ~0
    "home_goalie_gsax": 0.0,
    "away_goalie_gsax": 0.0,
    "diff_goalie_gsax": 0.0,
    # goalie rolling save%: league-average sv% is ~0.90
    "home_goalie_sv_pctg": 0.90,
    "away_goalie_sv_pctg": 0.90,
    "diff_goalie_sv_pctg": 0.0,
    # rest days: a normal rest is ~2 days; b2b flags default to 0
    "home_rest_days": 2.0,
    "away_rest_days": 2.0,
    "rest_advantage": 0.0,
    "home_is_b2b": 0.0,
    "away_is_b2b": 0.0,
    # raw team rolling counts/rates: fall back to their rough league means
    "home_sog": 30.0, "away_sog": 30.0, "diff_sog": 0.0,
    "home_hits": 20.0, "away_hits": 20.0,
    "home_blocked_shots": 14.0, "away_blocked_shots": 14.0,
    "home_pp_pctg": 0.20, "away_pp_pctg": 0.20, "diff_pp_pctg": 0.0,
    "home_faceoff_pctg": 0.50, "away_faceoff_pctg": 0.50, "diff_faceoff_pctg": 0.0,
    # goal-diff style integer features are centered at 0
    "home_goal_diff": 0.0, "away_goal_diff": 0.0, "diff_goal_diff": 0.0,
    "diff_points": 0.0, "diff_l10_points": 0.0,
    "home_l10_points": 0.0, "away_l10_points": 0.0,
    # elo
    "home_elo": 1500.0, "away_elo": 1500.0, "elo_diff": 0.0,
    # centered strength / lineup ratings
    "home_team_off_strength": 0.0, "away_team_off_strength": 0.0, "diff_team_off_strength": 0.0,
    "home_team_def_strength": 0.0, "away_team_def_strength": 0.0, "diff_team_def_strength": 0.0,
    "home_team_schedule_strength": 0.0, "away_team_schedule_strength": 0.0, "diff_team_schedule_strength": 0.0,
    "home_team_split_strength": 0.0, "away_team_split_strength": 0.0, "diff_team_split_strength": 0.0,
    "home_team_form_blend": 0.0, "away_team_form_blend": 0.0, "diff_team_form_blend": 0.0,
    "home_goalie_talent_strength": 0.0, "away_goalie_talent_strength": 0.0, "diff_goalie_talent_strength": 0.0,
    "home_goalie_workload": 0.0, "away_goalie_workload": 0.0, "diff_goalie_workload": 0.0,
    "home_goalie_fatigue": 0.0, "away_goalie_fatigue": 0.0, "diff_goalie_fatigue": 0.0,
    "home_goalie_team_adj_strength": 0.0, "away_goalie_team_adj_strength": 0.0, "diff_goalie_team_adj_strength": 0.0,
    "home_lineup_availability": 0.0, "away_lineup_availability": 0.0, "diff_lineup_availability": 0.0,
    "home_top_skater_impact": 0.0, "away_top_skater_impact": 0.0, "diff_top_skater_impact": 0.0,
    "home_deployment_concentration": 0.0, "away_deployment_concentration": 0.0, "diff_deployment_concentration": 0.0,
}


def fill_features(X: pd.DataFrame) -> pd.DataFrame:
    """Neutral-fill a feature DataFrame column-by-column.

    Uses NEUTRAL_FILLS for known columns and 0.5 for anything else (share/pct
    features and any future column). Returns a new filled DataFrame; does not
    mutate the input. This MUST be used identically in training and prediction
    so the model sees the same imputation at fit and inference time.
    """
    fills = {col: NEUTRAL_FILLS.get(col, 0.5) for col in X.columns}
    return X.fillna(value=fills)

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