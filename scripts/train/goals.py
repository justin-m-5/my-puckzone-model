"""
Train the goals-based bivariate Poisson model.
Saves to goals_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.goals
"""

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

from db import supabase, fetch_all
from features.training import build_features
from models import FEATURE_COLS, fill_features
from models.goals import BivariatePoissonGoalsModel, OT_TIE_RULE


def train():
    df = build_features(use_materialized=True)

    query = (
        supabase.table("games")
        .select("id, home_score, away_score")
        .eq("game_type", 2)
        .in_("game_state", ["OFF", "FINAL"])
    )
    scores = pd.DataFrame(fetch_all("games", query)).rename(columns={"id": "game_id"})
    df = df.merge(scores, on="game_id", how="left").dropna(subset=["home_score", "away_score"])

    X = fill_features(df[FEATURE_COLS])
    y_home = df["home_score"].astype(float)
    y_away = df["away_score"].astype(float)

    test_mask = df["season"] == 20252026
    X_train, X_test = X[~test_mask], X[test_mask]
    y_home_train, y_home_test = y_home[~test_mask], y_home[test_mask]
    y_away_train, y_away_test = y_away[~test_mask], y_away[test_mask]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)

    model = BivariatePoissonGoalsModel(use_shared_lambda3=True)
    model.fit(X_tr, y_home_train, y_away_train)

    home_rate, away_rate, _ = model.predict_rates(X_te)
    print("\nDiagnostics (test split):")
    print(f"  Mean home goals  — actual: {y_home_test.mean():.3f} | predicted: {home_rate.mean():.3f}")
    print(f"  Mean away goals  — actual: {y_away_test.mean():.3f} | predicted: {away_rate.mean():.3f}")
    print(f"  Shared covariance λ3 estimate: {model.lambda3_:.4f}")

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "model_name": "Bivariate Poisson goals model",
        "model_version": "phase-2.4.1",
        "payload_version": "2.4.1",
        "lambda3": model.lambda3_,
        "tie_rule": OT_TIE_RULE,
    }
    with open("goals_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("\nSaved to goals_model.pkl")


if __name__ == "__main__":
    train()
