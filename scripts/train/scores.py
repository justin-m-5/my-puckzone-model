# scripts/train/scores.py
"""
Trains score regressors for regular season games.
Saves to score_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.scores
"""

import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from db import supabase, fetch_all
from features.training import build_features
from models import FEATURE_COLS, fill_features


def train():
    df = build_features()

    query = supabase.table("games").select("id, home_score, away_score").eq("game_type", 2).in_("game_state", ["OFF", "FINAL"])
    scores = pd.DataFrame(fetch_all("games", query)).rename(columns={"id": "game_id"})
    df = df.merge(scores, on="game_id", how="left").dropna(subset=["home_score", "away_score"])

    X = fill_features(df[FEATURE_COLS])
    y_home = df["home_score"]
    y_away = df["away_score"]

    test_mask = df["season"] == 20252026
    X_train, X_test = X[~test_mask], X[test_mask]
    y_home_train, y_home_test = y_home[~test_mask], y_home[test_mask]
    y_away_train, y_away_test = y_away[~test_mask], y_away[test_mask]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\nTraining home score model...")
    home_model = GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42)
    home_model.fit(X_train, y_home_train)
    home_mae = mean_absolute_error(y_home_test, home_model.predict(X_test))
    print(f"  MAE: {home_mae:.3f}")

    print("Training away score model...")
    away_model = GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42)
    away_model.fit(X_train, y_away_train)
    away_mae = mean_absolute_error(y_away_test, away_model.predict(X_test))
    print(f"  MAE: {away_mae:.3f}")

    home_baseline = mean_absolute_error(y_home_test, [y_home_train.mean()] * len(y_home_test))
    away_baseline = mean_absolute_error(y_away_test, [y_away_train.mean()] * len(y_away_test))
    print(f"\nBaseline MAE — home: {home_baseline:.3f} | away: {away_baseline:.3f}")
    print(f"Beat baseline — home: {home_baseline - home_mae:+.3f} | away: {away_baseline - away_mae:+.3f}")

    payload = {
        "home_model": home_model,
        "away_model": away_model,
        "feature_cols": FEATURE_COLS,
    }
    with open("score_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("\nSaved to score_model.pkl")


if __name__ == "__main__":
    train()
