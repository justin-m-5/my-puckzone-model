# scripts/train/playoff_scores.py
"""
Trains score regressors for playoff games.
Saves to playoff_score_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.playoff_scores
"""

import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from db import supabase, fetch_all
from features.training import build_playoff_features
from models import fill_features
from models.playoff import PLAYOFF_FEATURE_COLS


def train():
    df = build_playoff_features()

    if df.empty:
        print("No playoff data found. Exiting.")
        return

    query = supabase.table("games").select("id, home_score, away_score").eq("game_type", 3).in_("game_state", ["OFF", "FINAL"])
    scores = pd.DataFrame(fetch_all("games", query)).rename(columns={"id": "game_id"})
    df = df.merge(scores, on="game_id", how="left").dropna(subset=["home_score", "away_score"])

    seasons = sorted(df["season"].unique())
    print(f"Seasons: {[str(s) for s in seasons]}")
    print(f"Total playoff games with scores: {len(df)}")

    X = fill_features(df[PLAYOFF_FEATURE_COLS])
    y_home = df["home_score"]
    y_away = df["away_score"]

    test_mask = df["season"] == 20252026
    X_train, X_test = X[~test_mask], X[test_mask]
    y_home_train, y_home_test = y_home[~test_mask], y_home[test_mask]
    y_away_train, y_away_test = y_away[~test_mask], y_away[test_mask]

    print(f"\nTrain: {len(X_train)} games | Test: {len(X_test)} games")

    if len(X_test) < 5:
        print("Test set too small — training on all data (accuracy will be optimistic).")
        X_train, y_home_train, y_away_train = X, y_home, y_away
        X_test, y_home_test, y_away_test = X, y_home, y_away

    print("\nTraining home score model...")
    home_model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    home_model.fit(X_train, y_home_train)
    home_mae = mean_absolute_error(y_home_test, home_model.predict(X_test))
    print(f"  MAE: {home_mae:.3f}")

    print("Training away score model...")
    away_model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
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
        "feature_cols": PLAYOFF_FEATURE_COLS,
    }
    with open("playoff_score_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("\nSaved to playoff_score_model.pkl")


if __name__ == "__main__":
    train()
