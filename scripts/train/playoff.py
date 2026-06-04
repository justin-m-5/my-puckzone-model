# scripts/train/playoff.py
"""
Trains the playoff-specific win prediction model.
Saves to playoff_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.playoff
"""

import pickle
import pandas as pd
from features.training import build_playoff_features
from models.playoff import PLAYOFF_FEATURE_COLS, get_playoff_model, train_playoff_model


def train():
    df = build_playoff_features()

    if df.empty:
        print("No playoff data found. Exiting.")
        return

    seasons = sorted(df["season"].unique())
    print(f"\nSeasons: {[str(s) for s in seasons]}")

    X = df[PLAYOFF_FEATURE_COLS].fillna(0.5)
    y = df["target"]

    test_mask = df["season"] == 20242025
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"\nTrain: {len(X_train)} games | Test: {len(X_test)} games")

    if len(X_test) < 5:
        print("Test set too small — training on all data.")
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = get_playoff_model()
    result = train_playoff_model(model, X_train, y_train, X_test, y_test)

    print(f"\nAccuracy:         {result['accuracy']:.3f}")
    print(f"Baseline (home):  {y_test.mean():.3f}")
    print(f"Beat baseline by: {result['accuracy'] - y_test.mean():+.3f}")
    print(f"\n{result['report']}")

    payload = {
        "model": result["model"],
        "scaler": None,
        "feature_cols": PLAYOFF_FEATURE_COLS,
        "model_name": "Playoff Gradient Boosting",
    }
    with open("playoff_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("\nSaved to playoff_model.pkl")


if __name__ == "__main__":
    train()
