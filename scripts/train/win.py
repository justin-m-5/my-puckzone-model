# scripts/train/win.py
"""
Trains the regular season win prediction model.
Saves to win_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.win
"""

import pickle
import pandas as pd
from features.training import build_features
from models import FEATURE_COLS, get_models, train_model

BEST_MODEL = "Gradient Boosting"


def train():
    df = build_features()

    X = df[FEATURE_COLS].fillna(0.5)
    y = df["target"]

    test_mask = df["season"] == 20242025
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"\nTraining {BEST_MODEL}...")
    model_cfg = get_models()[BEST_MODEL]
    result = train_model(BEST_MODEL, model_cfg, X_train, y_train, X_test, y_test)

    print(f"\nAccuracy on 2024-25 season: {result['accuracy']:.3f}")
    print(f"Baseline (always pick home): {y_test.mean():.3f}")
    print(f"Beat baseline by: {result['accuracy'] - y_test.mean():+.3f}")
    print(f"\n{result['report']}")

    payload = {
        "model": result["model"],
        "scaler": result["scaler"],
        "feature_cols": FEATURE_COLS,
        "model_name": BEST_MODEL,
    }
    with open("win_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("\nSaved to win_model.pkl")


if __name__ == "__main__":
    train()
