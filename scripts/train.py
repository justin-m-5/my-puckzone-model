# scripts/train.py

"""
Run this once you have decided on the best model from compare.py.
Trains the chosen model on ALL data and saves it to disk.

Usage:
    python3 train.py
"""

import pickle
import pandas as pd
from features import build_features
from models import FEATURE_COLS, get_models, train_model

# --- change this to whichever model won in compare.py ---
BEST_MODEL = "Gradient Boosting"

def train():
    df = build_features()

    X = df[FEATURE_COLS].fillna(0.5)
    y = df["target"]

    test_mask = df["season"] == 20242025
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"\nTraining {BEST_MODEL} on all data...")
    model_cfg = get_models()[BEST_MODEL]
    result = train_model(BEST_MODEL, model_cfg, X_train, y_train, X_test, y_test)

    print(f"\nAccuracy on 2024-25 season: {result['accuracy']:.3f}")
    print(f"Baseline (always pick home): {y_test.mean():.3f}")
    print(f"Beat baseline by: {result['accuracy'] - y_test.mean():+.3f}")
    print(f"\n{result['report']}")

    # save model + scaler to disk
    payload = {
        "model": result["model"],
        "scaler": result["scaler"],
        "feature_cols": FEATURE_COLS,
        "model_name": BEST_MODEL,
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print(f"\nModel saved to model.pkl")

if __name__ == "__main__":
    train()