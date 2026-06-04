# scripts/train/xg.py
"""
Trains the expected goals (xG) model on shot events.
Predicts probability that a shot on goal results in a goal.
Saves to xg_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.xg
"""

import pickle
import pandas as pd
from features.plays import get_shot_events, build_xg_features
from models.xg import XG_FEATURE_COLS, get_xg_models, train_xg_model


def train():
    print("Loading shot events...")
    shot_df = get_shot_events()
    print(f"  {len(shot_df)} shot events loaded")

    print("Building xG features...")
    df = build_xg_features(shot_df)
    df = df.dropna(subset=["x_coord", "y_coord"])
    print(f"  {len(df)} shots with valid coordinates")
    print(f"  Goal rate: {df['is_goal'].mean():.3f}")

    X = df[XG_FEATURE_COLS].fillna(0)
    y = df["is_goal"]

    test_mask = df["season"] == 20252026
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Goal rate — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}\n")

    best_result = None
    for name, model_cfg in get_xg_models().items():
        print(f"Training {name}...")
        result = train_xg_model(name, model_cfg, X_train, y_train, X_test, y_test)
        print(f"  AUC: {result['auc']:.4f}  Log Loss: {result['log_loss']:.4f}  Brier: {result['brier']:.4f}")
        if best_result is None or result["auc"] > best_result["auc"]:
            best_result = result

    print(f"\nBest: {best_result['name']} (AUC={best_result['auc']:.4f})")

    payload = {
        "model": best_result["model"],
        "scaler": best_result["scaler"],
        "feature_cols": XG_FEATURE_COLS,
        "model_name": best_result["name"],
    }
    with open("xg_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("Saved to xg_model.pkl")


if __name__ == "__main__":
    train()
