# scripts/train/player.py
"""
Trains the player scoring prediction model.
Predicts: will a skater record >= 1 point in a game?
Saves to player_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.player
"""

import pickle
import pandas as pd
from features.players import get_skater_stats, build_skater_rolling, build_player_training_rows
from models.player import PLAYER_FEATURE_COLS, get_player_models, train_player_model


def train():
    print("Loading skater stats...")
    skater_df = get_skater_stats()
    print(f"  {len(skater_df)} player-game rows loaded")

    print("Building rolling features...")
    rolling_df = build_skater_rolling(skater_df)

    print("Building training rows...")
    df = build_player_training_rows(rolling_df)

    X = df[PLAYER_FEATURE_COLS].fillna(0)
    y = df["scored_point"]

    test_mask = df["season"] == 20252026
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Baseline (always predict no point): {1 - y_test.mean():.3f}\n")

    best_result = None
    for name, model_cfg in get_player_models().items():
        print(f"Training {name}...")
        result = train_player_model(name, model_cfg, X_train, y_train, X_test, y_test)
        print(f"  Accuracy: {result['accuracy']:.3f}  Macro F1: {result['macro_f1']:.3f}")
        if best_result is None or result["macro_f1"] > best_result["macro_f1"]:
            best_result = result

    print(f"\nBest: {best_result['name']} (accuracy={best_result['accuracy']:.3f}, macro_f1={best_result['macro_f1']:.3f})")

    payload = {
        "model": best_result["model"],
        "scaler": best_result["scaler"],
        "feature_cols": PLAYER_FEATURE_COLS,
        "model_name": best_result["name"],
    }
    with open("player_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("Saved to player_model.pkl")


if __name__ == "__main__":
    train()
