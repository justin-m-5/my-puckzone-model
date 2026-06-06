# scripts/train/win.py
"""
Trains the regular season win prediction model.
Saves to win_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.win

Two experiments are wired in (toggle the flags below):
  EXCLUDE_TRAINING_SEASONS - drop empty/limited-arena COVID seasons so the model
                             learns a normal home-ice advantage. 2020-21 was
                             played almost entirely without fans, which flattens
                             home advantage and biases predictions toward the
                             away side on a normal test season.
  CALIBRATE                - wrap the model in isotonic calibration so a "65%"
                             prediction actually means ~65%.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from features.training import build_features
from models import FEATURE_COLS, fill_features, get_models
from scripts.backtest.metrics import (
    print_probability_metrics,  # noqa: F401 — re-exported for backward compat
    print_feature_importance,   # noqa: F401 — re-exported for backward compat
)

# Chosen from the latest compare run (best accuracy + calibrated linear/isotonic behavior).
BEST_MODEL = "Logistic Regression"

# 2020-21 regular season was played in empty/limited arenas -> weak home edge.
# Add 20192020 too if you want to drop the COVID-shortened tail as well.
EXCLUDE_TRAINING_SEASONS = []

# Isotonic calibration of the probabilities (does not change feature inputs).
CALIBRATE = True


def _make_calibrated(estimator, method="isotonic", cv=5):
    """CalibratedClassifierCV renamed base_estimator->estimator in newer sklearn."""
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def train():
    df = build_features()

    X = fill_features(df[FEATURE_COLS])
    y = df["target"]

    test_mask = df["season"] == 20252026
    exclude_mask = df["season"].isin(EXCLUDE_TRAINING_SEASONS)
    train_mask = (~test_mask) & (~exclude_mask)

    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        print(f"\nExcluding {n_excluded} training rows from seasons {EXCLUDE_TRAINING_SEASONS} "
            f"(empty/limited-arena COVID seasons).")

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    model_cfg = get_models()[BEST_MODEL]
    needs_scale = model_cfg["scale"]

    # Scale if the chosen model needs it (Gradient Boosting does not).
    scaler = None
    X_tr, X_te = X_train, X_test
    if needs_scale:
        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)
        X_te = scaler.transform(X_test)

    print(f"\nTraining {BEST_MODEL}{' + isotonic calibration' if CALIBRATE else ''}...")
    if CALIBRATE:
        model = _make_calibrated(get_models()[BEST_MODEL]["model"], "isotonic", cv=5)
    else:
        model = get_models()[BEST_MODEL]["model"]
    model.fit(X_tr, y_train)

    home_prob = model.predict_proba(X_te)[:, 1]
    preds = (home_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy on 2024-25 season: {acc:.3f}")
    print(f"Baseline (always pick home): {y_test.mean():.3f}")
    print(f"Beat baseline by: {acc - y_test.mean():+.3f}")
    print(f"\n{classification_report(y_test, preds, target_names=['Away Win', 'Home Win'])}")

    print_probability_metrics(home_prob, y_test)
    print_feature_importance(model, FEATURE_COLS)

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "model_name": BEST_MODEL + (" (calibrated)" if CALIBRATE else ""),
    }
    with open("win_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("\nSaved to win_model.pkl")


if __name__ == "__main__":
    train()