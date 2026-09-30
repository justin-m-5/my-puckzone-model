# scripts/train/win.py
"""
Trains the regular season win prediction model.

Scores the recipe on the most recent complete season (held out), then refits on
ALL seasons and saves to candidates/win_model.pkl (promote with
scripts.artifacts.promote after review).

Usage:
    PYTHONPATH=. python3 -m scripts.train.win
    PYTHONPATH=. python3 -m scripts.train.win --holdout-season 20252026 --out candidates

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
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from features.training import build_features
from models import FEATURE_COLS, fill_features, get_models
from scripts.backtest.metrics import (
    compute_metrics,
    print_probability_metrics,  # noqa: F401 — re-exported for backward compat
    print_feature_importance,   # noqa: F401 — re-exported for backward compat
)
from scripts.train.recipe import output_path, parse_args, pick_holdout_season, training_window

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


def _fit(X_train, y_train):
    """Fit the chosen recipe; returns (model, scaler)."""
    scaler = None
    X_tr = X_train
    if get_models()[BEST_MODEL]["scale"]:
        scaler = StandardScaler().fit(X_train)
        X_tr = scaler.transform(X_train)

    if CALIBRATE:
        model = _make_calibrated(get_models()[BEST_MODEL]["model"], "isotonic", cv=5)
    else:
        model = get_models()[BEST_MODEL]["model"]
    model.fit(X_tr, y_train)
    return model, scaler


def _home_prob(model, scaler, X):
    return model.predict_proba(scaler.transform(X) if scaler is not None else X)[:, 1]


def train(argv=None):
    args = parse_args("Train the regular-season win model (evaluate on holdout, refit on all).", argv)
    df = build_features()

    X = fill_features(df[FEATURE_COLS])
    y = df["target"]

    holdout_season = pick_holdout_season(df, args.holdout_season)
    test_mask = df["season"] == holdout_season
    exclude_mask = df["season"].isin(EXCLUDE_TRAINING_SEASONS)
    train_mask = (~test_mask) & (~exclude_mask)

    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        print(f"\nExcluding {n_excluded} training rows from seasons {EXCLUDE_TRAINING_SEASONS} "
            f"(empty/limited-arena COVID seasons).")

    # --- 1. Honest evaluation: train without the holdout season, score on it ---
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Holdout {holdout_season}: train {len(X_train)} rows | test {len(X_test)} rows")

    print(f"\nTraining {BEST_MODEL}{' + isotonic calibration' if CALIBRATE else ''} (holdout fit)...")
    eval_model, eval_scaler = _fit(X_train, y_train)
    home_prob = _home_prob(eval_model, eval_scaler, X_test)
    preds = (home_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy on {holdout_season} (held out): {acc:.3f}")
    print(f"Baseline (always pick home): {y_test.mean():.3f}")
    print(f"Beat baseline by: {acc - y_test.mean():+.3f}")
    print(f"\n{classification_report(y_test, preds, target_names=['Away Win', 'Home Win'])}")

    print_probability_metrics(home_prob, y_test)
    print_feature_importance(eval_model, FEATURE_COLS)
    holdout_metrics = compute_metrics(home_prob, y_test)

    # --- 2. Final model: same recipe on every season (including the holdout) ---
    full_mask = ~exclude_mask
    print(f"\nRefitting on all seasons: {int(full_mask.sum())} rows...")
    model, scaler = _fit(X[full_mask], y[full_mask])

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "model_name": BEST_MODEL + (" (calibrated)" if CALIBRATE else ""),
        "training": training_window(df[full_mask], holdout_season, holdout_metrics),
    }
    path = output_path(args.out, "win_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved to {path} (trained through {payload['training']['trained_through']})")


if __name__ == "__main__":
    train()
