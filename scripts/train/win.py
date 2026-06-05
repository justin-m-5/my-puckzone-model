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
from sklearn.metrics import (
    accuracy_score, classification_report,
    log_loss, brier_score_loss, roc_auc_score,
)
from features.training import build_features
from models import FEATURE_COLS, get_models

BEST_MODEL = "Gradient Boosting"

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


def _importances(model, feature_cols):
    """Pull feature importances, digging through a calibrated wrapper if needed."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_, "tree importance"
    if hasattr(model, "coef_"):
        return np.abs(model.coef_).ravel(), "abs(coef)"
    if hasattr(model, "calibrated_classifiers_"):
        imps = []
        for cc in model.calibrated_classifiers_:
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if est is None:
                continue
            if hasattr(est, "feature_importances_"):
                imps.append(est.feature_importances_)
            elif hasattr(est, "coef_"):
                imps.append(np.abs(est.coef_).ravel())
        if imps:
            return np.mean(imps, axis=0), "tree importance (avg over calibrated folds)"
    return None, None


def print_probability_metrics(home_prob, y_test):
    """Evaluate as a probabilistic classifier, not just a 0/1 predictor."""
    y_arr = np.asarray(y_test, dtype=float)

    ll = log_loss(y_test, home_prob, labels=[0, 1])
    brier = brier_score_loss(y_test, home_prob)
    auc = roc_auc_score(y_test, home_prob)

    base_rate = y_arr.mean()
    base_probs = np.full_like(home_prob, base_rate, dtype=float)
    base_ll = log_loss(y_test, base_probs, labels=[0, 1])
    base_brier = brier_score_loss(y_test, base_probs)

    print("\n--- Probabilistic metrics (lower log loss / Brier = better) ---")
    print(f"  Log loss:  {ll:.4f}   (baseline {base_ll:.4f}, better by {base_ll - ll:+.4f})")
    print(f"  Brier:     {brier:.4f}   (baseline {base_brier:.4f}, better by {base_brier - brier:+.4f})")
    print(f"  ROC AUC:   {auc:.4f}   (0.5 = coin flip, 1.0 = perfect ranking)")
    print(f"  Mean predicted home win%: {home_prob.mean() * 100:.1f}%  "
          f"(actual home win%: {base_rate * 100:.1f}%)")

    print("\n--- Calibration (predicted home win% vs actual) ---")
    print(f"  {'bucket':>11}  {'n':>5}  {'pred':>6}  {'actual':>6}")
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(home_prob, bins) - 1, 0, len(bins) - 2)
    for b in range(len(bins) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        label = f"{bins[b] * 100:>3.0f}-{bins[b + 1] * 100:>3.0f}%"
        print(f"  {label:>11}  {n:>5}  {home_prob[mask].mean() * 100:>5.1f}%  {y_arr[mask].mean() * 100:>5.1f}%")


def print_feature_importance(model, feature_cols, top_n=15):
    importances, kind = _importances(model, feature_cols)
    if importances is None:
        return
    ranked = sorted(zip(feature_cols, importances), key=lambda t: t[1], reverse=True)
    print(f"\n--- Top {top_n} features ({kind}) ---")
    for name, val in ranked[:top_n]:
        print(f"  {name:<28} {val:.4f}")


def train():
    df = build_features()

    X = df[FEATURE_COLS].fillna(0.5)
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