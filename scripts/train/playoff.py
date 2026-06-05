# scripts/train/playoff.py
"""
Trains the playoff-specific win prediction model.
Saves to playoff_model.pkl.

Usage:
    PYTHONPATH=. python3 -m scripts.train.playoff

Why this file is more involved than the others
-----------------------------------------------
The raw playoff model RANKS teams fine (AUC ~0.60) but is badly miscalibrated:
it predicts home wins ~45% of the time while playoff home teams actually win
~60%. At a 0.5 cutoff that flips a pile of games to the away side, so accuracy
falls below the always-pick-home baseline.

Two fixes, both fit WITHOUT touching the test season:
  1. Platt (sigmoid) calibration. Sigmoid — not isotonic — because the playoff
     set is tiny (~600 games). Its bias term absorbs the home-ice undershoot, so
     the calibrated probabilities (what scripts/predict/run.py shows and cuts at
     0.5) move toward the true home rate.
  2. A decision threshold tuned by leave-one-season-out CV on the TRAINING
     seasons. If calibration doesn't fully fix the undershoot, a threshold < 0.5
     handles the rest.

The leave-one-season-out pass PRINTS out-of-fold mean-predicted vs actual home
rate. If predicted << actual there too, the undershoot is STRUCTURAL and these
fixes transfer to the test season. If it only shows on the test season, no
calibration can fix it — that's a real finding, not a bug.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    log_loss, brier_score_loss, roc_auc_score,
)
from features.training import build_playoff_features
from models.playoff import PLAYOFF_FEATURE_COLS, get_playoff_model

# Last COMPLETE postseason. Do NOT use 20252026 while those playoffs are live.
TEST_SEASON = 20242025

# "sigmoid" (Platt) is right for this small dataset; "isotonic" overfits here;
# None disables calibration.
CALIBRATION_METHOD = "sigmoid"

# Tune the decision threshold via leave-one-season-out on training seasons and
# save it in the payload. See the note at the bottom about run.py honoring it.
TUNE_THRESHOLD = True

# Optional: drop neutral-site / no-crowd playoff seasons so the model learns a
# real home-ice edge. 2019-20 was played ENTIRELY in the bubble (no home ice);
# 2020-21 had limited/empty arenas. Playoff data is scarce, so dropping a season
# costs signal — try [] first, then [20192020] and watch log loss.
EXCLUDE_NEUTRAL_SITE_SEASONS = []


def _make_calibrated(estimator, method, cv):
    """CalibratedClassifierCV renamed base_estimator->estimator in newer sklearn."""
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def _build_estimator():
    """A fresh (optionally calibrated) playoff classifier."""
    base = get_playoff_model()
    if CALIBRATION_METHOD:
        return _make_calibrated(base, CALIBRATION_METHOD, cv=5)
    return base


def _importances(model, feature_cols):
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


def _loso_oof_probs(X, y, seasons):
    """
    Leave-one-season-out out-of-fold home-win probabilities over the training
    seasons. For each season, fit on the OTHER training seasons and predict it.
    """
    oof = np.full(len(X), np.nan)
    for s in sorted(seasons.unique()):
        hold = (seasons.values == s)
        if hold.sum() == 0 or (~hold).sum() == 0:
            continue
        est = _build_estimator()
        est.fit(X[~hold], y[~hold])
        oof[hold] = est.predict_proba(X[hold])[:, 1]
    return oof


def _best_threshold(y_true, probs, lo=0.30, hi=0.70, step=0.01):
    """Threshold on home-win prob that maximizes accuracy on the given data."""
    y_true = np.asarray(y_true)
    best_t, best_acc = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        acc = accuracy_score(y_true, (probs > t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def print_probability_metrics(home_prob, y_test):
    y_arr = np.asarray(y_test, dtype=float)
    ll = log_loss(y_test, home_prob, labels=[0, 1])
    brier = brier_score_loss(y_test, home_prob)
    try:
        auc_str = f"{roc_auc_score(y_test, home_prob):.4f}"
    except ValueError:
        auc_str = "n/a (one class in test)"

    base_rate = y_arr.mean()
    base_probs = np.full_like(home_prob, base_rate, dtype=float)
    base_ll = log_loss(y_test, base_probs, labels=[0, 1])
    base_brier = brier_score_loss(y_test, base_probs)

    print("\n--- Probabilistic metrics (lower log loss / Brier = better) ---")
    print(f"  Log loss:  {ll:.4f}   (baseline {base_ll:.4f}, better by {base_ll - ll:+.4f})")
    print(f"  Brier:     {brier:.4f}   (baseline {base_brier:.4f}, better by {base_brier - brier:+.4f})")
    print(f"  ROC AUC:   {auc_str}")
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
    df = build_playoff_features()
    if df.empty:
        print("No playoff data found. Exiting.")
        return

    if EXCLUDE_NEUTRAL_SITE_SEASONS:
        before = len(df)
        df = df[~df["season"].isin(EXCLUDE_NEUTRAL_SITE_SEASONS)].copy()
        print(f"Excluded {before - len(df)} rows from neutral-site seasons "
              f"{EXCLUDE_NEUTRAL_SITE_SEASONS}.")

    print(f"\nSeasons: {[str(s) for s in sorted(df['season'].unique())]}")
    print(f"Test season: {TEST_SEASON}")
    print(f"Calibration: {CALIBRATION_METHOD or 'none'}")

    X = df[PLAYOFF_FEATURE_COLS].fillna(0.5).reset_index(drop=True)
    y = df["target"].reset_index(drop=True)
    season_col = df["season"].reset_index(drop=True)

    test_mask = (season_col == TEST_SEASON).values
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    seasons_train = season_col[~test_mask]

    print(f"Train: {len(X_train)} games | Test: {len(X_test)} games")
    if len(X_test) < 5:
        print("Test set too small — set TEST_SEASON to a completed postseason.")
        return

    # --- leave-one-season-out on TRAINING data: diagnose + tune threshold ---
    threshold = 0.5
    if TUNE_THRESHOLD:
        print("\nLeave-one-season-out on training seasons...")
        Xtr = X_train.reset_index(drop=True)
        ytr = y_train.reset_index(drop=True)
        str_ = seasons_train.reset_index(drop=True)
        oof = _loso_oof_probs(Xtr, ytr, str_)
        valid = ~np.isnan(oof)
        print(f"  OOF mean predicted home win%: {oof[valid].mean() * 100:.1f}%  "
              f"(actual {ytr[valid].mean() * 100:.1f}%)")
        print("  -> predicted << actual here means the undershoot is STRUCTURAL "
              "and the fixes transfer to the test season.")
        threshold, oof_acc = _best_threshold(ytr[valid], oof[valid])
        print(f"  Tuned decision threshold: {threshold:.2f} (OOF accuracy {oof_acc:.3f})")

    # --- final model: fit on all training data, evaluate on the test season ---
    print(f"\nTraining final model on {len(X_train)} games...")
    model = _build_estimator()
    model.fit(X_train, y_train)
    home_prob = model.predict_proba(X_test)[:, 1]

    base = y_test.mean()
    acc_50 = accuracy_score(y_test, (home_prob > 0.5).astype(int))
    acc_t = accuracy_score(y_test, (home_prob > threshold).astype(int))

    print(f"\nBaseline (always home):       {base:.3f}")
    print(f"Accuracy @ 0.50:              {acc_50:.3f}  ({acc_50 - base:+.3f} vs baseline)")
    print(f"Accuracy @ {threshold:.2f} (tuned):      {acc_t:.3f}  ({acc_t - base:+.3f} vs baseline)")
    print(f"\n{classification_report(y_test, (home_prob > threshold).astype(int), target_names=['Away Win', 'Home Win'], labels=[0, 1], zero_division=0)}")

    print_probability_metrics(home_prob, y_test)
    print_feature_importance(model, PLAYOFF_FEATURE_COLS)

    payload = {
        "model": model,
        "scaler": None,
        "feature_cols": PLAYOFF_FEATURE_COLS,
        "model_name": "Playoff Gradient Boosting"
        + (f" ({CALIBRATION_METHOD}-calibrated)" if CALIBRATION_METHOD else ""),
        "decision_threshold": threshold,
    }
    with open("playoff_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved to playoff_model.pkl (decision_threshold={threshold:.2f})")


if __name__ == "__main__":
    train()