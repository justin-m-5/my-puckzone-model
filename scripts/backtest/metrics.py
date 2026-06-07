# scripts/backtest/metrics.py
"""
Shared evaluation metrics for PuckZone model assessment.

Factored out of scripts/train/win.py so both the training script and the
walk-forward backtest harness can call the same reporting functions without
duplication.

Public API
----------
  print_probability_metrics(home_prob, y_true)
      Log loss, Brier score, ROC AUC + calibration table.

  print_feature_importance(model, feature_cols, top_n=15)
      Top-N feature importances, handling calibrated wrappers.

  compute_metrics(home_prob, y_true) -> dict
      Machine-readable version of the probability metrics.
"""

import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score


def compute_metrics(home_prob: np.ndarray, y_true) -> dict:
    """
    Return a dict of evaluation metrics for a probabilistic classifier.

    Parameters
    ----------
    home_prob : array-like, shape (n,)   Predicted home-win probabilities.
    y_true    : array-like, shape (n,)   Actual outcomes (1=home win, 0=away win).

    Returns
    -------
    dict with keys: accuracy, log_loss, brier, roc_auc,
                    baseline_log_loss, baseline_brier, baseline_accuracy,
                    mean_predicted_prob, actual_home_rate, n_games.
    """
    home_prob = np.asarray(home_prob, dtype=float)
    y_arr = np.asarray(y_true, dtype=float)
    preds = (home_prob > 0.5).astype(int)

    base_rate = y_arr.mean()
    base_probs = np.full_like(home_prob, base_rate)

    return {
        "n_games":              int(len(y_arr)),
        "accuracy":             float(accuracy_score(y_arr, preds)),
        "baseline_accuracy":    float(base_rate),
        "log_loss":             float(log_loss(y_arr, home_prob, labels=[0, 1])),
        "baseline_log_loss":    float(log_loss(y_arr, base_probs, labels=[0, 1])),
        "brier":                float(brier_score_loss(y_arr, home_prob)),
        "baseline_brier":       float(brier_score_loss(y_arr, base_probs)),
        "roc_auc":              float(roc_auc_score(y_arr, home_prob)),
        "mean_predicted_prob":  float(home_prob.mean()),
        "actual_home_rate":     float(base_rate),
    }


def print_probability_metrics(home_prob, y_true):
    """
    Print a human-readable evaluation report.

    Displays log loss, Brier score, ROC AUC (with always-pick-home baselines)
    and a 10-bucket calibration table.

    This function is kept identical in signature to the version in
    scripts/train/win.py so existing call sites continue to work.
    """
    m = compute_metrics(home_prob, y_true)
    home_prob = np.asarray(home_prob, dtype=float)
    y_arr = np.asarray(y_true, dtype=float)

    acc_beat = m["accuracy"] - m["baseline_accuracy"]
    ll_beat = m["baseline_log_loss"] - m["log_loss"]
    brier_beat = m["baseline_brier"] - m["brier"]

    print("\n--- Probabilistic metrics (lower log loss / Brier = better) ---")
    print(
        f"  Accuracy:  {m['accuracy']:.3f}   "
        f"(baseline {m['baseline_accuracy']:.3f}, better by {acc_beat:+.3f})"
    )
    print(
        f"  Log loss:  {m['log_loss']:.4f}   "
        f"(baseline {m['baseline_log_loss']:.4f}, better by {ll_beat:+.4f})"
    )
    print(
        f"  Brier:     {m['brier']:.4f}   "
        f"(baseline {m['baseline_brier']:.4f}, better by {brier_beat:+.4f})"
    )
    print(
        f"  ROC AUC:   {m['roc_auc']:.4f}   "
        f"(0.5 = coin flip, 1.0 = perfect ranking)"
    )
    print(
        f"  Mean predicted home win%: {m['mean_predicted_prob'] * 100:.1f}%  "
        f"(actual: {m['actual_home_rate'] * 100:.1f}%)"
    )

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
        print(
            f"  {label:>11}  {n:>5}  "
            f"{home_prob[mask].mean() * 100:>5.1f}%  "
            f"{y_arr[mask].mean() * 100:>5.1f}%"
        )


def print_feature_importance(model, feature_cols, top_n: int = 15):
    """
    Print the top-N features by importance, handling calibrated wrappers.

    Identical to the version in scripts/train/win.py::_importances.
    """
    importances, kind = _get_importances(model)
    if importances is None:
        return
    ranked = sorted(zip(feature_cols, importances), key=lambda t: t[1], reverse=True)
    print(f"\n--- Top {top_n} features ({kind}) ---")
    for name, val in ranked[:top_n]:
        print(f"  {name:<28} {val:.4f}")


def _get_importances(model):
    """Extract feature importances, digging through a CalibratedClassifierCV wrapper."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_, "tree importance"
    if hasattr(model, "coef_"):
        return np.abs(model.coef_).ravel(), "abs(coef)"
    if hasattr(model, "calibrated_classifiers_"):
        imps = []
        kinds = set()
        for cc in model.calibrated_classifiers_:
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if est is None:
                continue
            if hasattr(est, "feature_importances_"):
                imps.append(est.feature_importances_)
                kinds.add("tree")
            elif hasattr(est, "coef_"):
                imps.append(np.abs(est.coef_).ravel())
                kinds.add("coef")
        if imps:
            label = (
                "abs(coef) (avg over calibrated folds)"
                if kinds == {"coef"}
                else "tree importance (avg over calibrated folds)"
            )
            return np.mean(imps, axis=0), label
    return None, None


def ranked_probability_score(score_probs: np.ndarray, home_score: int, away_score: int) -> float:
    """
    Ranked Probability Score on goal-differential outcomes induced by score_probs.
    Lower is better.
    """
    max_goals = score_probs.shape[0] - 1
    diffs = np.arange(-max_goals, max_goals + 1)
    diff_probs = np.zeros_like(diffs, dtype=float)

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            diff_probs[(i - j) + max_goals] += score_probs[i, j]

    diff_probs = diff_probs / max(diff_probs.sum(), 1e-12)
    obs = np.zeros_like(diff_probs, dtype=float)
    actual_diff = int(np.clip(int(home_score) - int(away_score), -max_goals, max_goals))
    obs[actual_diff + max_goals] = 1.0

    cdf_pred = np.cumsum(diff_probs)
    cdf_obs = np.cumsum(obs)
    return float(np.sum((cdf_pred - cdf_obs) ** 2) / (len(diff_probs) - 1))


def exact_scoreline_hit(score_probs: np.ndarray, home_score: int, away_score: int) -> float:
    """1.0 if argmax(score_probs) equals the observed scoreline, else 0.0."""
    max_goals = score_probs.shape[0] - 1
    home_score = int(home_score)
    away_score = int(away_score)
    if home_score > max_goals or away_score > max_goals:
        return 0.0
    pred = np.unravel_index(int(np.argmax(score_probs)), score_probs.shape)
    return 1.0 if pred == (home_score, away_score) else 0.0
