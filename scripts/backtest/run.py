# scripts/backtest/run.py
"""
PuckZone v2.0 — Walk-forward (rolling-origin) backtest harness.

Replaces the single-season holdout in scripts/train/win.py with an honest
rolling-origin evaluation: for each season S used as the test fold, the model
is trained on ALL prior seasons, then evaluated on season S.  This gives a
per-fold *and* aggregate picture of model performance without data leakage.

Usage
-----
    PYTHONPATH=. python3 -m scripts.backtest.run

Optional flags
--------------
  --model  {logistic,gradient_boosting,random_forest}
               Which model to evaluate (default: logistic, the production default)
  --calibrate  Wrap the chosen model in isotonic calibration (default: True)
  --min-train-seasons  N   Skip folds with fewer than N training seasons (default: 2)

Output
------
  Per-fold table: season | n_train | n_test | accuracy | log_loss | brier | auc
  Aggregate row:  totals + macro averages

  Optional hook: closing_line_value()
      Stub function for future market benchmarking (wiring real odds data is
      a Phase 2.3 task).  Uncomment the call at the bottom to activate.

Notes
-----
  - Features are built once (build_features_batch) then split by season — this
    is O(n) correct because the features themselves are already point-in-time-
    safe (built with as_of_date = game date).
  - Model instances are re-created fresh per fold so there is no state leakage
    between folds.
  - The calibration wrapper uses cv='prefit' when re-fitting on test data would
    be impractical at fold granularity; instead we use the standard 5-fold
    CalibratedClassifierCV on the training set.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from features.pipeline import DataContext, build_features_batch
from models import FEATURE_COLS, fill_features
from scripts.backtest.metrics import (
    compute_metrics,
    print_probability_metrics,
    print_feature_importance,
)


# ---------------------------------------------------------------------------
# Model catalogue (mirrors scripts/train/win.py::get_models)
# ---------------------------------------------------------------------------

_MODELS = {
    "logistic": {
        "cls": LogisticRegression,
        "kwargs": {"max_iter": 1000},
        "scale": True,
    },
    "gradient_boosting": {
        "cls": GradientBoostingClassifier,
        "kwargs": {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.05, "random_state": 42},
        "scale": False,
    },
    "random_forest": {
        "cls": RandomForestClassifier,
        "kwargs": {"n_estimators": 500, "max_depth": 6, "random_state": 42},
        "scale": False,
    },
}


def _make_model(model_key: str, calibrate: bool):
    cfg = _MODELS[model_key]
    base = cfg["cls"](**cfg["kwargs"])
    if calibrate:
        try:
            return CalibratedClassifierCV(estimator=base, method="isotonic", cv=5), cfg["scale"]
        except TypeError:
            return CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=5), cfg["scale"]
    return base, cfg["scale"]


# ---------------------------------------------------------------------------
# Optional market-benchmarking hook (Phase 2.3 placeholder)
# ---------------------------------------------------------------------------

def closing_line_value(fold_df: pd.DataFrame) -> None:
    """
    Compare model probabilities against the closing betting line.

    THIS IS A STUB.  Wiring real odds data is deferred to Phase 2.3.  When
    odds are available, implement this function to:
      1. Join fold_df on game_id / date to a table of closing de-vigged probabilities.
      2. Compute CLV = mean(log(model_prob / closing_prob)) over winning picks.
      3. Print a summary similar to print_probability_metrics.

    Parameters
    ----------
    fold_df : DataFrame with columns game_id, date, home_prob, y_true.
              Produced by the backtest loop for each fold.
    """
    # TODO (Phase 2.3): implement CLV calculation once odds ingestion is ready.
    pass


# ---------------------------------------------------------------------------
# Walk-forward evaluator
# ---------------------------------------------------------------------------

def run_backtest(
    model_key: str = "logistic",
    calibrate: bool = True,
    min_train_seasons: int = 2,
    exclude_seasons: list = None,
    game_type: int = 2,
):
    """
    Run the walk-forward backtest.

    Parameters
    ----------
    model_key          : str   Key into _MODELS.
    calibrate          : bool  Wrap with isotonic calibration.
    min_train_seasons  : int   Skip folds that would train on fewer seasons.
    exclude_seasons    : list  Seasons to skip entirely (e.g. COVID seasons).
    game_type          : int   2=regular season, 3=playoffs.

    Returns
    -------
    pd.DataFrame  Per-fold metrics (also printed to stdout).
    """
    exclude_seasons = exclude_seasons or []

    print("=" * 60)
    print("  PuckZone v2.0 — Walk-Forward Backtest")
    print("=" * 60)

    print("\nBuilding features...")
    ctx = DataContext.from_supabase()
    df = build_features_batch(ctx, game_type=game_type)

    if df.empty:
        print("ERROR: No feature rows built. Aborting backtest.")
        return pd.DataFrame()

    df = df[~df["season"].isin(exclude_seasons)].copy()
    X_all = fill_features(df[FEATURE_COLS])
    y_all = df["target"].values
    seasons = sorted(df["season"].unique())

    print(f"\n  {len(df)} total game rows across {len(seasons)} seasons")
    print(f"  Model: {model_key}{'  +  isotonic calibration' if calibrate else ''}")
    print(f"  Min training seasons before first fold: {min_train_seasons}")

    fold_results = []
    all_probs = []
    all_y = []

    print("\n" + "=" * 60)
    print(f"  {'Season':<12} {'Train':>7} {'Test':>6} {'Acc':>7} {'Base':>7} "
          f"{'LogLoss':>9} {'Brier':>8} {'AUC':>7}")
    print("-" * 60)

    for i, test_season in enumerate(seasons):
        train_seasons = seasons[:i]

        if len(train_seasons) < min_train_seasons:
            continue

        train_mask = df["season"].isin(train_seasons)
        test_mask = df["season"] == test_season

        X_train = X_all[train_mask].values
        y_train = y_all[train_mask]
        X_test = X_all[test_mask].values
        y_test = y_all[test_mask]

        if len(X_test) == 0:
            continue

        model, needs_scale = _make_model(model_key, calibrate)

        scaler = None
        if needs_scale:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        home_prob = model.predict_proba(X_test)[:, 1]

        m = compute_metrics(home_prob, y_test)
        fold_results.append({
            "season":    test_season,
            "n_train":   int(train_mask.sum()),
            "n_test":    int(test_mask.sum()),
            **m,
        })

        all_probs.extend(home_prob.tolist())
        all_y.extend(y_test.tolist())

        # Optional: call closing_line_value per fold (stub until Phase 2.3)
        fold_df = pd.DataFrame({
            "game_id":   df[test_mask]["game_id"].values,
            "date":      df[test_mask]["date"].values,
            "home_prob": home_prob,
            "y_true":    y_test,
        })
        closing_line_value(fold_df)

        print(
            f"  {test_season:<12} "
            f"{m['n_games'] + int(train_mask.sum()):>7} "
            f"{m['n_games']:>6} "
            f"{m['accuracy']:>7.3f} "
            f"{m['baseline_accuracy']:>7.3f} "
            f"{m['log_loss']:>9.4f} "
            f"{m['brier']:>8.4f} "
            f"{m['roc_auc']:>7.4f}"
        )

    if not fold_results:
        print("\nNo folds evaluated (not enough training seasons).")
        return pd.DataFrame()

    results_df = pd.DataFrame(fold_results)

    # Aggregate row.
    agg_probs = np.asarray(all_probs)
    agg_y = np.asarray(all_y)
    agg = compute_metrics(agg_probs, agg_y)

    print("-" * 60)
    print(
        f"  {'AGGREGATE':<12} "
        f"{results_df['n_train'].sum():>7} "
        f"{agg['n_games']:>6} "
        f"{agg['accuracy']:>7.3f} "
        f"{agg['baseline_accuracy']:>7.3f} "
        f"{agg['log_loss']:>9.4f} "
        f"{agg['brier']:>8.4f} "
        f"{agg['roc_auc']:>7.4f}"
    )
    print("=" * 60)

    print("\n--- Aggregate probabilistic metrics ---")
    print_probability_metrics(agg_probs, agg_y)

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Walk-forward backtest for the PuckZone win model."
    )
    p.add_argument(
        "--model",
        choices=list(_MODELS.keys()),
        default="logistic",
        help="Model to evaluate (default: logistic).",
    )
    p.add_argument(
        "--no-calibrate",
        dest="calibrate",
        action="store_false",
        default=True,
        help="Disable isotonic calibration.",
    )
    p.add_argument(
        "--min-train-seasons",
        type=int,
        default=2,
        metavar="N",
        help="Minimum number of training seasons before first fold (default: 2).",
    )
    p.add_argument(
        "--exclude-seasons",
        nargs="*",
        type=int,
        default=[],
        metavar="SEASON",
        help="Seasons to exclude entirely (e.g. 20202021 for the COVID season).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_backtest(
        model_key=args.model,
        calibrate=args.calibrate,
        min_train_seasons=args.min_train_seasons,
        exclude_seasons=args.exclude_seasons,
    )
