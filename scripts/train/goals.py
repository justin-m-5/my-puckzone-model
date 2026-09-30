"""
Train the goals-based bivariate Poisson model (the source of predicted scores).

Scores the recipe on the most recent complete season (held out), then refits on
ALL seasons and saves to candidates/goals_model.pkl (promote with
scripts.artifacts.promote after review).

Usage:
    PYTHONPATH=. python3 -m scripts.train.goals
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from db import supabase, fetch_all
from features.training import build_features
from models import FEATURE_COLS, fill_features
from models.goals import BivariatePoissonGoalsModel, OT_TIE_RULE
from scripts.backtest.metrics import compute_metrics
from scripts.train.recipe import output_path, parse_args, pick_holdout_season, training_window


def _fit(X_train, y_home_train, y_away_train):
    scaler = StandardScaler().fit(X_train)
    model = BivariatePoissonGoalsModel(use_shared_lambda3=True)
    model.fit(scaler.transform(X_train), y_home_train, y_away_train)
    return model, scaler


def train(argv=None):
    args = parse_args("Train the goals model (evaluate on holdout, refit on all).", argv)
    df = build_features(use_materialized=True)

    query = (
        supabase.table("games")
        .select("id, home_score, away_score")
        .eq("game_type", 2)
        .in_("game_state", ["OFF", "FINAL"])
    )
    scores = pd.DataFrame(fetch_all("games", query)).rename(columns={"id": "game_id"})
    df = df.merge(scores, on="game_id", how="left").dropna(subset=["home_score", "away_score"])

    X = fill_features(df[FEATURE_COLS])
    y_home = df["home_score"].astype(float)
    y_away = df["away_score"].astype(float)

    # --- 1. Honest evaluation on the held-out season ---
    holdout_season = pick_holdout_season(df, args.holdout_season)
    test_mask = df["season"] == holdout_season
    X_train, X_test = X[~test_mask], X[test_mask]
    y_home_train, y_home_test = y_home[~test_mask], y_home[test_mask]
    y_away_train, y_away_test = y_away[~test_mask], y_away[test_mask]

    print(f"Holdout {holdout_season}: train {len(X_train)} rows | test {len(X_test)} rows")

    eval_model, eval_scaler = _fit(X_train, y_home_train, y_away_train)
    X_te = eval_scaler.transform(X_test)
    home_rate, away_rate, _ = eval_model.predict_rates(X_te)
    home_win_prob = eval_model.predict_home_win_prob(X_te)
    home_won = (y_home_test > y_away_test).astype(int)

    holdout_metrics = {
        "n_games": int(len(X_test)),
        "home_goals_mae": float(np.mean(np.abs(home_rate - y_home_test))),
        "away_goals_mae": float(np.mean(np.abs(away_rate - y_away_test))),
        "total_goals_mae": float(np.mean(np.abs((home_rate + away_rate) - (y_home_test + y_away_test)))),
        "mean_home_goals": {"actual": float(y_home_test.mean()), "predicted": float(home_rate.mean())},
        "mean_away_goals": {"actual": float(y_away_test.mean()), "predicted": float(away_rate.mean())},
        "win": compute_metrics(home_win_prob, home_won) if len(X_test) and home_won.nunique() > 1 else None,
    }
    print(f"\nDiagnostics ({holdout_season} held out):")
    print(f"  Mean home goals  — actual: {y_home_test.mean():.3f} | predicted: {home_rate.mean():.3f}")
    print(f"  Mean away goals  — actual: {y_away_test.mean():.3f} | predicted: {away_rate.mean():.3f}")
    print(f"  Total goals MAE: {holdout_metrics['total_goals_mae']:.3f}")
    print(f"  Shared covariance λ3 estimate: {eval_model.lambda3_:.4f}")

    # --- 2. Final model on every season ---
    print(f"\nRefitting on all seasons: {len(X)} rows...")
    model, scaler = _fit(X, y_home, y_away)

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "model_name": "Bivariate Poisson goals model",
        "model_version": "phase-2.4.1",
        "payload_version": "2.4.1",
        "lambda3": model.lambda3_,
        "tie_rule": OT_TIE_RULE,
        "training": training_window(df, holdout_season, holdout_metrics),
    }
    path = output_path(args.out, "goals_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved to {path} (trained through {payload['training']['trained_through']})")


if __name__ == "__main__":
    train()
