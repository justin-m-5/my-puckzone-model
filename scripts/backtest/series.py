"""
Series-level playoff evaluation using goals-model lambdas + Monte Carlo best-of-7 simulation.

Usage:
    PYTHONPATH=. python3 -m scripts.backtest.series
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from features.pipeline import DataContext, build_feature_row
from features.training import build_features, build_playoff_features
from models import FEATURE_COLS, PLAYOFF_FEATURE_COLS, fill_features
from models.goals import BivariatePoissonGoalsModel
from models.series_sim import series_win_prob_from_lambdas, simulate_series
from scripts.backtest.metrics import compute_metrics, print_probability_metrics


def _completed_series(playoff_df: pd.DataFrame, games_df: pd.DataFrame) -> list[dict]:
    games = games_df.copy()
    if "game_type" in games.columns:
        games = games[games["game_type"] == 3].copy()
    games = games.dropna(subset=["home_score", "away_score"]).sort_values(["season", "date", "id"])
    games["pair"] = games.apply(
        lambda r: tuple(sorted((int(r["home_team_id"]), int(r["away_team_id"])))),
        axis=1,
    )

    first_rows = (
        playoff_df.sort_values(["season", "date", "game_id"])
        .drop_duplicates(subset=["season", "home_team_id", "away_team_id", "series_game_number"], keep="first")
    )

    out = []
    for (season, pair), g in games.groupby(["season", "pair"]):
        g = g.sort_values(["date", "id"]).reset_index(drop=True)
        team1, team2 = pair
        wins = {team1: 0, team2: 0}
        for _, row in g.iterrows():
            if int(row["home_score"]) > int(row["away_score"]):
                wins[int(row["home_team_id"])] += 1
            else:
                wins[int(row["away_team_id"])] += 1

        if max(wins.values()) < 4:
            continue

        first_game = g.iloc[0]
        seed_row = first_rows[
            (first_rows["game_id"] == first_game["id"])
        ]
        if seed_row.empty:
            continue
        seed_diff = seed_row["seed_diff"].iloc[0]
        if pd.isna(seed_diff):
            continue

        if float(seed_diff) < 0:
            team_a = int(first_game["home_team_id"])
            team_b = int(first_game["away_team_id"])
            seed_diff_a_home = float(seed_diff)
        else:
            team_a = int(first_game["away_team_id"])
            team_b = int(first_game["home_team_id"])
            seed_diff_a_home = -float(seed_diff)

        team_a_wins = wins[team_a]
        team_b_wins = wins[team_b]
        if team_a_wins == team_b_wins:
            continue

        out.append(
            {
                "season": int(season),
                "start_date": first_game["date"],
                "team_a_id": team_a,
                "team_b_id": team_b,
                "seed_diff_a_home": float(seed_diff_a_home),
                "team_a_won": int(team_a_wins > team_b_wins),
                "n_games": int(team_a_wins + team_b_wins),
            }
        )

    return out


def _predict_playoff_home_prob(payload: dict, rows: pd.DataFrame) -> np.ndarray:
    model = payload["model"]
    scaler = payload.get("scaler")
    X = fill_features(rows[PLAYOFF_FEATURE_COLS]).values
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]


def run_series_backtest(
    *,
    n_sims: int = 20000,
    seed: int = 42,
    min_train_seasons: int = 2,
    max_goals: int = 10,
):
    print("=" * 72)
    print("  PuckZone v2.0 — Playoff Series Simulation Backtest")
    print("=" * 72)

    ctx = DataContext.from_supabase()
    reg_df = build_features(use_materialized=True)
    playoff_df = build_playoff_features(use_materialized=True)
    if reg_df.empty or playoff_df.empty:
        print("ERROR: missing feature rows.")
        return pd.DataFrame()

    score_cols = (
        ctx.games[["id", "home_score", "away_score", "game_type"]]
        .rename(columns={"id": "game_id"})
        .drop_duplicates("game_id")
    )
    reg_df = reg_df.merge(score_cols, on="game_id", how="left")
    reg_df = reg_df[(reg_df["game_type"] == 2)].dropna(subset=["home_score", "away_score"]).copy()

    series_rows = _completed_series(playoff_df, ctx.games)
    if not series_rows:
        print("No completed playoff series found.")
        return pd.DataFrame()

    seasons = sorted(reg_df["season"].unique())
    goals_by_season: dict[int, BivariatePoissonGoalsModel] = {}
    for i, season in enumerate(seasons):
        train_seasons = seasons[:i]
        if len(train_seasons) < min_train_seasons:
            continue
        train_mask = reg_df["season"].isin(train_seasons)
        X_train = fill_features(reg_df.loc[train_mask, FEATURE_COLS])
        model = BivariatePoissonGoalsModel(use_shared_lambda3=True)
        model.fit(
            X_train,
            reg_df.loc[train_mask, "home_score"].values,
            reg_df.loc[train_mask, "away_score"].values,
        )
        goals_by_season[int(season)] = model

    playoff_payload = None
    try:
        with open("playoff_model.pkl", "rb") as f:
            playoff_payload = pickle.load(f)
    except Exception:
        pass

    y_true, p_goals, p_playoff = [], [], []
    rows_out = []
    h2h_source = ctx.games
    for s in sorted(series_rows, key=lambda r: (r["season"], r["start_date"])):
        season = int(s["season"])
        goals_model = goals_by_season.get(season)
        if goals_model is None:
            continue

        row_a_home = build_feature_row(
            home_team_id=s["team_a_id"],
            away_team_id=s["team_b_id"],
            as_of_date=s["start_date"],
            ctx=ctx,
            season=season,
            h2h_games_df=h2h_source,
        )
        row_b_home = build_feature_row(
            home_team_id=s["team_b_id"],
            away_team_id=s["team_a_id"],
            as_of_date=s["start_date"],
            ctx=ctx,
            season=season,
            h2h_games_df=h2h_source,
        )
        if row_a_home is None or row_b_home is None:
            continue

        Xa = fill_features(pd.DataFrame([row_a_home])[FEATURE_COLS])
        Xb = fill_features(pd.DataFrame([row_b_home])[FEATURE_COLS])
        la_h, la_a, la3 = goals_model.predict_rates(Xa)
        lb_h, lb_a, lb3 = goals_model.predict_rates(Xb)
        sim_out = series_win_prob_from_lambdas(
            lambdas_a_home=(float(la_h[0]), float(la_a[0]), float(la3[0])),
            lambdas_b_home=(float(lb_h[0]), float(lb_a[0]), float(lb3[0])),
            n_sims=n_sims,
            rng=np.random.default_rng(seed + season + int(s["team_a_id"]) + int(s["team_b_id"])),
            max_goals=max_goals,
        )
        prob_goals = float(sim_out["p_team_a_wins"])

        prob_playoff = np.nan
        if playoff_payload is not None:
            ra = dict(row_a_home)
            rb = dict(row_b_home)
            ra["series_game_number"] = 1
            ra["home_series_wins"] = 0
            ra["away_series_wins"] = 0
            ra["seed_diff"] = s["seed_diff_a_home"]
            rb["series_game_number"] = 1
            rb["home_series_wins"] = 0
            rb["away_series_wins"] = 0
            rb["seed_diff"] = -s["seed_diff_a_home"]
            p_home_a, p_home_b = _predict_playoff_home_prob(playoff_payload, pd.DataFrame([ra, rb]))
            prob_playoff = float(
                simulate_series(
                    (float(p_home_a), float(p_home_b)),
                    n_sims=n_sims,
                    rng=np.random.default_rng(seed + 10_000 + season + int(s["team_a_id"]) + int(s["team_b_id"])),
                )["p_team_a_wins"]
            )

        y_true.append(int(s["team_a_won"]))
        p_goals.append(prob_goals)
        if not np.isnan(prob_playoff):
            p_playoff.append(prob_playoff)
        rows_out.append(
            {
                **s,
                "p_team_a_wins_goals_mc": prob_goals,
                "p_team_a_wins_playoff_baseline": prob_playoff,
            }
        )

    results = pd.DataFrame(rows_out)
    if results.empty:
        print("No series evaluated (insufficient training seasons or missing rows).")
        return pd.DataFrame()

    y_arr = np.asarray(y_true, dtype=int)
    goals_probs = np.asarray(p_goals, dtype=float)
    goals_metrics = compute_metrics(goals_probs, y_arr)

    print("\nEvaluated series:", len(results))
    print("\n--- Series-level metrics: goals MC simulation ---")
    print(
        f"  Accuracy: {goals_metrics['accuracy']:.3f} | "
        f"LogLoss: {goals_metrics['log_loss']:.4f} | "
        f"Brier: {goals_metrics['brier']:.4f} | "
        f"ROC AUC: {goals_metrics['roc_auc']:.4f}"
    )
    print_probability_metrics(goals_probs, y_arr)

    if playoff_payload is not None and len(p_playoff) == len(y_arr):
        playoff_probs = np.asarray(p_playoff, dtype=float)
        playoff_metrics = compute_metrics(playoff_probs, y_arr)
        print("\n--- Series-level metrics: playoff classifier baseline ---")
        print(
            f"  Accuracy: {playoff_metrics['accuracy']:.3f} | "
            f"LogLoss: {playoff_metrics['log_loss']:.4f} | "
            f"Brier: {playoff_metrics['brier']:.4f} | "
            f"ROC AUC: {playoff_metrics['roc_auc']:.4f}"
        )
        print_probability_metrics(playoff_probs, y_arr)
    else:
        print("\nPlayoff classifier baseline unavailable; skipping baseline comparison.")

    return results


def _parse_args():
    p = argparse.ArgumentParser(description="Series-level playoff simulation backtest.")
    p.add_argument("--n-sims", type=int, default=20000, metavar="N", help="Monte Carlo simulations per series.")
    p.add_argument("--seed", type=int, default=42, metavar="SEED", help="Base RNG seed.")
    p.add_argument("--min-train-seasons", type=int, default=2, metavar="N", help="Minimum training seasons.")
    p.add_argument("--max-goals", type=int, default=10, metavar="N", help="Score-matrix max goals cap.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_series_backtest(
        n_sims=args.n_sims,
        seed=args.seed,
        min_train_seasons=args.min_train_seasons,
        max_goals=args.max_goals,
    )
