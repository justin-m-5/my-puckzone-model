from __future__ import annotations

import argparse
import datetime
import pickle
from dataclasses import dataclass

import pandas as pd

from db import fetch_all, supabase
from features.pipeline import DataContext, build_feature_row
from models import fill_features
from models.goals import (
    expected_away_goals,
    expected_home_goals,
    final_home_win_probability,
    most_likely_scoreline,
    p_regulation_tie,
    score_matrix,
)
from scripts.serve.benchmark import build_benchmark_rows
from scripts.serve.odds import OddsProvider, SupabaseOddsProvider, normalize_market_odds
from scripts.serve.writer import write_benchmark_rows, write_serving_rows

CASCADE_STEPS = (
    "refresh_inputs",
    "refresh_features",
    "generate_predictions",
    "write_serving",
    "benchmark",
)


@dataclass
class PipelineContext:
    target_date: datetime.date
    dry_run: bool
    skip_odds: bool
    force_recompute: bool
    feature_version: str
    model_version: str
    run_id: str
    generated_at: str


def _parse_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def _date_range(start_date: datetime.date, end_date: datetime.date) -> list[datetime.date]:
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")
    days = (end_date - start_date).days
    return [start_date + datetime.timedelta(days=i) for i in range(days + 1)]


def _default_model_version(payload: dict, path: str) -> str:
    model_name = payload.get("model_name", "goals_model")
    lam3 = payload.get("lambda3")
    return f"{model_name}|{path}|lambda3={lam3}"


def load_goals_payload(path: str = "goals_model.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def fetch_games_for_date(target_date: datetime.date) -> pd.DataFrame:
    query = (
        supabase.table("games")
        .select("id,season,date,game_type,game_state,home_team_id,away_team_id,home_score,away_score")
        .eq("date", target_date.isoformat())
        .in_("game_type", [2, 3])
    )
    rows = fetch_all("games", query)
    return pd.DataFrame(rows)


def _build_row_for_game(ctx: DataContext, game: pd.Series, target_date: datetime.date) -> dict | None:
    season = int(game["season"])
    is_playoff = int(game.get("game_type", 2)) == 3
    h2h_games_df = (
        ctx.games[ctx.games["game_type"] == 2]
        if (not is_playoff and "game_type" in ctx.games.columns)
        else ctx.games
    )
    row = build_feature_row(
        home_team_id=int(game["home_team_id"]),
        away_team_id=int(game["away_team_id"]),
        as_of_date=target_date,
        ctx=ctx,
        game_id=None,
        season=season,
        h2h_games_df=h2h_games_df,
    )
    if row is None:
        return None
    return row


def generate_serving_rows(
    *,
    pipeline_ctx: PipelineContext,
    games_df: pd.DataFrame,
    ctx: DataContext,
    goals_payload: dict,
) -> list[dict]:
    if games_df.empty:
        return []

    model = goals_payload["model"]
    feature_cols = goals_payload["feature_cols"]

    rows = []
    for _, game in games_df.sort_values("id").iterrows():
        feature_row = _build_row_for_game(ctx, game, pipeline_ctx.target_date)
        if feature_row is None:
            continue

        X = fill_features(pd.DataFrame([feature_row])[feature_cols])
        home_rate, away_rate, lambda3_arr = model.predict_rates(X)
        lambda_home = float(home_rate[0])
        lambda_away = float(away_rate[0])
        lambda3 = float(lambda3_arr[0])
        matrix = score_matrix(lambda_home, lambda_away, lambda3, max_goals=10)

        ml_home, ml_away = most_likely_scoreline(matrix)
        is_finalized = str(game.get("game_state", "")).upper() in {"OFF", "FINAL"}
        home_score = game.get("home_score")
        away_score = game.get("away_score")
        outcome = None
        if is_finalized and home_score is not None and away_score is not None:
            outcome = 1 if float(home_score) > float(away_score) else 0

        rows.append(
            {
                "game_id": int(game["id"]),
                "date": pipeline_ctx.target_date,
                "season": int(game["season"]),
                "home_team_id": int(game["home_team_id"]),
                "away_team_id": int(game["away_team_id"]),
                "feature_version": pipeline_ctx.feature_version,
                "model_version": pipeline_ctx.model_version,
                "run_id": pipeline_ctx.run_id,
                "generated_at": pipeline_ctx.generated_at,
                "home_win_probability": final_home_win_probability(lambda_home, lambda_away, lambda3, max_goals=10),
                "expected_home_goals": expected_home_goals(matrix),
                "expected_away_goals": expected_away_goals(matrix),
                "most_likely_home_score": ml_home,
                "most_likely_away_score": ml_away,
                "regulation_tie_probability": p_regulation_tie(matrix),
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "lambda3": lambda3,
                "is_finalized": is_finalized,
                "data_source": "goals_model+supabase_games",
                "prediction_date": pipeline_ctx.target_date,
                "home_score": home_score,
                "away_score": away_score,
                "home_win_outcome": outcome,
            }
        )
    return rows


def refresh_inputs(target_date: datetime.date, odds_provider: OddsProvider | None, skip_odds: bool) -> list[dict]:
    if skip_odds or odds_provider is None:
        return []
    return odds_provider.fetch_for_date(target_date)


def refresh_features(*, force_recompute: bool, dry_run: bool) -> None:
    if force_recompute:
        mode = "dry-run" if dry_run else "write"
        print(f"Force recompute enabled ({mode}): run scripts.materialize.run separately for full refresh.")


def run_single_date(
    target_date: datetime.date,
    *,
    dry_run: bool,
    skip_odds: bool,
    force_recompute: bool,
    feature_version: str,
    model_version: str | None,
    run_id: str | None,
    odds_provider: OddsProvider | None = None,
    now_fn=lambda: datetime.datetime.now(datetime.timezone.utc),
    step_recorder=None,
) -> dict:
    goals_payload = load_goals_payload()
    mv = model_version or _default_model_version(goals_payload, "goals_model.pkl")
    rid = run_id or f"{target_date.isoformat()}|{feature_version}|{mv}"
    generated_at = now_fn().isoformat()
    pctx = PipelineContext(
        target_date=target_date,
        dry_run=dry_run,
        skip_odds=skip_odds,
        force_recompute=force_recompute,
        feature_version=feature_version,
        model_version=mv,
        run_id=rid,
        generated_at=generated_at,
    )

    def _step(name: str):
        if step_recorder:
            step_recorder(name)

    _step("refresh_inputs")
    odds_rows = refresh_inputs(target_date, odds_provider=odds_provider, skip_odds=skip_odds)

    _step("refresh_features")
    refresh_features(force_recompute=force_recompute, dry_run=dry_run)

    _step("generate_predictions")
    games_df = fetch_games_for_date(target_date)
    ctx = DataContext.from_supabase()
    serving_rows = generate_serving_rows(pipeline_ctx=pctx, games_df=games_df, ctx=ctx, goals_payload=goals_payload)

    _step("write_serving")
    written_serving = write_serving_rows(serving_rows, dry_run=dry_run)

    _step("benchmark")
    benchmark_game_rows, benchmark_daily_rows = [], []
    benchmark_written = (0, 0)
    if not skip_odds:
        odds_by_game = normalize_market_odds(odds_rows)
        benchmark_game_rows, benchmark_daily_rows = build_benchmark_rows(serving_rows, odds_by_game)
        benchmark_written = write_benchmark_rows(benchmark_game_rows, benchmark_daily_rows, dry_run=dry_run)

    return {
        "date": target_date.isoformat(),
        "serving_rows": written_serving,
        "benchmark_game_rows": benchmark_written[0],
        "benchmark_daily_rows": benchmark_written[1],
        "n_predictions": len(serving_rows),
        "n_odds_rows": len(odds_rows),
        "run_id": rid,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Phase 2.3 daily serving + market benchmark orchestrator")
    parser.add_argument("--date", type=_parse_date, default=None, help="Target date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--start-date", type=_parse_date, default=None, help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=_parse_date, default=None, help="Backfill end date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Run full cascade without writing to DB")
    parser.add_argument("--skip-odds", action="store_true", help="Skip odds fetch + market benchmark steps")
    parser.add_argument("--force-recompute", action="store_true", help="Force recompute hooks for feature refresh step")
    parser.add_argument("--feature-version", default="v2.3", help="Serving feature contract version")
    parser.add_argument("--model-version", default=None, help="Explicit model version tag")
    parser.add_argument("--run-id", default=None, help="Explicit run identifier for deterministic reruns")
    return parser.parse_args()


def main():
    args = _parse_args()
    target_date = args.date or datetime.date.today()
    start_date = args.start_date or target_date
    end_date = args.end_date or start_date
    dates = _date_range(start_date, end_date)

    provider = None if args.skip_odds else SupabaseOddsProvider()

    for target_date in dates:
        res = run_single_date(
            target_date,
            dry_run=args.dry_run,
            skip_odds=args.skip_odds,
            force_recompute=args.force_recompute,
            feature_version=args.feature_version,
            model_version=args.model_version,
            run_id=args.run_id,
            odds_provider=provider,
        )
        print(
            f"[{res['date']}] predictions={res['n_predictions']} "
            f"serving_written={res['serving_rows']} benchmark_written=({res['benchmark_game_rows']},{res['benchmark_daily_rows']}) "
            f"odds_rows={res['n_odds_rows']} run_id={res['run_id']}"
        )


if __name__ == "__main__":
    main()
