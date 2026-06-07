import argparse
import pandas as pd

from db import upsert_rows
from features.pipeline import (
    DataContext,
    build_features_batch,
    _advanced_as_of,
    _goalie_gsax_as_of,
    _goalie_sv_as_of,
    _standings_as_of,
    _team_stats_as_of,
)
from features.training.playoff_builder import attach_playoff_columns
from models.game import FEATURE_COLS
from models.playoff import PLAYOFF_EXTRA_COLS
from features.elo import build_elo_lookup

REGULAR_SEASON_GAME_TYPE = 2
PLAYOFF_GAME_TYPE = 3
FEATURE_VERSION = "v2.0"
CHUNK_SIZE = 500


def _rows_for_game_type(ctx: DataContext, game_type: int) -> pd.DataFrame:
    base_df = build_features_batch(ctx, game_type=game_type)
    if base_df.empty:
        return base_df
    if game_type == PLAYOFF_GAME_TYPE:
        base_df = attach_playoff_columns(base_df, ctx)
    else:
        for col in PLAYOFF_EXTRA_COLS:
            base_df[col] = None
    base_df["game_type"] = game_type
    return base_df


def _materialized_contract(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "game_id",
        "season",
        "date",
        "game_type",
        "home_team_id",
        "away_team_id",
        "target",
        "feature_version",
    ] + FEATURE_COLS + PLAYOFF_EXTRA_COLS
    return df.reindex(columns=cols)


def build_materialized_rows(
    ctx: DataContext,
    *,
    limit: int | None = None,
    game_type: str = "all",
) -> list[dict]:
    """Build rows for model_game_features using game_type-specific pipelines."""
    frames = []
    if game_type in ("2", "all"):
        frames.append(_rows_for_game_type(ctx, REGULAR_SEASON_GAME_TYPE))
    if game_type in ("3", "all"):
        frames.append(_rows_for_game_type(ctx, PLAYOFF_GAME_TYPE))
    if not frames:
        return []

    non_empty_frames = [f for f in frames if not f.empty]
    if not non_empty_frames:
        return []
    df = pd.concat(non_empty_frames, ignore_index=True)

    df["feature_version"] = FEATURE_VERSION
    df = _materialized_contract(df).sort_values(["date", "game_id"]).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return df.to_dict(orient="records")


def _latest_goalie_id_as_of(ctx: DataContext, team_id: int, as_of_date):
    prior = ctx.goalie_df[
        (ctx.goalie_df["team_id"] == team_id)
        & (ctx.goalie_df["date"] < as_of_date)
    ].sort_values("date")
    if prior.empty:
        return None
    return int(prior.iloc[-1]["player_id"])


def build_form_snapshot_rows(ctx: DataContext, materialized_rows: list[dict]):
    """Build team/goalie snapshot rows keyed by (entity_id, as_of_date)."""
    if not materialized_rows:
        return [], []

    game_df = pd.DataFrame(materialized_rows)
    if game_df.empty:
        return [], []

    if "date" in game_df.columns:
        game_df["date"] = pd.to_datetime(game_df["date"], errors="coerce").dt.date

    elo_lookup = build_elo_lookup(ctx.games)
    season_by_game = {
        int(g["id"]): int(g["season"])
        for _, g in ctx.games.iterrows()
    }
    team_snapshots = {}
    goalie_snapshots = {}

    for _, row in game_df.iterrows():
        game_id = int(row["game_id"])
        as_of_date = row["date"]
        season = season_by_game.get(game_id, int(row["season"]))
        elo_entry = elo_lookup.get(game_id, {})

        for side in ("home", "away"):
            team_id = int(row[f"{side}_team_id"])
            std = _standings_as_of(ctx.standings, team_id, season, as_of_date) or {}
            team_stats = _team_stats_as_of(ctx.team_stats_df, team_id, as_of_date)
            advanced = _advanced_as_of(ctx.advanced_df, team_id, as_of_date)

            team_snapshots[(team_id, as_of_date)] = {
                "team_id": team_id,
                "as_of_date": as_of_date,
                "season": season,
                "point_pctg": std.get("point_pctg"),
                "win_pctg": std.get("win_pctg"),
                "reg_win_pctg": std.get("regulation_win_pctg"),
                "goal_diff": std.get("goal_differential"),
                "l10_points": std.get("l10_points"),
                "pp_pctg": team_stats.get("pp_pctg"),
                "faceoff_pctg": team_stats.get("faceoff_winning_pctg"),
                "sog": team_stats.get("sog"),
                "hits": team_stats.get("hits"),
                "blocked_shots": team_stats.get("blocked_shots"),
                "cf_pct": advanced.get("cf_pct"),
                "xgf_pct": advanced.get("xgf_pct"),
                "hdcf_pct": advanced.get("hdcf_pct"),
                "cf_pct_5v5": advanced.get("cf_pct_5v5"),
                "xgf_pct_5v5": advanced.get("xgf_pct_5v5"),
                "hdcf_pct_5v5": advanced.get("hdcf_pct_5v5"),
                "elo": elo_entry.get(f"{side}_elo"),
                "feature_version": FEATURE_VERSION,
            }

            player_id = _latest_goalie_id_as_of(ctx, team_id, as_of_date)
            if player_id is not None:
                goalie_snapshots[(player_id, as_of_date)] = {
                    "player_id": player_id,
                    "team_id": team_id,
                    "as_of_date": as_of_date,
                    "goalie_sv_pctg": _goalie_sv_as_of(ctx.goalie_df, team_id, as_of_date),
                    "goalie_gsax": _goalie_gsax_as_of(ctx.gsax_df, ctx.goalie_df, team_id, as_of_date),
                    "feature_version": FEATURE_VERSION,
                }

    return list(team_snapshots.values()), list(goalie_snapshots.values())


def _upsert_in_chunks(table: str, rows: list[dict], on_conflict: str):
    for i in range(0, len(rows), CHUNK_SIZE):
        upsert_rows(table, rows[i:i + CHUNK_SIZE], on_conflict=on_conflict)


def _parse_args():
    parser = argparse.ArgumentParser(description="Materialize leak-safe feature rows into Supabase.")
    parser.add_argument("--dry-run", action="store_true", help="Build rows and print sample counts without writing.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--game-type", choices=["2", "3", "all"], default="all", help="Which game types to materialize.")
    return parser.parse_args()


def main():
    args = _parse_args()

    ctx = DataContext.from_supabase()
    rows = build_materialized_rows(ctx, limit=args.limit, game_type=args.game_type)
    team_rows, goalie_rows = build_form_snapshot_rows(ctx, rows)

    by_type = pd.DataFrame(rows).groupby("game_type").size().to_dict() if rows else {}
    print(f"Built {len(rows)} model_game_features rows: {by_type}")
    print(f"Built {len(team_rows)} team_form_snapshots rows")
    print(f"Built {len(goalie_rows)} goalie_form_snapshots rows")
    if rows:
        print(f"Sample row: {rows[0]}")

    if args.dry_run:
        print("Dry run complete. No writes performed.")
        return

    _upsert_in_chunks("model_game_features", rows, on_conflict="game_id")
    _upsert_in_chunks("team_form_snapshots", team_rows, on_conflict="team_id,as_of_date")
    _upsert_in_chunks("goalie_form_snapshots", goalie_rows, on_conflict="player_id,as_of_date")
    print("Materialization complete.")


if __name__ == "__main__":
    main()
