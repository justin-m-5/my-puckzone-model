# scripts/predict/builder.py
"""
Live serving feature builder — thin wrapper over features/pipeline.py.

Assembles the feature row for a single matchup using only data available
strictly before game_date (point-in-time correct).

WHY this is now a wrapper
-------------------------
v1.x computed features inline here, diverging from the training builders in
two ways:
  1. Goalie rolling: used tail(ROLL_WINDOW).mean() over all available data,
     not shift(1).rolling() like training.
  2. Advanced rolling: same tail() issue, with no as_of_date cutoff.

Both skews are fixed by delegating to features/pipeline.py::build_feature_row,
which uses the same shift(1)-equivalent filtering (date < as_of_date) as the
training batch builder.
"""

import datetime
import pandas as pd

from features.pipeline import DataContext, build_feature_row
from features.playoffs import get_series_context


def build_prediction_row(home_team_id, away_team_id, game_date, is_playoff):
    """
    Pull live feature data from Supabase for the given matchup and return
    (feature_row_dict, debug_info_dict).

    Parameters
    ----------
    home_team_id : int
    away_team_id : int
    game_date    : datetime.date
    is_playoff   : bool

    Returns
    -------
    (row, debug)  where ``row`` is ready for fill_features + model.predict_proba,
                  and ``debug`` is a dict with human-readable display values.
    """
    print("\nFetching data from Supabase...")
    ctx = DataContext.from_supabase()

    season_year = game_date.year if game_date.month >= 10 else game_date.year - 1
    season = int(f"{season_year}{season_year + 1}")

    # build_feature_row computes all features using only data before game_date.
    row = build_feature_row(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        as_of_date=game_date,
        ctx=ctx,
        game_id=None,   # serving mode: use most recent starter, not actual starter
        season=season,
    )

    if row is None:
        print("ERROR: Could not build feature row — standings missing for one or both teams.")
        exit(1)

    # --- playoff series context (extra display info, not model features) ---
    series = None
    if is_playoff:
        try:
            series = get_series_context(home_team_id, away_team_id, game_date)
        except Exception:
            series = None

    # --- debug / display values ---
    home_std_gp = row.get("home_games_played", 0)
    away_std_gp = row.get("away_games_played", 0)

    # Reconstruct standings-based record strings for display.
    # We don't store W/L/OTL directly in the feature row so pull from ctx.
    from features.standings import get_latest_standings_before
    home_std = get_latest_standings_before(ctx.standings, home_team_id, season, game_date)
    away_std = get_latest_standings_before(ctx.standings, away_team_id, season, game_date)

    def record_str(std):
        if std is None:
            return "N/A"
        w = std.get("wins") or 0
        l = std.get("losses") or 0
        otl = std.get("ot_losses") or 0
        return f"{w}-{l}-{otl}"

    home_gf_pg = row.get("home_goals_for_per_game", 0.0)
    away_gf_pg = row.get("away_goals_for_per_game", 0.0)

    debug = {
        "home_sv":   row.get("home_goalie_sv_pctg"),
        "away_sv":   row.get("away_goalie_sv_pctg"),
        "home_rest": row.get("home_rest_days"),
        "away_rest": row.get("away_rest_days"),
        "h2h":       row.get("h2h_home_win_pctg"),
        "home_elo":  row.get("home_elo"),
        "away_elo":  row.get("away_elo"),
        "home_record": record_str(home_std),
        "away_record": record_str(away_std),
        "home_gf_pg": home_gf_pg,
        "away_gf_pg": away_gf_pg,
        "series": series,
    }

    return row, debug
