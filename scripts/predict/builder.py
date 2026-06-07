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

from features.pipeline import DataContext, build_feature_row
from features.playoffs import get_series_context

REGULAR_SEASON_GAME_TYPE = 2


def build_prediction_row(home_team_id, away_team_id, game_date, is_playoff, use_materialized=False):
    """
    Pull live feature data from Supabase for the given matchup and return
    (feature_row_dict, debug_info_dict).

    Parameters
    ----------
    home_team_id : int
    away_team_id : int
    game_date    : datetime.date
    is_playoff   : bool
    use_materialized : bool
        Reserved fast-path flag for future snapshot-based serving. Default False
        keeps current live behavior.

    Returns
    -------
    (row, debug)  where ``row`` is ready for fill_features + model.predict_proba,
                  and ``debug`` is a dict with human-readable display values.
    """
    print("\nFetching data from Supabase...")
    if use_materialized:
        # TODO(phase-2.0): optional serving fast path can read team/goalie form
        # snapshots here when explicitly enabled.
        pass
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
        h2h_games_df=(
            ctx.games[ctx.games["game_type"] == REGULAR_SEASON_GAME_TYPE]
            if (not is_playoff and "game_type" in ctx.games.columns)
            else ctx.games
        ),
    )

    if row is None:
        print("ERROR: Could not build feature row — standings missing for one or both teams.")
        exit(1)

    # --- playoff series context (extra display info, not model features) ---
    series = None
    if is_playoff:
        # Playoff bracket year is the season end year (e.g., 2023-24 -> 2024).
        season_year = game_date.year if game_date.month >= 10 else game_date.year - 1
        bracket_year = season_year + 1

        series_games = ctx.games[
            (ctx.games["date"] < game_date)
            & (
                ((ctx.games["home_team_id"] == home_team_id) & (ctx.games["away_team_id"] == away_team_id))
                | ((ctx.games["home_team_id"] == away_team_id) & (ctx.games["away_team_id"] == home_team_id))
            )
            & ((ctx.games["game_type"] == 3) if "game_type" in ctx.games.columns else True)
        ]
        home_sw = int(sum(
            1
            for _, g in series_games.iterrows()
            if (
                (g["home_team_id"] == home_team_id and g["home_score"] > g["away_score"])
                or (g["away_team_id"] == home_team_id and g["away_score"] > g["home_score"])
            )
        ))
        away_sw = len(series_games) - home_sw

        row["home_series_wins"] = home_sw
        row["away_series_wins"] = away_sw
        row["series_game_number"] = home_sw + away_sw + 1

        try:
            series_meta = get_series_context(home_team_id, away_team_id, bracket_year)
        except Exception as exc:
            print(f"Warning: unable to load series context: {exc}")
            series_meta = None

        row["seed_diff"] = (
            (series_meta["team_a_seed"] - series_meta["team_b_seed"])
            if (
                series_meta
                and series_meta.get("team_a_seed") is not None
                and series_meta.get("team_b_seed") is not None
            )
            else 0
        )
        series = {
            "round_number": series_meta["round_number"] if series_meta else None,
            "series_title": series_meta["series_title"] if series_meta else None,
            "series_abbrev": series_meta["series_abbrev"] if series_meta else None,
            "team_a_wins": home_sw,
            "team_b_wins": away_sw,
            "team_a_seed": series_meta["team_a_seed"] if series_meta else None,
            "team_b_seed": series_meta["team_b_seed"] if series_meta else None,
            "series_clinched": series_meta["series_clinched"] if series_meta else False,
            "series_winner_id": series_meta["series_winner_id"] if series_meta else None,
        }
    else:
        row["home_series_wins"] = 0
        row["away_series_wins"] = 0
        row["series_game_number"] = 1
        row["seed_diff"] = 0

    # --- debug / display values ---
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
