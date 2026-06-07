# features/training/playoff_builder.py
"""
Playoff training feature builder — thin wrapper over features/pipeline.py.

Builds the 51-column base row (same as regular season) plus the 4
playoff-specific columns (series_game_number, home_series_wins,
away_series_wins, seed_diff).  The 2020 bubble play-in qualifiers
(round_number == 0) and round-robin seeding games (no playoff_series row)
are excluded.

All feature computation has moved to the unified pipeline (Phase 2.0 refactor)
so that training and serving share a single code path.  This module is kept
for backward compatibility: everything that imports
``from features.training import build_playoff_features`` continues to work.
"""

import pandas as pd
from db import supabase, fetch_all
from features.pipeline import DataContext, build_features_batch


def _get_seeds_lookup():
    query = supabase.table("playoff_series").select(
        "bracket_year, top_seed_team_id, top_seed_rank, bottom_seed_team_id, bottom_seed_rank"
    )
    rows = fetch_all("playoff_series", query)
    lookup = {}
    for row in rows:
        if row["top_seed_team_id"] and row["top_seed_rank"]:
            lookup[(row["bracket_year"], row["top_seed_team_id"])] = row["top_seed_rank"]
        if row["bottom_seed_team_id"] and row["bottom_seed_rank"]:
            lookup[(row["bracket_year"], row["bottom_seed_team_id"])] = row["bottom_seed_rank"]
    return lookup


def _get_series_round_lookup():
    """(bracket_year, frozenset{team_a, team_b}) -> round_number, to drop bubble games."""
    query = supabase.table("playoff_series").select(
        "bracket_year, round_number, top_seed_team_id, bottom_seed_team_id"
    )
    rows = fetch_all("playoff_series", query)
    lookup = {}
    for row in rows:
        top = row["top_seed_team_id"]
        bottom = row["bottom_seed_team_id"]
        if top and bottom and row["round_number"] is not None:
            lookup[(row["bracket_year"], frozenset((top, bottom)))] = row["round_number"]
    return lookup


def _build_series_state_lookup(playoff_games):
    pg = playoff_games.copy().sort_values("date").reset_index(drop=True)
    pg["team_pair"] = pg.apply(
        lambda r: tuple(sorted([int(r["home_team_id"]), int(r["away_team_id"])])), axis=1
    )
    lookup = {}
    for (season, team_pair), group in pg.groupby(["season", "team_pair"]):
        group = group.sort_values("date").reset_index(drop=True)
        team_a_id, team_b_id = team_pair
        wins = {team_a_id: 0, team_b_id: 0}
        for game_num, (_, game) in enumerate(group.iterrows(), start=1):
            home_id = int(game["home_team_id"])
            away_id = int(game["away_team_id"])
            lookup[game["id"]] = {
                "series_game_number": game_num,
                "home_series_wins": wins[home_id],
                "away_series_wins": wins[away_id],
            }
            if game["home_score"] > game["away_score"]:
                wins[home_id] += 1
            else:
                wins[away_id] += 1
    return lookup


def build_playoff_features():
    """
    Build the playoff training dataset: 51 base features + 4 playoff-specific
    columns (series_game_number, home_series_wins, away_series_wins, seed_diff).
    """
    ctx = DataContext.from_supabase()

    # Filter to playoff games only.
    all_games = ctx.games
    playoff_games = all_games[all_games["game_type"] == 3].copy()
    print(f"  {len(playoff_games)} playoff games found")

    if playoff_games.empty:
        print("ERROR: No playoff games found. Check game_type=3 rows exist in Supabase.")
        return pd.DataFrame()

    # Load playoff-specific lookups.
    print("Loading playoff series seeds and rounds...")
    seeds_lookup = _get_seeds_lookup()
    round_lookup = _get_series_round_lookup()
    series_lookup = _build_series_state_lookup(playoff_games)

    def bracket_year(season):
        return int(str(season)[4:])

    # Build base features for all playoff games using the unified pipeline.
    print("Building base features (via unified pipeline)...")
    base_df = build_features_batch(ctx, game_type=3)

    if base_df.empty:
        return pd.DataFrame()

    # Attach playoff-specific columns.
    rows = []
    excluded_bubble = 0
    for _, row in base_df.iterrows():
        game_id = row["game_id"]
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])
        season = row["season"]
        by = bracket_year(season)

        # Exclude 2020 bubble play-in (round 0) + round-robin (no series row -> None).
        series_round = round_lookup.get(
            (by, frozenset((home_id, away_id)))
        )
        if series_round is None or series_round == 0:
            excluded_bubble += 1
            continue

        series = series_lookup.get(game_id, {})
        home_seed = seeds_lookup.get((by, home_id))
        away_seed = seeds_lookup.get((by, away_id))
        seed_diff = (home_seed - away_seed) if (home_seed is not None and away_seed is not None) else None

        playoff_row = dict(row)
        playoff_row["series_game_number"] = series.get("series_game_number")
        playoff_row["home_series_wins"] = series.get("home_series_wins")
        playoff_row["away_series_wins"] = series.get("away_series_wins")
        playoff_row["seed_diff"] = seed_diff
        rows.append(playoff_row)

    df = pd.DataFrame(rows)
    print(
        f"  Built {len(df)} playoff feature rows "
        f"(excluded {excluded_bubble} bubble play-in/round-robin)"
    )
    return df
