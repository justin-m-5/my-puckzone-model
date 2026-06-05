# features/training/playoff_builder.py
"""
Builds training features for playoff games (base FEATURE_COLS + 4 playoff cols).
Now also includes the rolling advanced shot-share diffs (Corsi / xG / high-danger)
from features/advanced.py, matching the regular-season builder.

The 2020 "bubble" play-in qualifiers (round_number == 0) and round-robin seeding
games (no playoff_series row) are EXCLUDED.
"""

import pandas as pd
from db import supabase, fetch_all
from features.games import get_all_games, build_h2h_lookup, build_rest_days_lookup
from features.standings import get_standings, get_latest_standings_before
from features.goalies import (
    get_goalie_stats,
    build_goalie_rolling,
    get_goalie_sv_for_game,
    get_goalie_advanced_stats,
    build_gsax_rolling,
    get_goalie_gsax_for_game,
)
from features.team_stats import get_team_stats, build_team_stats_rolling, get_team_stats_for_game
from features.elo import build_elo_lookup, STARTING_ELO
from features.advanced import build_advanced_rolling, get_advanced_for_game


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
    print("Loading all games (regular season + playoffs)...")
    all_games = get_all_games()
    all_games["date"] = pd.to_datetime(all_games["date"]).dt.date
    print(f"  {len(all_games)} total completed games")

    playoff_games = all_games[all_games["game_type"] == 3].copy()
    print(f"  {len(playoff_games)} playoff games")

    if len(playoff_games) == 0:
        print("ERROR: No playoff games found. Check game_type=3 rows exist in Supabase.")
        return pd.DataFrame()

    rest_lookup = build_rest_days_lookup(all_games)

    print("Loading standings...")
    standings = get_standings()
    print(f"  {len(standings)} standings snapshots")

    print("Loading goalie stats (regular season + playoffs)...")
    goalie_df = get_goalie_stats(games_df=all_games)
    goalie_lookup = build_goalie_rolling(goalie_df)
    print(f"  {len(goalie_df)} goalie starts")
    gsax_df = get_goalie_advanced_stats()
    gsax_lookup = build_gsax_rolling(gsax_df)
    print(f"  {len(gsax_df)} goalie advanced rows, {len(gsax_lookup)} rolling GSAx entries")

    print("Loading team stats (regular season + playoffs)...")
    team_stats_df = get_team_stats(games_df=all_games)
    team_stats_lookup = build_team_stats_rolling(team_stats_df)

    print("Building H2H lookup...")
    h2h_lookup = build_h2h_lookup(all_games)

    print("Building Elo lookup...")
    elo_lookup = build_elo_lookup(all_games)

    print("Building advanced metrics (Corsi / xG / high-danger)...")
    advanced_lookup = build_advanced_rolling(games_df=all_games)

    print("Loading playoff series seeds...")
    seeds_lookup = _get_seeds_lookup()

    print("Loading playoff series rounds...")
    round_lookup = _get_series_round_lookup()

    print("Building series state lookup...")
    series_lookup = _build_series_state_lookup(playoff_games)

    def bracket_year(season):
        return int(str(season)[4:])

    rows = []
    skipped = 0
    excluded_bubble = 0

    print("Building features for each playoff game...")
    for _, game in playoff_games.iterrows():
        by = bracket_year(game["season"])

        # Exclude 2020 bubble play-in (round 0) + round-robin (no series row -> None)
        series_round = round_lookup.get(
            (by, frozenset((int(game["home_team_id"]), int(game["away_team_id"]))))
        )
        if series_round is None or series_round == 0:
            excluded_bubble += 1
            continue

        home = get_latest_standings_before(standings, game["home_team_id"], game["season"], game["date"])
        away = get_latest_standings_before(standings, game["away_team_id"], game["season"], game["date"])
        if home is None or away is None:
            skipped += 1
            continue

        home_win = 1 if game["home_score"] > game["away_score"] else 0

        home_sv = get_goalie_sv_for_game(goalie_df, goalie_lookup, game["id"], game["home_team_id"], True)
        away_sv = get_goalie_sv_for_game(goalie_df, goalie_lookup, game["id"], game["away_team_id"], False)
        home_gsax = get_goalie_gsax_for_game(goalie_df, gsax_lookup, game["id"], game["home_team_id"], True)
        away_gsax = get_goalie_gsax_for_game(goalie_df, gsax_lookup, game["id"], game["away_team_id"], False)

        home_rest = rest_lookup.get((game["id"], game["home_team_id"]), None)
        away_rest = rest_lookup.get((game["id"], game["away_team_id"]), None)

        home_ts = get_team_stats_for_game(team_stats_lookup, game["id"], game["home_team_id"])
        away_ts = get_team_stats_for_game(team_stats_lookup, game["id"], game["away_team_id"])

        home_adv = get_advanced_for_game(advanced_lookup, game["id"], game["home_team_id"])
        away_adv = get_advanced_for_game(advanced_lookup, game["id"], game["away_team_id"])

        home_pk = (1 - away_ts.get("pp_pctg")) if away_ts.get("pp_pctg") is not None else None
        away_pk = (1 - home_ts.get("pp_pctg")) if home_ts.get("pp_pctg") is not None else None

        home_gf_pg = (home["goal_for"] or 0) / max(home["games_played"], 1)
        home_ga_pg = (home["goal_against"] or 0) / max(home["games_played"], 1)
        away_gf_pg = (away["goal_for"] or 0) / max(away["games_played"], 1)
        away_ga_pg = (away["goal_against"] or 0) / max(away["games_played"], 1)

        home_home_win_pctg = (home["home_wins"] or 0) / max((home["home_wins"] or 0) + (home["home_losses"] or 0), 1)
        away_road_win_pctg = (away["road_wins"] or 0) / max((away["road_wins"] or 0) + (away["road_losses"] or 0), 1)

        h2h = h2h_lookup.get(game["id"], {})
        elo = elo_lookup.get(game["id"], {})

        series = series_lookup.get(game["id"], {})
        home_seed = seeds_lookup.get((by, game["home_team_id"]))
        away_seed = seeds_lookup.get((by, game["away_team_id"]))
        seed_diff = (home_seed - away_seed) if (home_seed is not None and away_seed is not None) else None

        row = {
            "game_id": game["id"],
            "season": game["season"],
            "date": game["date"],
            "home_team_id": game["home_team_id"],
            "away_team_id": game["away_team_id"],
            "target": home_win,

            "home_point_pctg": home["point_pctg"] or 0.5,
            "home_win_pctg": home["win_pctg"] or 0.5,
            "home_reg_win_pctg": home["regulation_win_pctg"] or 0.5,
            "home_goal_diff": home["goal_differential"] or 0,
            "home_l10_points": home["l10_points"] or 0,
            "home_goalie_sv_pctg": home_sv,
            "home_goalie_gsax": home_gsax,
            "home_rest_days": home_rest,
            "home_is_b2b": 1 if home_rest == 1 else 0,
            "home_pp_pctg": home_ts.get("pp_pctg"),
            "home_faceoff_pctg": home_ts.get("faceoff_winning_pctg"),
            "home_sog": home_ts.get("sog"),
            "home_hits": home_ts.get("hits"),
            "home_blocked_shots": home_ts.get("blocked_shots"),

            "away_point_pctg": away["point_pctg"] or 0.5,
            "away_win_pctg": away["win_pctg"] or 0.5,
            "away_reg_win_pctg": away["regulation_win_pctg"] or 0.5,
            "away_goal_diff": away["goal_differential"] or 0,
            "away_l10_points": away["l10_points"] or 0,
            "away_goalie_sv_pctg": away_sv,
            "away_goalie_gsax": away_gsax,
            "away_rest_days": away_rest,
            "away_is_b2b": 1 if away_rest == 1 else 0,
            "away_pp_pctg": away_ts.get("pp_pctg"),
            "away_faceoff_pctg": away_ts.get("faceoff_winning_pctg"),
            "away_sog": away_ts.get("sog"),
            "away_hits": away_ts.get("hits"),
            "away_blocked_shots": away_ts.get("blocked_shots"),

            "diff_point_pctg": (home["point_pctg"] or 0.5) - (away["point_pctg"] or 0.5),
            "diff_goal_diff": (home["goal_differential"] or 0) - (away["goal_differential"] or 0),
            "diff_l10_points": (home["l10_points"] or 0) - (away["l10_points"] or 0),
            "diff_points": (home["points"] or 0) - (away["points"] or 0),
            "diff_goalie_sv_pctg": (home_sv or 0) - (away_sv or 0),
            "diff_goalie_gsax": (home_gsax or 0) - (away_gsax or 0),
            "rest_advantage": (home_rest or 2) - (away_rest or 2),
            "diff_pp_pctg": (home_ts.get("pp_pctg") or 0) - (away_ts.get("pp_pctg") or 0),
            "diff_pk_pctg": (home_pk or 0) - (away_pk or 0),
            "diff_faceoff_pctg": (home_ts.get("faceoff_winning_pctg") or 0) - (away_ts.get("faceoff_winning_pctg") or 0),
            "diff_sog": (home_ts.get("sog") or 0) - (away_ts.get("sog") or 0),

            # advanced shot-share diffs
            "diff_cf_pct": (home_adv.get("cf_pct") or 0.5) - (away_adv.get("cf_pct") or 0.5),
            "diff_xgf_pct": (home_adv.get("xgf_pct") or 0.5) - (away_adv.get("xgf_pct") or 0.5),
            "diff_hdcf_pct": (home_adv.get("hdcf_pct") or 0.5) - (away_adv.get("hdcf_pct") or 0.5),
            "diff_cf_pct_5v5": (home_adv.get("cf_pct_5v5") or 0.5) - (away_adv.get("cf_pct_5v5") or 0.5),
            "diff_xgf_pct_5v5": (home_adv.get("xgf_pct_5v5") or 0.5) - (away_adv.get("xgf_pct_5v5") or 0.5),
            "diff_hdcf_pct_5v5": (home_adv.get("hdcf_pct_5v5") or 0.5) - (away_adv.get("hdcf_pct_5v5") or 0.5),

            "home_home_win_pctg": home_home_win_pctg,
            "away_road_win_pctg": away_road_win_pctg,
            "diff_home_road_pctg": home_home_win_pctg - away_road_win_pctg,
            "h2h_home_win_pctg": h2h.get("h2h_home_win_pctg"),
            "home_elo": elo.get("home_elo", STARTING_ELO),
            "away_elo": elo.get("away_elo", STARTING_ELO),
            "elo_diff": elo.get("elo_diff", 0),

            # playoff-specific
            "series_game_number": series.get("series_game_number"),
            "home_series_wins": series.get("home_series_wins"),
            "away_series_wins": series.get("away_series_wins"),
            "seed_diff": seed_diff,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(
        f"  Built {len(df)} playoff feature rows "
        f"(excluded {excluded_bubble} bubble play-in/round-robin, "
        f"skipped {skipped} missing standings)"
    )
    return df