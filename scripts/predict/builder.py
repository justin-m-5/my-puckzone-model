# scripts/predict/builder.py

import pandas as pd
from db import supabase, fetch_all
from features.standings import get_standings, get_latest_standings_before
from features.goalies import get_goalie_stats, build_goalie_rolling
from features.team_stats import get_team_stats, build_team_stats_rolling
from features.elo import build_elo_lookup, STARTING_ELO
from features.games import get_games, get_all_games, build_rest_days_lookup, build_h2h_lookup
from features.playoffs import get_series_context


def build_prediction_row(home_team_id, away_team_id, game_date, is_playoff):
    """
    Pull live feature data from Supabase for the given matchup
    and return a feature row dict + a debug info dict.
    """
    season_year = game_date.year if game_date.month >= 10 else game_date.year - 1
    season = int(f"{season_year}{season_year + 1}")

    print("\nFetching data from Supabase...")

    # --- standings ---
    print("  Loading standings...")
    standings = get_standings()
    home_std = get_latest_standings_before(standings, home_team_id, season, game_date)
    away_std = get_latest_standings_before(standings, away_team_id, season, game_date)

    if home_std is None or away_std is None:
        print("ERROR: Could not find standings for one or both teams before this date.")
        print(f"  Home standings found: {home_std is not None}")
        print(f"  Away standings found: {away_std is not None}")
        exit(1)

    # --- load all games (regular season + playoffs) up front ---
    # This ensures rest days, H2H, Elo, goalie rolling, and team rolling stats
    # all reflect the current playoff series, not just the regular season.
    print("  Loading games (regular season + playoffs)...")
    games = get_all_games()
    games["date"] = pd.to_datetime(games["date"])

    # --- goalie rolling sv% ---
    print("  Loading goalie stats...")
    goalie_df = get_goalie_stats(games_df=games)
    goalie_lookup = build_goalie_rolling(goalie_df)

    def get_latest_goalie_sv(team_id):
        recent = goalie_df[goalie_df["team_id"] == team_id].sort_values("date")
        if recent.empty:
            return None
        last = recent.iloc[-1]
        return goalie_lookup.get((last["game_id"], last["player_id"]), None)

    home_sv = get_latest_goalie_sv(home_team_id)
    away_sv = get_latest_goalie_sv(away_team_id)

    # --- team rolling stats ---
    print("  Loading team stats...")
    team_stats_df = get_team_stats(games_df=games)
    team_stats_lookup = build_team_stats_rolling(team_stats_df)

    def get_latest_team_stats(team_id):
        recent = team_stats_df[team_stats_df["team_id"] == team_id].sort_values("date")
        if recent.empty:
            return {}
        last = recent.iloc[-1]
        return team_stats_lookup.get((last["game_id"], team_id), {})

    home_ts = get_latest_team_stats(home_team_id)
    away_ts = get_latest_team_stats(away_team_id)

    # --- rest days ---
    print("  Loading rest days...")
    games_date = games.copy()
    games_date["date"] = games_date["date"].dt.date

    def get_rest_days(team_id):
        team_games = games_date[
            (games_date["home_team_id"] == team_id) | (games_date["away_team_id"] == team_id)
        ].sort_values("date")
        past = team_games[team_games["date"] < game_date]
        if past.empty:
            return None
        return (game_date - past.iloc[-1]["date"]).days

    home_rest = get_rest_days(home_team_id)
    away_rest = get_rest_days(away_team_id)

    # --- H2H ---
    print("  Building H2H...")
    prior = games_date[
        (games_date["date"] < game_date) &
        (
            ((games_date["home_team_id"] == home_team_id) & (games_date["away_team_id"] == away_team_id)) |
            ((games_date["home_team_id"] == away_team_id) & (games_date["away_team_id"] == home_team_id))
        )
    ]
    if prior.empty:
        h2h_home_win_pctg = None
    else:
        home_wins = sum(
            1 for _, g in prior.iterrows()
            if (g["home_team_id"] == home_team_id and g["home_score"] > g["away_score"]) or (g["away_team_id"] == home_team_id and g["away_score"] > g["home_score"])
        )
        h2h_home_win_pctg = home_wins / len(prior)

    # --- Elo ---
    print("  Building Elo...")
    elo_lookup = build_elo_lookup(games_date)

    def get_latest_elo(team_id):
        team_games = games_date[
            (games_date["home_team_id"] == team_id) | (games_date["away_team_id"] == team_id)
        ].sort_values("date")
        if team_games.empty:
            return STARTING_ELO
        last_gid = team_games.iloc[-1]["id"]
        entry = elo_lookup.get(last_gid, {})
        if games_date[games_date["id"] == last_gid].iloc[0]["home_team_id"] == team_id:
            return entry.get("home_elo", STARTING_ELO)
        return entry.get("away_elo", STARTING_ELO)

    home_elo = get_latest_elo(home_team_id)
    away_elo = get_latest_elo(away_team_id)

    # --- derived values ---
    home_pk = (1 - away_ts["pp_pctg"]) if away_ts.get("pp_pctg") is not None else None
    away_pk = (1 - home_ts["pp_pctg"]) if home_ts.get("pp_pctg") is not None else None

    home_gf_pg = (home_std["goal_for"] or 0) / max(home_std["games_played"], 1)
    home_ga_pg = (home_std["goal_against"] or 0) / max(home_std["games_played"], 1)
    away_gf_pg = (away_std["goal_for"] or 0) / max(away_std["games_played"], 1)
    away_ga_pg = (away_std["goal_against"] or 0) / max(away_std["games_played"], 1)

    home_home_win_pctg = (home_std["home_wins"] or 0) / max((home_std["home_wins"] or 0) + (home_std["home_losses"] or 0), 1)
    away_road_win_pctg = (away_std["road_wins"] or 0) / max((away_std["road_wins"] or 0) + (away_std["road_losses"] or 0), 1)

    row = {
        "home_point_pctg": home_std["point_pctg"] or 0.5,
        "home_win_pctg": home_std["win_pctg"] or 0.5,
        "home_reg_win_pctg": home_std["regulation_win_pctg"] or 0.5,
        "home_goal_diff": home_std["goal_differential"] or 0,
        "home_l10_points": home_std["l10_points"] or 0,
        "home_goalie_sv_pctg": home_sv,
        "home_rest_days": home_rest,
        "home_is_b2b": 1 if home_rest == 1 else 0,
        "home_pp_pctg": home_ts.get("pp_pctg"),
        "home_faceoff_pctg": home_ts.get("faceoff_winning_pctg"),
        "home_sog": home_ts.get("sog"),
        "home_hits": home_ts.get("hits"),
        "home_blocked_shots": home_ts.get("blocked_shots"),

        "away_point_pctg": away_std["point_pctg"] or 0.5,
        "away_win_pctg": away_std["win_pctg"] or 0.5,
        "away_reg_win_pctg": away_std["regulation_win_pctg"] or 0.5,
        "away_goal_diff": away_std["goal_differential"] or 0,
        "away_l10_points": away_std["l10_points"] or 0,
        "away_goalie_sv_pctg": away_sv,
        "away_rest_days": away_rest,
        "away_is_b2b": 1 if away_rest == 1 else 0,
        "away_pp_pctg": away_ts.get("pp_pctg"),
        "away_faceoff_pctg": away_ts.get("faceoff_winning_pctg"),
        "away_sog": away_ts.get("sog"),
        "away_hits": away_ts.get("hits"),
        "away_blocked_shots": away_ts.get("blocked_shots"),

        "diff_point_pctg": (home_std["point_pctg"] or 0.5) - (away_std["point_pctg"] or 0.5),
        "diff_goal_diff": (home_std["goal_differential"] or 0) - (away_std["goal_differential"] or 0),
        "diff_l10_points": (home_std["l10_points"] or 0) - (away_std["l10_points"] or 0),
        "diff_points": (home_std["points"] or 0) - (away_std["points"] or 0),
        "diff_goalie_sv_pctg": (home_sv or 0) - (away_sv or 0),
        "rest_advantage": (home_rest or 2) - (away_rest or 2),
        "diff_pp_pctg": (home_ts.get("pp_pctg") or 0) - (away_ts.get("pp_pctg") or 0),
        "diff_pk_pctg": (home_pk or 0) - (away_pk or 0),
        "diff_faceoff_pctg": (home_ts.get("faceoff_winning_pctg") or 0) - (away_ts.get("faceoff_winning_pctg") or 0),
        "diff_sog": (home_ts.get("sog") or 0) - (away_ts.get("sog") or 0),
        "diff_goals_for_per_game": home_gf_pg - away_gf_pg,
        "diff_goals_against_per_game": home_ga_pg - away_ga_pg,

        "home_home_win_pctg": home_home_win_pctg,
        "away_road_win_pctg": away_road_win_pctg,
        "diff_home_road_pctg": home_home_win_pctg - away_road_win_pctg,

        "h2h_home_win_pctg": h2h_home_win_pctg,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
    }

    debug = {
        "home_sv": home_sv,
        "away_sv": away_sv,
        "home_rest": home_rest,
        "away_rest": away_rest,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "h2h": h2h_home_win_pctg,
        "home_record": f"{home_std['wins']}W-{home_std['losses']}L",
        "away_record": f"{away_std['wins']}W-{away_std['losses']}L",
        "home_gf_pg": home_gf_pg,
        "away_gf_pg": away_gf_pg,
        "series": None,
    }

    if is_playoff:
        bracket_year = season_year + 1  # e.g. 2025-26 season -> 2026

        # Compute series wins directly from game results — always accurate
        # regardless of whether populate_playoffs.py has been run.
        series_games = games_date[
            (games_date["date"] < game_date) &
            (
                ((games_date["home_team_id"] == home_team_id) & (games_date["away_team_id"] == away_team_id)) |
                ((games_date["home_team_id"] == away_team_id) & (games_date["away_team_id"] == home_team_id))
            ) &
            (games_date.get("game_type", 3) == 3 if "game_type" in games_date.columns else True)
        ]
        home_sw = sum(
            1 for _, g in series_games.iterrows()
            if (g["home_team_id"] == home_team_id and g["home_score"] > g["away_score"]) or
               (g["away_team_id"] == home_team_id and g["away_score"] > g["home_score"])
        )
        away_sw = len(series_games) - home_sw
        series_game_number = home_sw + away_sw + 1

        row["home_series_wins"] = home_sw
        row["away_series_wins"] = away_sw
        row["series_game_number"] = series_game_number

        # Seeding and display info still come from playoff_series table
        series_meta = get_series_context(home_team_id, away_team_id, bracket_year)
        row["seed_diff"] = (
            (series_meta["team_a_seed"] - series_meta["team_b_seed"])
            if (series_meta and series_meta["team_a_seed"] is not None and series_meta["team_b_seed"] is not None)
            else 0
        )

        # Build a combined series dict for display (uses computed wins, not stale table)
        debug["series"] = {
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

    return row, debug