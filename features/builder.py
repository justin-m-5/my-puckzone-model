# builder.py

import pandas as pd
from features.games import get_games, build_rest_days_lookup, build_h2h_lookup
from features.standings import get_standings, get_latest_standings_before
from features.goalies import get_goalie_stats, build_goalie_rolling, get_goalie_sv_for_game
from features.team_stats import get_team_stats, build_team_stats_rolling, get_team_stats_for_game
from features.elo import build_elo_lookup, STARTING_ELO

def build_features():
    print("Loading games...")
    games = get_games()
    games["date"] = pd.to_datetime(games["date"]).dt.date
    print(f"  {len(games)} games loaded")

    rest_lookup = build_rest_days_lookup(games)

    print("Loading standings...")
    standings = get_standings()
    print(f"  {len(standings)} standings snapshots loaded")

    print("Loading goalie stats...")
    goalie_df = get_goalie_stats()
    goalie_lookup = build_goalie_rolling(goalie_df)
    print(f"  {len(goalie_df)} goalie starts loaded, {len(goalie_lookup)} rolling sv% entries")

    print("Loading team stats...")
    team_stats_df = get_team_stats()
    team_stats_lookup = build_team_stats_rolling(team_stats_df)
    print(f"  {len(team_stats_df)} team game rows loaded, {len(team_stats_lookup)} rolling entries")

    print("Building H2H lookup...")
    h2h_lookup = build_h2h_lookup(games)
    print(f"  {len(h2h_lookup)} H2H entries built")

    print("Building Elo lookup...")
    elo_lookup = build_elo_lookup(games)
    print(f"  {len(elo_lookup)} Elo entries built")

    rows = []
    skipped = 0

    print("Building features...")
    for _, game in games.iterrows():
        home = get_latest_standings_before(standings, game["home_team_id"], game["season"], game["date"])
        away = get_latest_standings_before(standings, game["away_team_id"], game["season"], game["date"])

        if home is None or away is None:
            skipped += 1
            continue

        if home["games_played"] == 0 and away["games_played"] == 0:
            skipped += 1
            continue

        home_win = 1 if game["home_score"] > game["away_score"] else 0

        # goalie rolling sv%
        home_sv = get_goalie_sv_for_game(goalie_df, goalie_lookup, game["id"], game["home_team_id"], True)
        away_sv = get_goalie_sv_for_game(goalie_df, goalie_lookup, game["id"], game["away_team_id"], False)

        # rest days
        home_rest = rest_lookup.get((game["id"], game["home_team_id"]), None)
        away_rest = rest_lookup.get((game["id"], game["away_team_id"]), None)

        # team rolling stats
        home_ts = get_team_stats_for_game(team_stats_lookup, game["id"], game["home_team_id"])
        away_ts = get_team_stats_for_game(team_stats_lookup, game["id"], game["away_team_id"])

        # pk% derived from opponent's rolling pp%
        home_pk = (1 - away_ts.get("pp_pctg")) if away_ts.get("pp_pctg") is not None else None
        away_pk = (1 - home_ts.get("pp_pctg")) if home_ts.get("pp_pctg") is not None else None

        # goals per game derived from standings
        home_gf_pg = (home["goal_for"] or 0) / max(home["games_played"], 1)
        home_ga_pg = (home["goal_against"] or 0) / max(home["games_played"], 1)
        away_gf_pg = (away["goal_for"] or 0) / max(away["games_played"], 1)
        away_ga_pg = (away["goal_against"] or 0) / max(away["games_played"], 1)

        # home/away splits
        home_home_win_pctg = (home["home_wins"] or 0) / max((home["home_wins"] or 0) + (home["home_losses"] or 0), 1)
        away_road_win_pctg = (away["road_wins"] or 0) / max((away["road_wins"] or 0) + (away["road_losses"] or 0), 1)

        # H2H
        h2h = h2h_lookup.get(game["id"], {})

        # Elo
        elo = elo_lookup.get(game["id"], {})

        row = {
            "game_id": game["id"],
            "season": game["season"],
            "date": game["date"],
            "home_team_id": game["home_team_id"],
            "away_team_id": game["away_team_id"],
            "target": home_win,

            # standings features
            "home_games_played": home["games_played"] or 0,
            "home_point_pctg": home["point_pctg"] or 0.5,
            "home_win_pctg": home["win_pctg"] or 0.5,
            "home_reg_win_pctg": home["regulation_win_pctg"] or 0.5,
            "home_goal_diff": home["goal_differential"] or 0,
            "home_l10_points": home["l10_points"] or 0,
            "home_points": home["points"] or 0,
            "home_goals_for_per_game": home_gf_pg,
            "home_goals_against_per_game": home_ga_pg,

            "away_games_played": away["games_played"] or 0,
            "away_point_pctg": away["point_pctg"] or 0.5,
            "away_win_pctg": away["win_pctg"] or 0.5,
            "away_reg_win_pctg": away["regulation_win_pctg"] or 0.5,
            "away_goal_diff": away["goal_differential"] or 0,
            "away_l10_points": away["l10_points"] or 0,
            "away_points": away["points"] or 0,
            "away_goals_for_per_game": away_gf_pg,
            "away_goals_against_per_game": away_ga_pg,

            # goalie features
            "home_goalie_sv_pctg": home_sv,
            "away_goalie_sv_pctg": away_sv,

            # rest features
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "home_is_b2b": 1 if home_rest == 1 else 0,
            "away_is_b2b": 1 if away_rest == 1 else 0,
            "rest_advantage": (home_rest or 2) - (away_rest or 2),

            # team rolling stats
            "home_pp_pctg": home_ts.get("pp_pctg"),
            "home_pk_pctg": home_pk,
            "home_faceoff_pctg": home_ts.get("faceoff_winning_pctg"),
            "home_sog": home_ts.get("sog"),
            "home_hits": home_ts.get("hits"),
            "home_blocked_shots": home_ts.get("blocked_shots"),

            "away_pp_pctg": away_ts.get("pp_pctg"),
            "away_pk_pctg": away_pk,
            "away_faceoff_pctg": away_ts.get("faceoff_winning_pctg"),
            "away_sog": away_ts.get("sog"),
            "away_hits": away_ts.get("hits"),
            "away_blocked_shots": away_ts.get("blocked_shots"),

            # differential features
            "diff_point_pctg": (home["point_pctg"] or 0.5) - (away["point_pctg"] or 0.5),
            "diff_goal_diff": (home["goal_differential"] or 0) - (away["goal_differential"] or 0),
            "diff_l10_points": (home["l10_points"] or 0) - (away["l10_points"] or 0),
            "diff_points": (home["points"] or 0) - (away["points"] or 0),
            "diff_goalie_sv_pctg": (home_sv or 0) - (away_sv or 0),
            "diff_pp_pctg": (home_ts.get("pp_pctg") or 0) - (away_ts.get("pp_pctg") or 0),
            "diff_pk_pctg": (home_pk or 0) - (away_pk or 0),
            "diff_faceoff_pctg": (home_ts.get("faceoff_winning_pctg") or 0) - (away_ts.get("faceoff_winning_pctg") or 0),
            "diff_sog": (home_ts.get("sog") or 0) - (away_ts.get("sog") or 0),
            "diff_goals_for_per_game": home_gf_pg - away_gf_pg,
            "diff_goals_against_per_game": home_ga_pg - away_ga_pg,

            # home/away splits
            "home_home_win_pctg": home_home_win_pctg,
            "away_road_win_pctg": away_road_win_pctg,
            "diff_home_road_pctg": home_home_win_pctg - away_road_win_pctg,

            "h2h_home_win_pctg": h2h.get("h2h_home_win_pctg"),

            "home_elo":  elo.get("home_elo", STARTING_ELO),
            "away_elo":  elo.get("away_elo", STARTING_ELO),
            "elo_diff":  elo.get("elo_diff", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Built {len(df)} feature rows ({skipped} games skipped — no prior standings)")
    return df