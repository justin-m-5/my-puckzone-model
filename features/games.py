# games.py

import pandas as pd
from db import supabase, fetch_all


def get_games():
    query = supabase.table("games").select("id, season, date, home_team_id, away_team_id, home_score, away_score").eq("game_type", 2).in_("game_state", ["OFF", "FINAL"])
    return pd.DataFrame(fetch_all("games", query))

def build_rest_days_lookup(games_df):
    """
    For each team + game, compute how many days rest they had
    since their previous game.
    Returns dict keyed by (game_id, team_id) -> rest_days
    """
    home = games_df[["id", "date", "home_team_id"]].rename(columns={"home_team_id": "team_id"})
    away = games_df[["id", "date", "away_team_id"]].rename(columns={"away_team_id": "team_id"})
    all_games = pd.concat([home, away]).sort_values(["team_id", "date"]).reset_index(drop=True)

    all_games["prev_date"] = all_games.groupby("team_id")["date"].shift(1)
    all_games["rest_days"] = (pd.to_datetime(all_games["date"]) - pd.to_datetime(all_games["prev_date"])).dt.days

    lookup = {}
    for _, row in all_games.iterrows():
        lookup[(row["id"], row["team_id"])] = row["rest_days"]
    return lookup

def build_h2h_lookup(games_df):
    """
    For each game, compute H2H record between the two teams
    from all prior meetings. Returns dict keyed by game_id.
    """
    games_df = games_df.sort_values("date").reset_index(drop=True)
    lookup = {}

    for idx, game in games_df.iterrows():
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]
        game_date = game["date"]

        # all prior meetings between these two teams (either direction)
        prior = games_df[(games_df["date"] < game_date) & (((games_df["home_team_id"] == home_id) & (games_df["away_team_id"] == away_id)) | ((games_df["home_team_id"] == away_id) & (games_df["away_team_id"] == home_id)))].tail(10)  # last 10 meetings

        if len(prior) == 0:
            lookup[game["id"]] = {
                "h2h_home_win_pctg": None,
            }
            continue

        # did the current home team win each prior meeting?
        home_wins = sum(
            1 for _, g in prior.iterrows()
            if (g["home_team_id"] == home_id and g["home_score"] > g["away_score"]) or (g["away_team_id"] == home_id and g["away_score"] > g["home_score"])
        )

        lookup[game["id"]] = {
            "h2h_home_win_pctg": home_wins / len(prior)
        }

    return lookup