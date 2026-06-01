# standings.py

import pandas as pd
from db import supabase, fetch_all


def get_standings():
    query = supabase.table("team_standings") \
        .select(
            "team_id, season_id, as_of_date, games_played, "
            "point_pctg, win_pctg, regulation_win_pctg, "
            "goal_for, goal_against, goal_differential, "
            "l10_wins, l10_losses, l10_ot_losses, l10_points, "
            "home_wins, home_losses, road_wins, road_losses, "
            "points, wins, losses, ot_losses", "streak_code, streak_count"
        ) \
        .eq("game_type_id", 2)
    df = pd.DataFrame(fetch_all("team_standings", query))
    df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    return df


def get_latest_standings_before(standings_df, team_id, season, game_date):
    """
    For a given team + game date, find the most recent standings snapshot
    taken BEFORE the game was played.
    """
    mask = (
        (standings_df["team_id"] == team_id) &
        (standings_df["season_id"] == season) &
        (standings_df["as_of_date"] < game_date)
    )
    filtered = standings_df[mask].sort_values("as_of_date", ascending=False)
    if filtered.empty:
        return None
    return filtered.iloc[0]