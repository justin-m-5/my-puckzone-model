# team_stats.py

import pandas as pd
from db import supabase, fetch_all
from features.games import get_games


def get_team_stats(games_df=None):
    """Fetch game team stats with game dates.
    Pass games_df to include playoff games in the date join (used during live prediction).
    """
    query = supabase.table("game_team_stats") \
        .select("game_id, team_id, is_home, pp_pctg, faceoff_winning_pctg, sog, hits, blocked_shots")
    df = pd.DataFrame(fetch_all("game_team_stats", query))

    if games_df is None:
        games_df = get_games()
    games = games_df[["id", "date"]].rename(columns={"id": "game_id"})
    df = df.merge(games, on="game_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team_id", "date"]).reset_index(drop=True)
    return df


def build_team_stats_rolling(team_stats_df, window=10):
    """
    For each team + game, compute rolling averages of key stats
    over the previous `window` games.
    Returns dict keyed by (game_id, team_id) -> dict of rolling stats
    """
    cols = ["pp_pctg", "faceoff_winning_pctg", "sog", "hits", "blocked_shots"]
    lookup = {}
    for team_id, group in team_stats_df.groupby("team_id"):
        group = group.sort_values("date").reset_index(drop=True)
        for col in cols:
            group[f"rolling_{col}"] = (
                group[col].shift(1).rolling(window, min_periods=3).mean()
            )
        for _, row in group.iterrows():
            entry = {}
            for col in cols:
                val = row[f"rolling_{col}"]
                entry[col] = val if pd.notna(val) else None
            lookup[(row["game_id"], team_id)] = entry
    return lookup


def get_team_stats_for_game(team_stats_lookup, game_id, team_id):
    """Look up rolling team stats for a team in a game."""
    return team_stats_lookup.get((game_id, team_id), {})