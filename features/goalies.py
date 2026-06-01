# goalies.py

import pandas as pd
from db import supabase, fetch_all
from features.games import get_games


def get_goalie_stats():
    """Fetch all starter goalie stats with game dates."""
    query = supabase.table("game_goalie_stats") \
        .select("game_id, player_id, team_id, is_home, saves, shots_against") \
        .eq("starter", True)
    df = pd.DataFrame(fetch_all("game_goalie_stats", query))

    games = get_games()[["id", "date"]].rename(columns={"id": "game_id"})
    df = df.merge(games, on="game_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    return df


def build_goalie_rolling(goalie_df, window=10):
    """
    For each goalie start, compute rolling save% over the previous
    `window` starts (not including current game).
    Returns dict keyed by (game_id, player_id) -> rolling_sv_pctg
    """
    lookup = {}
    for player_id, group in goalie_df.groupby("player_id"):
        group = group.sort_values("date").reset_index(drop=True)
        rolling_saves = group["saves"].shift(1).rolling(window, min_periods=3).sum()
        rolling_shots = group["shots_against"].shift(1).rolling(window, min_periods=3).sum()
        rolling_sv = rolling_saves / rolling_shots.replace(0, float("nan"))
        for idx, row in group.iterrows():
            sv = rolling_sv.iloc[idx]
            if pd.notna(sv):
                lookup[(row["game_id"], player_id)] = sv
    return lookup


def get_goalie_sv_for_game(goalie_df, goalie_lookup, game_id, team_id, is_home):
    """Look up rolling sv% for the starting goalie of a team in a game."""
    row = goalie_df[
        (goalie_df["game_id"] == game_id) &
        (goalie_df["team_id"] == team_id) &
        (goalie_df["is_home"] == is_home)
    ]
    if row.empty:
        return None
    player_id = row.iloc[0]["player_id"]
    return goalie_lookup.get((game_id, player_id), None)