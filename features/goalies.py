# goalies.py

import pandas as pd
from db import supabase, fetch_all
from features.games import get_games


def get_goalie_stats(games_df=None):
    """Fetch all starter goalie stats with game dates.
    Pass games_df to include playoff games in the date join (used during live prediction).
    """
    query = supabase.table("game_goalie_stats") \
        .select("game_id, player_id, team_id, is_home, saves, shots_against") \
        .eq("starter", True)
    df = pd.DataFrame(fetch_all("game_goalie_stats", query))

    if games_df is None:
        games_df = get_games()
    games = games_df[["id", "date"]].rename(columns={"id": "game_id"})
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


def get_goalie_advanced_stats(games_df=None):
    """Fetch per-game goalie advanced stats (GSAx) with game dates."""
    try:
        query = supabase.table("game_goalie_advanced_stats") \
            .select("game_id, player_id, team_id, date, gsax")
        df = pd.DataFrame(fetch_all("game_goalie_advanced_stats", query))
    except Exception as e:
        print(f"  (game_goalie_advanced_stats not readable: {e})")
        return pd.DataFrame()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["gsax"] = pd.to_numeric(df["gsax"], errors="coerce")
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    return df


def build_gsax_rolling(gsax_df, window=10):
    """
    For each goalie appearance, compute rolling GSAx mean over the previous
    `window` appearances (not including current game).
    Returns dict keyed by (game_id, player_id) -> rolling_gsax
    """
    lookup = {}
    for player_id, group in gsax_df.groupby("player_id"):
        group = group.sort_values("date").reset_index(drop=True)
        rolling_gsax = group["gsax"].shift(1).rolling(window, min_periods=3).mean()
        for idx, row in group.iterrows():
            gsax = rolling_gsax.iloc[idx]
            if pd.notna(gsax):
                lookup[(row["game_id"], player_id)] = gsax
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


def get_goalie_gsax_for_game(goalie_df, gsax_lookup, game_id, team_id, is_home):
    """Look up rolling GSAx for the starting goalie of a team in a game."""
    row = goalie_df[
        (goalie_df["game_id"] == game_id) &
        (goalie_df["team_id"] == team_id) &
        (goalie_df["is_home"] == is_home)
    ]
    if row.empty:
        return None
    player_id = row.iloc[0]["player_id"]
    return gsax_lookup.get((game_id, player_id), None)