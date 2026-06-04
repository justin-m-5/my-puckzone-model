# features/plays.py

import numpy as np
import pandas as pd
from db import supabase, fetch_all
from features.games import get_games


# Ordered list of known shot types for label encoding
SHOT_TYPES = [
    "wrist", "slap", "snap", "backhand",
    "tip-in", "deflection", "bat", "between-legs",
    "poke", "wrap-around",
]
SHOT_TYPE_MAP = {t: i for i, t in enumerate(SHOT_TYPES)}


def get_shot_events():
    """
    Fetch shot-on-goal and goal events from game_plays, joined with
    regular season game metadata (date, season, home_team_id).
    """
    query = supabase.table("game_plays") \
        .select(
            "id, game_id, sort_order, type_desc_key, period, period_type, "
            "situation_code, x_coord, y_coord, shot_type, event_owner_team_id, "
            "shooting_player_id, goalie_in_net_id"
        ) \
        .in_("type_desc_key", ["shot-on-goal", "goal"])

    df = pd.DataFrame(fetch_all("game_plays", query))
    if df.empty:
        return df

    # inner join — only keep shots from regular season games we track
    games = get_games()[["id", "date", "season", "home_team_id"]].rename(columns={"id": "game_id"})
    df = df.merge(games, on="game_id", how="inner")

    return df


def build_xg_features(shot_df):
    """
    Compute xG model features for each shot event.
    Returns the input DataFrame with feature columns appended.

    Features produced:
        shot_distance   — feet from net
        shot_angle      — degrees from center of net (0 = straight on)
        is_behind_net   — shot originated behind the goal line
        shot_type_code  — label-encoded shot type
        is_rebound      — previous event was also a shot-on-goal
        period          — 1-4 (OT capped at 4)
        is_pp           — shooter's team on power play
        is_sh           — shooter's team shorthanded
        is_en           — opponent's net is empty
    """
    df = shot_df.copy()

    # --- shot geometry ---
    # Shots from positive x side attack the right goal (89, 0); negative side attacks left goal (-89, 0).
    # Using signed x to detect behind-net shots (|x| > 89).
    x = df["x_coord"].fillna(0)
    y = df["y_coord"].fillna(0)

    goal_x = np.where(x >= 0, 89.0, -89.0)
    raw_dx = goal_x - x  # positive = in front of goal, negative = behind
    dy = y

    df["shot_distance"] = np.sqrt(raw_dx ** 2 + dy ** 2)
    df["shot_angle"] = np.degrees(np.arctan2(dy.abs(), raw_dx.abs().clip(lower=0.01)))
    df["is_behind_net"] = (x.abs() > 89).astype(int)

    # --- strength state (vectorized) ---
    # situation_code format: {away_goalie}{away_skaters}{home_skaters}{home_goalie}
    # e.g. "1551" = 5v5 even strength
    sc = df["situation_code"].fillna("1551")
    away_goalie  = pd.to_numeric(sc.str[0], errors="coerce").fillna(1).astype(int)
    away_skaters = pd.to_numeric(sc.str[1], errors="coerce").fillna(5).astype(int)
    home_skaters = pd.to_numeric(sc.str[2], errors="coerce").fillna(5).astype(int)
    home_goalie  = pd.to_numeric(sc.str[3], errors="coerce").fillna(1).astype(int)

    is_home = (df["event_owner_team_id"] == df["home_team_id"])
    shooter_sk = np.where(is_home, home_skaters, away_skaters)
    opp_sk     = np.where(is_home, away_skaters, home_skaters)
    opp_goalie = np.where(is_home, away_goalie,  home_goalie)

    df["is_en"] = (opp_goalie == 0).astype(int)
    df["is_pp"] = ((shooter_sk > opp_sk) & (opp_goalie == 1)).astype(int)
    df["is_sh"] = ((shooter_sk < opp_sk) & (opp_goalie == 1)).astype(int)

    # --- shot type encoding ---
    df["shot_type_code"] = (
        df["shot_type"].str.lower()
        .map(SHOT_TYPE_MAP)
        .fillna(len(SHOT_TYPES))
        .astype(int)
    )

    # --- is rebound ---
    # Flag if the previous event in this game was also a shot-on-goal
    df = df.sort_values(["game_id", "sort_order"]).reset_index(drop=True)
    prev_game = df["game_id"].shift(1)
    prev_type = df["type_desc_key"].shift(1)
    df["is_rebound"] = (
        (prev_game == df["game_id"]) &
        (prev_type == "shot-on-goal")
    ).astype(int)

    # --- period (cap OT at 4) ---
    df["period"] = df["period"].fillna(1).clip(upper=4).astype(int)

    # --- target ---
    df["is_goal"] = (df["type_desc_key"] == "goal").astype(int)

    return df
