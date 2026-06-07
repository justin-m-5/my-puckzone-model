# features/plays.py

import numpy as np
import pandas as pd
from db import supabase, fetch_all
from features.games import get_games


SHOT_TYPES = [
    "wrist", "slap", "snap", "backhand",
    "tip-in", "deflection", "bat", "between-legs",
    "poke", "wrap-around",
]
SHOT_TYPE_MAP = {t: i for i, t in enumerate(SHOT_TYPES)}
MISSING_EVENT_SECONDS = 999.0
MISSING_EVENT_GAP = 99.0
MAX_EVENT_INTERVAL_SECONDS = 300.0
MIN_SCORE_STATE = -3.0
MAX_SCORE_STATE = 3.0
COORD_NORM_MAX = 1.2


def _optional_numeric(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _event_elapsed_seconds(df):
    """
    Best-effort event time in elapsed-game seconds.
    Uses optional columns when available, otherwise falls back to sort_order proxy.
    """
    period = _optional_numeric(df, "period", default=1.0).clip(lower=1.0)
    if "time_in_period" in df.columns:
        mins_secs = df["time_in_period"].fillna("00:00").astype(str).str.split(":", n=1, expand=True)
        mm = pd.to_numeric(mins_secs[0], errors="coerce").fillna(0)
        ss = pd.to_numeric(mins_secs[1], errors="coerce").fillna(0)
        return ((period - 1.0) * 20.0 * 60.0 + mm * 60.0 + ss).astype(float)
    if "period_time" in df.columns:
        mins_secs = df["period_time"].fillna("00:00").astype(str).str.split(":", n=1, expand=True)
        mm = pd.to_numeric(mins_secs[0], errors="coerce").fillna(0)
        ss = pd.to_numeric(mins_secs[1], errors="coerce").fillna(0)
        return ((period - 1.0) * 20.0 * 60.0 + mm * 60.0 + ss).astype(float)
    if "time_in_period_seconds" in df.columns:
        sec = _optional_numeric(df, "time_in_period_seconds", default=0.0).clip(lower=0.0)
        return ((period - 1.0) * 20.0 * 60.0 + sec).astype(float)
    return pd.to_numeric(df["sort_order"], errors="coerce").fillna(0).astype(float)


def get_shot_events():
    """
    Fetch shot-on-goal and goal events from game_plays, joined with regular
    season game metadata (date, season, home_team_id).

    Fetches in batches of game IDs so every query hits the (game_id, event_id)
    primary-key index instead of scanning all ~4.6M plays. A single filtered
    full-table read trips Postgres' statement timeout (57014); per-game-batch
    reads do not.
    """
    games = get_games()[["id", "date", "season", "home_team_id"]].rename(columns={"id": "game_id"})
    game_ids = games["game_id"].tolist()

    rows = []
    BATCH = 50
    for i in range(0, len(game_ids), BATCH):
        batch = game_ids[i:i + BATCH]
        query = supabase.table("game_plays") \
            .select(
                "id, game_id, sort_order, type_desc_key, period, period_type, "
                "situation_code, x_coord, y_coord, shot_type, event_owner_team_id, "
                "shooting_player_id, goalie_in_net_id"
            ) \
            .in_("game_id", batch) \
            .in_("type_desc_key", ["shot-on-goal", "goal"])
        rows.extend(fetch_all("game_plays", query))
        if (i // BATCH) % 20 == 0:
            print(f"  ...fetched shots through game {i + len(batch)}/{len(game_ids)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.merge(games, on="game_id", how="inner")
    return df


def build_xg_features(shot_df):
    """
    Compute xG model features for each shot event.
    Returns the input DataFrame with feature columns appended.

    Required columns:
      game_id, sort_order, type_desc_key, event_owner_team_id, home_team_id,
      x_coord, y_coord, shot_type, situation_code, period
    Optional context columns (used when present, otherwise neutral defaults):
      time_in_period, period_time, time_in_period_seconds
    """
    df = shot_df.copy()

    x = df["x_coord"].fillna(0)
    y = df["y_coord"].fillna(0)

    goal_x = np.where(x >= 0, 89.0, -89.0)
    raw_dx = goal_x - x
    dy = y

    df["shot_distance"] = np.sqrt(raw_dx ** 2 + dy ** 2)
    df["shot_angle"] = np.degrees(np.arctan2(dy.abs(), raw_dx.abs().clip(lower=0.01)))
    df["is_behind_net"] = (x.abs() > 89).astype(int)

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

    df["shot_type_code"] = (
        df["shot_type"].str.lower()
        .map(SHOT_TYPE_MAP)
        .fillna(len(SHOT_TYPES))
        .astype(int)
    )

    df = df.sort_values(["game_id", "sort_order"]).reset_index(drop=True)
    prev_game = df["game_id"].shift(1)
    prev_team = df["event_owner_team_id"].shift(1)
    prev_type = df["type_desc_key"].shift(1)
    same_game = prev_game == df["game_id"]

    event_seconds = _event_elapsed_seconds(df)
    prev_seconds = event_seconds.shift(1)
    elapsed = np.where(same_game, (event_seconds - prev_seconds).clip(lower=0), np.nan)
    fallback_elapsed = (
        pd.to_numeric(df["sort_order"], errors="coerce")
        .diff()
        .clip(lower=0)
        .fillna(MISSING_EVENT_SECONDS)
        .to_numpy()
    )
    df["seconds_since_last_event"] = np.where(np.isnan(elapsed), fallback_elapsed, elapsed)
    df["seconds_since_last_event"] = np.clip(
        df["seconds_since_last_event"], 0, MAX_EVENT_INTERVAL_SECONDS
    ).astype(float)

    sort_gap = pd.to_numeric(df["sort_order"], errors="coerce").diff().fillna(MISSING_EVENT_GAP)
    df["event_gap"] = np.clip(
        np.where(same_game, sort_gap, MISSING_EVENT_GAP),
        0,
        MISSING_EVENT_GAP,
    ).astype(float)
    df["prev_event_same_team"] = (same_game & (prev_team == df["event_owner_team_id"])).astype(int)
    df["possession_change"] = (same_game & (prev_team != df["event_owner_team_id"])).astype(int)

    df["is_rebound"] = (
        same_game &
        (prev_type == "shot-on-goal") &
        (df["event_gap"] <= 2) &
        (df["seconds_since_last_event"] <= 8)
    ).astype(int)
    df["is_rush_proxy"] = (same_game & (df["event_gap"] <= 3) & (df["seconds_since_last_event"] <= 10)).astype(int)
    df["is_one_timer_proxy"] = (
        (df["shot_type"].str.lower().isin(["slap", "snap", "wrist"])) &
        (df["shot_angle"] >= 25) &
        (df["event_gap"] <= 3)
    ).astype(int)

    df["period"] = df["period"].fillna(1).clip(upper=4).astype(int)
    df["is_5v5"] = ((shooter_sk == 5) & (opp_sk == 5) & (opp_goalie == 1)).astype(int)
    df["strength_state_code"] = (df["is_pp"] * 2 + df["is_sh"]).astype(int)

    home_goal_event = ((df["type_desc_key"] == "goal") & (df["event_owner_team_id"] == df["home_team_id"])).astype(int)
    away_goal_event = ((df["type_desc_key"] == "goal") & (df["event_owner_team_id"] != df["home_team_id"])).astype(int)
    goals_for = home_goal_event.groupby(df["game_id"]).cumsum() - home_goal_event
    goals_against = away_goal_event.groupby(df["game_id"]).cumsum() - away_goal_event
    shooter_is_home = (df["event_owner_team_id"] == df["home_team_id"]).astype(int)
    shooter_goals = np.where(shooter_is_home == 1, goals_for, goals_against)
    opp_goals = np.where(shooter_is_home == 1, goals_against, goals_for)
    df["score_state"] = np.clip(shooter_goals - opp_goals, MIN_SCORE_STATE, MAX_SCORE_STATE).astype(float)

    x_loc = df["x_coord"].fillna(0)
    y_loc = df["y_coord"].fillna(0)
    x_norm = x_loc.abs() / 100.0
    y_abs = y_loc.abs()
    y_norm = y_abs / 42.5
    df["x_abs_norm"] = np.clip(x_norm, 0, COORD_NORM_MAX)
    df["y_abs_norm"] = np.clip(y_norm, 0, COORD_NORM_MAX)
    df["x_sq"] = df["x_abs_norm"] ** 2
    df["y_sq"] = df["y_abs_norm"] ** 2
    df["xy_interaction"] = df["x_abs_norm"] * df["y_abs_norm"]

    center = (y_abs < 10).astype(int)
    left = (y_loc >= 10).astype(int)
    right = (y_loc <= -10).astype(int)
    angle_low = (df["shot_angle"] < 20).astype(int)
    angle_mid = ((df["shot_angle"] >= 20) & (df["shot_angle"] < 45)).astype(int)
    angle_high = (df["shot_angle"] >= 45).astype(int)
    df["lane_center_low_angle"] = (center & angle_low).astype(int)
    df["lane_center_mid_angle"] = (center & angle_mid).astype(int)
    df["lane_wide_high_angle"] = ((left | right) & angle_high).astype(int)

    df["is_goal"] = (df["type_desc_key"] == "goal").astype(int)

    return df