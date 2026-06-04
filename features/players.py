# features/players.py

import pandas as pd
from db import supabase, fetch_all
from features.games import get_games


SKATER_COLS = ["goals", "assists", "points", "sog", "plus_minus", "hits", "blocked_shots"]

POSITION_ORDER = {"C": 0, "L": 1, "R": 2, "D": 3}


def get_skater_stats():
    """Fetch all skater game stats with game dates and season."""
    query = supabase.table("game_skater_stats") \
        .select(
            "game_id, player_id, team_id, is_home, position, toi, "
            "goals, assists, points, sog, plus_minus, hits, blocked_shots, "
            "faceoff_winning_pctg"
        )
    df = pd.DataFrame(fetch_all("game_skater_stats", query))

    games = get_games()[["id", "date", "season"]].rename(columns={"id": "game_id"})
    df = df.merge(games, on="game_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df["toi_sec"] = df["toi"].apply(_parse_toi)
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    return df


def _parse_toi(toi_str):
    """Convert 'MM:SS' string to total seconds."""
    if not toi_str or pd.isna(toi_str):
        return 0.0
    try:
        parts = str(toi_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return 0.0


def build_skater_rolling(skater_df, window=10):
    """
    For each player + game, compute rolling averages of key stats
    over the previous `window` games (not including current game).
    Returns the input DataFrame with rolling_* columns appended.
    Uses vectorized groupby.transform — fast on large datasets.
    """
    cols = SKATER_COLS + ["toi_sec"]
    df = skater_df.sort_values(["player_id", "date"]).reset_index(drop=True)

    for col in cols:
        df[f"rolling_{col}"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )

    return df


def build_player_training_rows(rolling_df):
    """
    Build one row per player-game for training a player scoring model.
    Expects the DataFrame returned by build_skater_rolling (with rolling_* cols).

    Features:
        - rolling_* stats from prior games
        - is_home, position_code
        - season (for train/test splitting)

    Targets:
        - scored_point: 1 if the player recorded >= 1 point in this game
        - points: raw point total (for regression)
    """
    df = rolling_df.dropna(subset=["rolling_points"]).copy()
    skipped = len(rolling_df) - len(df)

    df["scored_point"] = (df["points"].fillna(0) > 0).astype(int)
    df["is_home"] = df["is_home"].astype(int)
    df["position_code"] = (
        df["position"].str.upper().str[:1]
        .map(POSITION_ORDER)
        .fillna(3)
        .astype(int)
    )

    print(f"  Built {len(df)} player-game rows ({skipped} skipped — no rolling history yet)")
    return df


def get_team_skater_form(rolling_df, game_id, team_id, top_n=12):
    """
    For a given team in a game, aggregate the rolling form of their top
    skaters (by prior rolling TOI). Returns per-game averages for the top_n.
    Expects the DataFrame returned by build_skater_rolling.
    Useful for feeding team-level player quality into game predictions.
    """
    team_rows = rolling_df[
        (rolling_df["game_id"] == game_id) &
        (rolling_df["team_id"] == team_id) &
        rolling_df["rolling_toi_sec"].notna()
    ].nlargest(top_n, "rolling_toi_sec")

    if team_rows.empty:
        return {}

    result = {}
    for col in SKATER_COLS:
        key = f"rolling_{col}"
        valid = team_rows[key].dropna()
        result[key] = float(valid.mean()) if not valid.empty else None

    return result
