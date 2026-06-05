# features/advanced.py
"""
Advanced team-strength metrics derived from play-by-play shot events.

Per team, per game:
  CF  / CA    Corsi  — all shot attempts (goal + SOG + missed + blocked) for/against
  xGF / xGA   Expected goals — xg_model.pkl scored over shots-on-goal + goals
  HDCF/ HDCA  High-danger chances — unblocked attempts from the home-plate slot
...plus shares (CF%, xGF%, HDCF%) and rolling (prior-N-game) versions for use as
features in the win/score models.

NOT trained models. Corsi/HDCF are COUNTED; xG is the only learned piece (loaded
from xg_model.pkl). The point is feeding rolling team versions into win/scores.

VERIFY before trusting the numbers:
  1. Needs missed-shot / blocked-shot rows in game_plays:
       select type_desc_key, count(*) from game_plays group by 1 order by 2 desc;
  2. NHL api-web stores a blocked-shot's event_owner_team_id as the BLOCKING
     (defending) team. Corsi credits the ATTACKER, so we flip it below. If your
     data already stores the shooting team as owner, set the flag to False.
"""

import pickle
import numpy as np
import pandas as pd
from db import supabase, fetch_all
from features.games import get_all_games
from features.plays import build_xg_features
from models.xg import XG_FEATURE_COLS

SHOT_ATTEMPT_TYPES = ["goal", "shot-on-goal", "missed-shot", "blocked-shot"]
UNBLOCKED_TYPES = ["goal", "shot-on-goal", "missed-shot"]
SOG_GOAL_TYPES = ["goal", "shot-on-goal"]

# See data note #2.
BLOCKED_SHOT_OWNER_IS_BLOCKER = True

# Home-plate / inner-slot approximation for high-danger (tunable, in feet).
HD_MAX_DIST_FT = 20.0       # feet in front of the goal line
HD_MAX_HALFWIDTH_FT = 22.0  # half-width of the slot (faceoff dots ~ ±22)

ROLL_WINDOW = 10
ROLL_COLS = ["cf_pct", "xgf_pct", "hdcf_pct"]


def _load_xg_model(path="xg_model.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("  WARNING: xg_model.pkl not found — xGF/xGA will be 0. "
              "Run scripts.train.xg first.")
        return None


def get_shot_attempts(games_df=None):
    """All shot-attempt events joined to game metadata."""
    query = supabase.table("game_plays").select(
        "game_id, sort_order, type_desc_key, period, situation_code, "
        "x_coord, y_coord, shot_type, event_owner_team_id"
    ).in_("type_desc_key", SHOT_ATTEMPT_TYPES)
    df = pd.DataFrame(fetch_all("game_plays", query))
    if df.empty:
        return df

    if games_df is None:
        games_df = get_all_games()
    games = games_df[["id", "season", "date", "home_team_id", "away_team_id"]].rename(
        columns={"id": "game_id"}
    )
    df = df.merge(games, on="game_id", how="inner")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _is_high_danger(x, y):
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    dist_in_front = 89.0 - x.abs()  # feet in front of goal line (neg = behind net)
    return (
        (dist_in_front >= 0)
        & (dist_in_front <= HD_MAX_DIST_FT)
        & (y.abs() <= HD_MAX_HALFWIDTH_FT)
    ).astype(int)


def build_advanced_team_games(games_df=None):
    """
    Returns one row per (game_id, team_id) with CF/CA, xGF/xGA, HDCF/HDCA and
    their shares. Raw single-game values (not yet rolled).
    """
    df = get_shot_attempts(games_df)
    if df.empty:
        print("  No shot-attempt events found. Is game_plays populated?")
        return pd.DataFrame()

    # --- attribute each attempt to the ATTACKING team ---
    df["attack_team_id"] = df["event_owner_team_id"]
    if BLOCKED_SHOT_OWNER_IS_BLOCKER:
        blocked = df["type_desc_key"] == "blocked-shot"
        other = np.where(
            df["event_owner_team_id"] == df["home_team_id"],
            df["away_team_id"], df["home_team_id"],
        )
        df.loc[blocked, "attack_team_id"] = other[blocked.values]

    # --- per-attempt flags ---
    df["is_unblocked"] = df["type_desc_key"].isin(UNBLOCKED_TYPES).astype(int)
    df["hd"] = (_is_high_danger(df["x_coord"], df["y_coord"]) & (df["is_unblocked"] == 1)).astype(int)

    # --- xG over the same population the model was trained on (SOG + goals) ---
    df["xg"] = 0.0
    xg_payload = _load_xg_model()
    if xg_payload is not None:
        sub = df[df["type_desc_key"].isin(SOG_GOAL_TYPES)].copy()
        if not sub.empty:
            feat = build_xg_features(sub)  # resets index; we merge back on keys
            X = feat[XG_FEATURE_COLS].fillna(0)
            model, scaler = xg_payload["model"], xg_payload.get("scaler")
            X = scaler.transform(X) if scaler is not None else X
            feat = feat.assign(_xg=model.predict_proba(X)[:, 1])
            df = df.merge(
                feat[["game_id", "sort_order", "_xg"]],
                on=["game_id", "sort_order"], how="left",
            )
            df["xg"] = df["_xg"].fillna(0.0)
            df = df.drop(columns="_xg")

    # --- aggregate each team's "FOR" totals per game ---
    fg = (
        df.groupby(["game_id", "season", "date", "attack_team_id"])
        .agg(cf=("type_desc_key", "size"), xgf=("xg", "sum"), hdcf=("hd", "sum"))
        .reset_index()
        .rename(columns={"attack_team_id": "team_id"})
    )

    # --- "AGAINST" = the opponent's "FOR" in the same game ---
    meta = df[["game_id", "home_team_id", "away_team_id"]].drop_duplicates("game_id")
    fg = fg.merge(meta, on="game_id", how="left")
    fg["opp_id"] = np.where(fg["team_id"] == fg["home_team_id"], fg["away_team_id"], fg["home_team_id"])
    opp = fg[["game_id", "team_id", "cf", "xgf", "hdcf"]].rename(
        columns={"team_id": "opp_id", "cf": "ca", "xgf": "xga", "hdcf": "hdca"}
    )
    fg = fg.merge(opp, on=["game_id", "opp_id"], how="left")
    for c in ["ca", "xga", "hdca"]:
        fg[c] = fg[c].fillna(0.0)

    # --- shares (safe denominators) ---
    def share(f, a):
        denom = f + a
        return np.where(denom > 0, f / denom, 0.5)

    fg["cf_pct"] = share(fg["cf"], fg["ca"])
    fg["xgf_pct"] = share(fg["xgf"], fg["xga"])
    fg["hdcf_pct"] = share(fg["hdcf"], fg["hdca"])
    return fg


def build_advanced_rolling(games_df=None, window=ROLL_WINDOW):
    """
    Rolling (prior-N-game, leak-safe) advanced shares per team.
    Returns dict: (game_id, team_id) -> {cf_pct, xgf_pct, hdcf_pct}
    """
    fg = build_advanced_team_games(games_df)
    if fg.empty:
        return {}

    lookup = {}
    for team_id, group in fg.groupby("team_id"):
        group = group.sort_values("date").reset_index(drop=True)
        for col in ROLL_COLS:
            group[f"r_{col}"] = group[col].shift(1).rolling(window, min_periods=3).mean()
        for _, row in group.iterrows():
            lookup[(row["game_id"], team_id)] = {
                col: (row[f"r_{col}"] if pd.notna(row[f"r_{col}"]) else None)
                for col in ROLL_COLS
            }
    return lookup


def get_advanced_for_game(advanced_lookup, game_id, team_id):
    """Rolling advanced shares for a team in a game ({} if unavailable)."""
    return advanced_lookup.get((game_id, team_id), {})