"""
Compute per-game advanced stats (Corsi / xG / high-danger) ONCE and store them
in the game_advanced_stats table, so training runs read a small table instead
of crunching ~1.4M play rows every time.

RUN THIS:
  - after training/retraining the xG model (xGF/xGA depend on xg_model.pkl)
  - after ingesting new games
Otherwise the table goes stale and training reads old numbers.

Usage:
    PYTHONPATH=. python3 -m scripts.materialize_advanced
"""

import math
import numpy as np
import pandas as pd
from db import supabase
from features.games import get_all_games
from features.advanced import build_advanced_team_games

BATCH = 500
COLS = ["game_id", "team_id", "season", "date",
        "cf", "ca", "xgf", "xga", "hdcf", "hdca",
        "cf_pct", "xgf_pct", "hdcf_pct"]


def _clean(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if math.isnan(v) else float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def materialize():
    print("Loading all games...")
    games = get_all_games()

    print("Computing advanced per-game stats (this is the slow part)...")
    fg = build_advanced_team_games(games_df=games)
    if fg.empty:
        print("No advanced stats computed — aborting (is game_plays populated?).")
        return

    print("\n--- Sanity check (verify BEFORE retraining) ---")
    print(fg[["cf_pct", "xgf_pct", "hdcf_pct"]].describe())
    print(f"\n  cf_pct mean = {fg['cf_pct'].mean():.4f}  (must be ~0.500 by symmetry)")
    print(f"  xgf_pct std = {fg['xgf_pct'].std():.4f}  (must be > 0; ~0 means xg_model.pkl didn't load)")
    if fg["xgf_pct"].std() < 1e-6:
        print("  WARNING: xgf_pct has no spread — train scripts.train.xg FIRST, then re-run this.")

    fg = fg[COLS].copy()
    fg["date"] = pd.to_datetime(fg["date"]).dt.strftime("%Y-%m-%d")
    records = [{k: _clean(v) for k, v in r.items()} for r in fg.to_dict(orient="records")]

    print(f"\nUpserting {len(records)} rows into game_advanced_stats...")
    for i in range(0, len(records), BATCH):
        chunk = records[i:i + BATCH]
        supabase.table("game_advanced_stats").upsert(chunk, on_conflict="game_id,team_id").execute()
        print(f"  {min(i + BATCH, len(records))}/{len(records)}")

    print("\nDone. Training runs will now read this table.")


if __name__ == "__main__":
    materialize()
