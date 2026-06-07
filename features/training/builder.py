# features/training/builder.py
"""
Regular-season training feature builder — thin wrapper over features/pipeline.py.

All feature computation has moved to the unified pipeline so that training and
serving share a single code path (Phase 2.0 refactor).  This module is kept
for backward compatibility: everything that imports
``from features.training import build_features`` continues to work unchanged.
"""

from features.pipeline import DataContext, build_features_batch
from features.materialized import get_materialized_game_features
from models.game import FEATURE_COLS


def _regular_contract(df):
    cols = ["game_id", "season", "date", "home_team_id", "away_team_id"] + FEATURE_COLS + ["target"]
    return df.reindex(columns=cols)


def build_features(use_materialized: bool = True):
    """
    Build the regular-season training dataset (51 features + metadata).

    Loads all data from Supabase via DataContext.from_supabase() and delegates
    to build_features_batch(game_type=2).  Returns a DataFrame identical in
    schema to the v1.x output.
    """
    if use_materialized:
        materialized_df = get_materialized_game_features(game_type=2)
        if not materialized_df.empty:
            print(f"Using materialized regular-season feature store ({len(materialized_df)} rows)")
            return _regular_contract(materialized_df)

    ctx = DataContext.from_supabase()
    print("Building regular-season features (game_type=2)...")
    return _regular_contract(build_features_batch(ctx, game_type=2))