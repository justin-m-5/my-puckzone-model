# features/training/builder.py
"""
Regular-season training feature builder — thin wrapper over features/pipeline.py.

All feature computation has moved to the unified pipeline so that training and
serving share a single code path (Phase 2.0 refactor).  This module is kept
for backward compatibility: everything that imports
``from features.training import build_features`` continues to work unchanged.
"""

from features.pipeline import DataContext, build_features_batch


def build_features():
    """
    Build the regular-season training dataset (51 features + metadata).

    Loads all data from Supabase via DataContext.from_supabase() and delegates
    to build_features_batch(game_type=2).  Returns a DataFrame identical in
    schema to the v1.x output.
    """
    ctx = DataContext.from_supabase()
    print("Building regular-season features (game_type=2)...")
    return build_features_batch(ctx, game_type=2)