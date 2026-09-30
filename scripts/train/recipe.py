"""
scripts/train/recipe.py — shared evaluate-then-refit recipe for the serving models.

1. Hold out the most recent complete season and score a model trained on
   everything before it (honest out-of-sample metrics).
2. Refit the same recipe on ALL seasons, so the saved model includes the
   latest complete season instead of never seeing it.

Candidates are written to candidates/ by default; scripts.artifacts.promote
moves them live after review.
"""

from __future__ import annotations

import argparse
import datetime
import os

import pandas as pd

# A regular season has 1,312 games; a season with at least this many training
# rows counts as complete enough to hold out.
MIN_COMPLETE_SEASON_ROWS = 1000
CANDIDATE_DIR = "candidates"


def parse_args(description: str, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--holdout-season", type=int, default=None,
                        help="Season to score on (default: most recent complete season)")
    parser.add_argument("--out", default=CANDIDATE_DIR,
                        help=f"Output directory (default: {CANDIDATE_DIR}/; '.' writes live files directly)")
    return parser.parse_args(argv)


def pick_holdout_season(df: pd.DataFrame, requested: int | None) -> int:
    if requested is not None:
        return requested
    counts = df["season"].value_counts()
    complete = sorted(int(s) for s, n in counts.items() if n >= MIN_COMPLETE_SEASON_ROWS)
    if not complete:
        raise SystemExit("No complete season to hold out.")
    return complete[-1]


def training_window(df: pd.DataFrame, holdout_season: int, holdout_metrics: dict) -> dict:
    """Metadata stored in the payload so a model can be compared and audited later."""
    dates = pd.to_datetime(df["date"]) if "date" in df.columns else None
    return {
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seasons": sorted(int(s) for s in df["season"].unique()),
        "trained_through": dates.max().date().isoformat() if dates is not None else None,
        "n_rows": int(len(df)),
        "holdout_season": holdout_season,
        "holdout_metrics": holdout_metrics,
    }


def output_path(out_dir: str, filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)
