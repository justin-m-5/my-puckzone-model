from __future__ import annotations

import datetime
from typing import Callable

import numpy as np
import pandas as pd

from db import upsert_rows

SERVING_TABLE = "model_game_predictions"
BENCHMARK_GAME_TABLE = "model_market_benchmarks"
BENCHMARK_DAILY_TABLE = "model_market_benchmark_daily"

SERVING_UPSERT_CONFLICT = "game_id,prediction_date,feature_version,model_version"
BENCHMARK_GAME_CONFLICT = "game_id,prediction_date,feature_version,model_version"
BENCHMARK_DAILY_CONFLICT = "date,feature_version,model_version"

REQUIRED_SERVING_FIELDS = [
    "game_id",
    "date",
    "season",
    "home_team_id",
    "away_team_id",
    "feature_version",
    "model_version",
    "run_id",
    "generated_at",
    "home_win_probability",
    "expected_home_goals",
    "expected_away_goals",
    "most_likely_home_score",
    "most_likely_away_score",
    "is_finalized",
    "data_source",
    "prediction_date",
]


def _json_safe(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        dt_value = value.to_pydatetime()
        if dt_value.time() == datetime.time.min:
            return dt_value.date().isoformat()
        return dt_value.isoformat()
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if pd.isna(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def clean_rows(rows: list[dict]) -> list[dict]:
    return [{key: _json_safe(value) for key, value in row.items()} for row in rows]


def validate_serving_rows(rows: list[dict]) -> None:
    for row in rows:
        missing = [f for f in REQUIRED_SERVING_FIELDS if f not in row]
        if missing:
            raise ValueError(f"Serving row missing required fields: {missing}")


def dedupe_rows(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    if not rows:
        return []
    deduped = {}
    for row in rows:
        key = tuple(row.get(k) for k in key_fields)
        deduped[key] = row
    return list(deduped.values())


def upsert_table_rows(
    table: str,
    rows: list[dict],
    *,
    on_conflict: str,
    dry_run: bool = False,
    upsert_fn: Callable[[str, list[dict], str | None], object] = upsert_rows,
) -> int:
    cleaned = clean_rows(rows)
    if dry_run:
        return len(cleaned)
    upsert_fn(table, cleaned, on_conflict=on_conflict)
    return len(cleaned)


def write_serving_rows(
    rows: list[dict],
    *,
    dry_run: bool = False,
    table: str = SERVING_TABLE,
    upsert_fn: Callable[[str, list[dict], str | None], object] = upsert_rows,
) -> int:
    deduped = dedupe_rows(
        rows,
        key_fields=("game_id", "prediction_date", "feature_version", "model_version"),
    )
    validate_serving_rows(deduped)
    return upsert_table_rows(
        table,
        deduped,
        on_conflict=SERVING_UPSERT_CONFLICT,
        dry_run=dry_run,
        upsert_fn=upsert_fn,
    )


def write_benchmark_rows(
    game_rows: list[dict],
    daily_rows: list[dict],
    *,
    dry_run: bool = False,
    game_table: str = BENCHMARK_GAME_TABLE,
    daily_table: str = BENCHMARK_DAILY_TABLE,
    upsert_fn: Callable[[str, list[dict], str | None], object] = upsert_rows,
) -> tuple[int, int]:
    game_count = upsert_table_rows(
        game_table,
        dedupe_rows(game_rows, key_fields=("game_id", "prediction_date", "feature_version", "model_version")),
        on_conflict=BENCHMARK_GAME_CONFLICT,
        dry_run=dry_run,
        upsert_fn=upsert_fn,
    )
    daily_count = upsert_table_rows(
        daily_table,
        dedupe_rows(daily_rows, key_fields=("date", "feature_version", "model_version")),
        on_conflict=BENCHMARK_DAILY_CONFLICT,
        dry_run=dry_run,
        upsert_fn=upsert_fn,
    )
    return game_count, daily_count
