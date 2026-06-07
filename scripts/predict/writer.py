from __future__ import annotations

from db import upsert_rows
from scripts.serve.writer import clean_rows

PREDICTION_TABLE = "game_predictions"
PREDICTION_UPSERT_CONFLICT = "game_id"

REQUIRED_PREDICTION_FIELDS = [
    "game_id",
    "game_date",
    "game_type",
    "home_team_id",
    "away_team_id",
    "home_win_prob",
    "away_win_prob",
    "pred_home_goals",
    "pred_away_goals",
    "model_versions",
    "input_snapshot",
    "output_snapshot",
    "status",
]


def build_prediction_record(
    *,
    game_id,
    game_date,
    game_type,
    home_team_id,
    away_team_id,
    home_goalie_id,
    away_goalie_id,
    home_win_prob,
    away_win_prob,
    pred_home_goals,
    pred_away_goals,
    predicted_winner_team_id,
    model_versions,
    input_snapshot,
    output_snapshot,
    status="ok",
    error_message=None,
):
    return {
        "game_id": game_id,
        "game_date": game_date,
        "game_type": game_type,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_goalie_id": home_goalie_id,
        "away_goalie_id": away_goalie_id,
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "pred_home_goals": pred_home_goals,
        "pred_away_goals": pred_away_goals,
        "predicted_winner_team_id": predicted_winner_team_id,
        "model_versions": model_versions or {},
        "input_snapshot": input_snapshot or {},
        "output_snapshot": output_snapshot or {},
        "status": status,
        "error_message": error_message,
    }


def validate_prediction_record(record: dict) -> None:
    missing = [field for field in REQUIRED_PREDICTION_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Prediction record missing required fields: {missing}")


def write_game_prediction(
    record: dict,
    *,
    table: str = PREDICTION_TABLE,
    upsert_fn=upsert_rows,
):
    validate_prediction_record(record)
    rows = clean_rows([record])
    upsert_fn(table, rows, on_conflict=PREDICTION_UPSERT_CONFLICT)
    return rows[0]
