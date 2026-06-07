import datetime

import numpy as np

from scripts.predict import run as predict_run
from scripts.predict.writer import (
    PREDICTION_UPSERT_CONFLICT,
    build_prediction_record,
    write_game_prediction,
)


class FakeWinModel:
    def predict_proba(self, X):
        return np.array([[0.42, 0.58]])


class FakeScoreModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.array([self.value])


def _raise_db_write_error(record):
    raise RuntimeError("db write failed")


def _patch_prediction_dependencies(monkeypatch):
    monkeypatch.setattr(
        predict_run,
        "load_model",
        lambda path="win_model.pkl": {
            "model": FakeWinModel(),
            "scaler": None,
            "feature_cols": ["feature"],
            "model_name": "win-model-test",
            "decision_threshold": 0.5,
        },
    )
    monkeypatch.setattr(predict_run, "load_playoff_model", lambda path="playoff_model.pkl": None)
    monkeypatch.setattr(
        predict_run,
        "load_score_model",
        lambda path="score_model.pkl": {
            "home_model": FakeScoreModel(3.2),
            "away_model": FakeScoreModel(2.4),
            "feature_cols": ["feature"],
            "model_name": "score-model-test",
        },
    )
    monkeypatch.setattr(predict_run, "fill_features", lambda df: df)
    monkeypatch.setattr(
        predict_run,
        "build_prediction_row",
        lambda *args, **kwargs: (
            {"feature": 1.0},
            {
                "home_record": "10-5-1",
                "away_record": "9-6-1",
                "home_sv": 0.9123,
                "away_sv": 0.9055,
                "home_rest": 2,
                "away_rest": 1,
                "h2h": 0.6,
                "home_elo": 1512,
                "away_elo": 1498,
                "home_gf_pg": 3.2,
                "away_gf_pg": 2.4,
                "series": None,
            },
        ),
    )


def test_game_id_flow_maps_db_game_row_into_prediction_inputs(monkeypatch):
    captured = {}

    _patch_prediction_dependencies(monkeypatch)
    monkeypatch.setattr(
        predict_run,
        "fetch_game_by_id",
        lambda game_id: {
            "id": game_id,
            "date": "2024-01-15",
            "game_type": 3,
            "home_team_id": 1,
            "away_team_id": 2,
        },
    )
    monkeypatch.setattr(
        predict_run,
        "build_prediction_row",
        lambda home_team_id, away_team_id, game_date, is_playoff, home_goalie_id=None, away_goalie_id=None: (
            captured.update(
                {
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "game_date": game_date,
                    "is_playoff": is_playoff,
                    "home_goalie_id": home_goalie_id,
                    "away_goalie_id": away_goalie_id,
                }
            )
            or {"feature": 1.0},
            {
                "home_record": "10-5-1",
                "away_record": "9-6-1",
                "home_sv": None,
                "away_sv": None,
                "home_rest": 2,
                "away_rest": 1,
                "h2h": 0.6,
                "home_elo": 1512,
                "away_elo": 1498,
                "home_gf_pg": 3.2,
                "away_gf_pg": 2.4,
                "series": None,
            },
        ),
    )
    inputs = iter(["77", "88"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(predict_run, "write_game_prediction", lambda record: record)

    exit_code = predict_run.predict(["--game-id", "13"])

    assert exit_code == 0
    assert captured == {
        "home_team_id": 1,
        "away_team_id": 2,
        "game_date": datetime.date(2024, 1, 15),
        "is_playoff": True,
        "home_goalie_id": 77,
        "away_goalie_id": 88,
    }


def test_game_id_not_found_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(predict_run, "fetch_game_by_id", lambda game_id: None)

    exit_code = predict_run.predict(["--game-id", "999999"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No game found for game_id 999999" in captured.out


def test_game_id_goalie_prompt_valid_blank_and_invalid(monkeypatch, capsys):
    captured = {}

    _patch_prediction_dependencies(monkeypatch)
    monkeypatch.setattr(
        predict_run,
        "fetch_game_by_id",
        lambda game_id: {
            "id": game_id,
            "date": "2024-01-15",
            "game_type": 2,
            "home_team_id": 1,
            "away_team_id": 2,
        },
    )
    monkeypatch.setattr(
        predict_run,
        "build_prediction_row",
        lambda home_team_id, away_team_id, game_date, is_playoff, home_goalie_id=None, away_goalie_id=None: (
            captured.update(
                {
                    "home_goalie_id": home_goalie_id,
                    "away_goalie_id": away_goalie_id,
                }
            )
            or {"feature": 1.0},
            {
                "home_record": "10-5-1",
                "away_record": "9-6-1",
                "home_sv": None,
                "away_sv": None,
                "home_rest": 2,
                "away_rest": 1,
                "h2h": 0.6,
                "home_elo": 1512,
                "away_elo": 1498,
                "home_gf_pg": 3.2,
                "away_gf_pg": 2.4,
                "series": None,
            },
        ),
    )
    monkeypatch.setattr(predict_run, "write_game_prediction", lambda record: record)

    for raw_value, expected_value in [("77", 77), ("", None), ("oops", None)]:
        captured.clear()
        inputs = iter([raw_value, ""])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        exit_code = predict_run.predict(["--game-id", "13"])

        assert exit_code == 0
        assert captured == {
            "home_goalie_id": expected_value,
            "away_goalie_id": None,
        }

    output = capsys.readouterr().out
    assert "falling back to auto inference" in output


def test_write_game_prediction_payload_includes_required_fields():
    calls = []
    record = build_prediction_record(
        game_id=13,
        game_date=datetime.date(2024, 1, 15),
        game_type=2,
        home_team_id=1,
        away_team_id=2,
        home_goalie_id=77,
        away_goalie_id=None,
        home_win_prob=0.58,
        away_win_prob=0.42,
        pred_home_goals=3.2,
        pred_away_goals=2.4,
        predicted_winner_team_id=1,
        model_versions={"win_model": "win-model-test"},
        input_snapshot={"game_id": 13},
        output_snapshot={"winner": "New Jersey Devils"},
    )

    write_game_prediction(record, upsert_fn=lambda table, rows, on_conflict=None: calls.append((table, rows, on_conflict)))

    assert len(calls) == 1
    table, rows, on_conflict = calls[0]
    assert table == "game_predictions"
    assert on_conflict == PREDICTION_UPSERT_CONFLICT
    assert rows[0]["game_id"] == 13
    assert rows[0]["game_date"] == "2024-01-15"
    assert rows[0]["model_versions"] == {"win_model": "win-model-test"}
    assert rows[0]["input_snapshot"] == {"game_id": 13}
    assert rows[0]["output_snapshot"] == {"winner": "New Jersey Devils"}
    assert rows[0]["status"] == "ok"


def test_db_write_failure_still_prints_prediction_and_warns(monkeypatch, capsys):
    _patch_prediction_dependencies(monkeypatch)
    monkeypatch.setattr(
        predict_run,
        "fetch_game_by_id",
        lambda game_id: {
            "id": game_id,
            "date": "2024-01-15",
            "game_type": 2,
            "home_team_id": 1,
            "away_team_id": 2,
        },
    )
    inputs = iter(["", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr(predict_run, "write_game_prediction", _raise_db_write_error)

    exit_code = predict_run.predict(["--game-id", "13"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Predicted winner:" in captured.out
    assert "failed to persist prediction to public.game_predictions: db write failed" in captured.out
