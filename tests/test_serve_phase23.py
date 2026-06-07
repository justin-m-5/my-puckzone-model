import datetime

import pandas as pd

from scripts.serve.benchmark import build_benchmark_rows
from scripts.serve.odds import american_to_implied_prob, normalize_market_odds
from scripts.serve import run as serve_run
from scripts.serve.writer import write_serving_rows


def test_write_serving_rows_schema_and_idempotent_upsert():
    calls = []

    def fake_upsert(table, rows, on_conflict=None):
        calls.append((table, rows, on_conflict))

    base = {
        "game_id": 1,
        "date": datetime.date(2026, 1, 2),
        "season": 20252026,
        "home_team_id": 10,
        "away_team_id": 20,
        "feature_version": "v2.3",
        "model_version": "goals-v1",
        "run_id": "rid",
        "generated_at": datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc),
        "home_win_probability": 0.55,
        "expected_home_goals": 3.1,
        "expected_away_goals": 2.7,
        "most_likely_home_score": 3,
        "most_likely_away_score": 2,
        "is_finalized": False,
        "data_source": "test",
        "prediction_date": datetime.date(2026, 1, 2),
    }
    rows = [base, {**base, "home_win_probability": 0.60}]

    written = write_serving_rows(rows, dry_run=False, upsert_fn=fake_upsert)

    assert written == 1
    assert len(calls) == 1
    table, payload, on_conflict = calls[0]
    assert table == "model_game_predictions"
    assert on_conflict == "game_id,prediction_date,feature_version,model_version"
    assert payload[0]["home_win_probability"] == 0.60


def test_odds_normalization_implied_probability_math():
    assert round(american_to_implied_prob(+150), 4) == 0.4
    assert round(american_to_implied_prob(-200), 4) == 0.6667

    rows = [
        {"game_id": 1, "date": "2026-01-02", "snapshot_type": "opening", "home_moneyline": -120, "away_moneyline": +110, "provider": "x"},
        {"game_id": 1, "date": "2026-01-02", "snapshot_type": "closing", "home_moneyline": -130, "away_moneyline": +115, "provider": "x"},
    ]
    normalized = normalize_market_odds(rows)
    assert 1 in normalized
    closing = normalized[1]
    assert closing.closing_home_prob is not None
    assert 0.0 < closing.closing_home_prob < 1.0
    assert round((closing.closing_home_prob or 0) + (closing.closing_away_prob or 0), 6) == 1.0


def test_benchmark_metric_computations_with_fixture_rows():
    serving_rows = [
        {
            "game_id": 1,
            "date": datetime.date(2026, 1, 2),
            "prediction_date": datetime.date(2026, 1, 2),
            "feature_version": "v2.3",
            "model_version": "goals-v1",
            "run_id": "rid",
            "generated_at": "2026-01-02T01:00:00+00:00",
            "home_win_probability": 0.62,
            "home_win_outcome": 1,
        },
        {
            "game_id": 2,
            "date": datetime.date(2026, 1, 2),
            "prediction_date": datetime.date(2026, 1, 2),
            "feature_version": "v2.3",
            "model_version": "goals-v1",
            "run_id": "rid",
            "generated_at": "2026-01-02T01:00:00+00:00",
            "home_win_probability": 0.41,
            "home_win_outcome": 0,
        },
    ]
    odds_rows = normalize_market_odds(
        [
            {"game_id": 1, "date": "2026-01-02", "snapshot_type": "closing", "home_moneyline": -110, "away_moneyline": -110, "provider": "x"},
            {"game_id": 2, "date": "2026-01-02", "snapshot_type": "closing", "home_moneyline": +120, "away_moneyline": -130, "provider": "x"},
        ]
    )

    game_rows, daily_rows = build_benchmark_rows(serving_rows, odds_rows)
    assert len(game_rows) == 2
    assert len(daily_rows) == 1
    assert daily_rows[0]["model_brier"] is not None
    assert daily_rows[0]["model_log_loss"] is not None
    assert isinstance(daily_rows[0]["calibration_slices"], list)


def test_benchmark_handles_missing_optional_fields_without_crash():
    serving_rows = [
        {
            "game_id": 10,
            "date": datetime.date(2026, 1, 2),
            "prediction_date": datetime.date(2026, 1, 2),
            "feature_version": "v2.3",
            "model_version": "goals-v1",
            "run_id": "rid",
            "generated_at": "2026-01-02T01:00:00+00:00",
            "home_win_probability": 0.5,
        }
    ]
    game_rows, daily_rows = build_benchmark_rows(serving_rows, odds_by_game={})
    assert len(game_rows) == 1
    assert len(daily_rows) == 1
    assert daily_rows[0]["model_brier"] is None


def test_orchestrator_dry_run_step_order(monkeypatch):
    calls = []

    class FakeProvider:
        def fetch_for_date(self, target_date):
            return [{"game_id": 1, "date": target_date.isoformat(), "snapshot_type": "closing", "home_moneyline": -110, "away_moneyline": -110, "provider": "x"}]

    monkeypatch.setattr(serve_run, "load_goals_payload", lambda: {"model": object(), "feature_cols": [], "model_name": "goals", "lambda3": 0.1})
    monkeypatch.setattr(
        serve_run,
        "fetch_games_for_date",
        lambda target_date: pd.DataFrame(
            [{"id": 1, "season": 20252026, "date": target_date, "game_type": 2, "game_state": "PRE", "home_team_id": 1, "away_team_id": 2}]
        ),
    )
    monkeypatch.setattr(serve_run.DataContext, "from_supabase", lambda: object())
    monkeypatch.setattr(
        serve_run,
        "generate_serving_rows",
        lambda **kwargs: [
            {
                "game_id": 1,
                "date": kwargs["pipeline_ctx"].target_date,
                "season": 20252026,
                "home_team_id": 1,
                "away_team_id": 2,
                "feature_version": kwargs["pipeline_ctx"].feature_version,
                "model_version": kwargs["pipeline_ctx"].model_version,
                "run_id": kwargs["pipeline_ctx"].run_id,
                "generated_at": kwargs["pipeline_ctx"].generated_at,
                "home_win_probability": 0.5,
                "expected_home_goals": 2.7,
                "expected_away_goals": 2.7,
                "most_likely_home_score": 3,
                "most_likely_away_score": 2,
                "is_finalized": False,
                "data_source": "test",
                "prediction_date": kwargs["pipeline_ctx"].target_date,
            }
        ],
    )

    def fake_write_serving(rows, dry_run=False):
        assert dry_run is True
        return len(rows)

    def fake_write_benchmark(game_rows, daily_rows, dry_run=False):
        assert dry_run is True
        return len(game_rows), len(daily_rows)

    monkeypatch.setattr(serve_run, "write_serving_rows", fake_write_serving)
    monkeypatch.setattr(serve_run, "write_benchmark_rows", fake_write_benchmark)

    result = serve_run.run_single_date(
        datetime.date(2026, 1, 2),
        dry_run=True,
        skip_odds=False,
        force_recompute=False,
        feature_version="v2.3",
        model_version="goals-v1",
        run_id="rid",
        odds_provider=FakeProvider(),
        step_recorder=calls.append,
    )

    assert calls == list(serve_run.CASCADE_STEPS)
    assert result["serving_rows"] == 1
    assert result["benchmark_daily_rows"] == 1
