import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models import FEATURE_COLS
from scripts.serve import run as serve_run
from scripts.train import goals as train_goals


def test_train_goals_payload_includes_scaler_and_metadata(monkeypatch, tmp_path):
    rng = np.random.default_rng(7)
    n_rows = 24
    feature_data = {col: rng.normal(size=n_rows) for col in FEATURE_COLS}
    feature_data["game_id"] = np.arange(1, n_rows + 1)
    feature_data["season"] = np.array([20242025] * 18 + [20252026] * 6)
    feature_df = pd.DataFrame(feature_data)

    score_rows = [
        {"id": i + 1, "home_score": int(rng.integers(1, 6)), "away_score": int(rng.integers(1, 6))}
        for i in range(n_rows)
    ]

    class _DummyQuery:
        def select(self, *_args, **_kwargs):
            return self

        def eq(self, *_args, **_kwargs):
            return self

        def in_(self, *_args, **_kwargs):
            return self

    class _DummySupabase:
        def table(self, _name):
            return _DummyQuery()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_goals, "build_features", lambda use_materialized=True: feature_df.copy())
    monkeypatch.setattr(train_goals, "fetch_all", lambda _table, _query: score_rows)
    monkeypatch.setattr(train_goals, "supabase", _DummySupabase())

    train_goals.train()

    payload = pickle.load(open(tmp_path / "goals_model.pkl", "rb"))
    assert payload["feature_cols"] == FEATURE_COLS
    assert payload["scaler"] is not None
    assert payload["model_name"] == "Bivariate Poisson goals model"
    assert payload["model_version"] == "phase-2.4.1"
    assert payload["payload_version"] == "2.4.1"
    assert payload["tie_rule"] is not None
    assert float(payload["lambda3"]) >= 0.0


def test_generate_serving_rows_compatible_with_and_without_scaler(monkeypatch):
    class _FakeModel:
        def __init__(self):
            self.last_first_feature = None

        def predict_rates(self, X):
            arr = np.asarray(X, dtype=float)
            self.last_first_feature = float(arr[0, 0])
            return np.array([2.8]), np.array([2.4]), np.array([0.2])

    game_date = datetime.date(2026, 1, 2)
    games_df = pd.DataFrame(
        [
            {
                "id": 1,
                "season": 20252026,
                "date": game_date,
                "game_state": "PRE",
                "game_type": 2,
                "home_team_id": 1,
                "away_team_id": 2,
            }
        ]
    )
    pctx = serve_run.PipelineContext(
        target_date=game_date,
        dry_run=True,
        skip_odds=True,
        force_recompute=False,
        feature_version="v2.3",
        model_version="goals-v1",
        run_id="rid",
        generated_at="2026-01-02T00:00:00+00:00",
    )

    row = {col: 0.0 for col in FEATURE_COLS}
    row[FEATURE_COLS[0]] = 10.0
    monkeypatch.setattr(serve_run, "_build_row_for_game", lambda _ctx, _game, _target_date: row.copy())

    old_model = _FakeModel()
    rows_old = serve_run.generate_serving_rows(
        pipeline_ctx=pctx,
        games_df=games_df,
        ctx=object(),
        goals_payload={"model": old_model, "feature_cols": FEATURE_COLS},
    )
    assert np.isclose(old_model.last_first_feature, 10.0)

    scaler = StandardScaler().fit(pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}, row]))
    new_model = _FakeModel()
    rows_new = serve_run.generate_serving_rows(
        pipeline_ctx=pctx,
        games_df=games_df,
        ctx=object(),
        goals_payload={"model": new_model, "feature_cols": FEATURE_COLS, "scaler": scaler},
    )
    assert not np.isclose(new_model.last_first_feature, 10.0)

    for rows in (rows_old, rows_new):
        assert len(rows) == 1
        out = rows[0]
        assert 0.0 <= out["home_win_probability"] <= 1.0
        assert out["expected_home_goals"] > 0.0
        assert out["expected_away_goals"] > 0.0
        assert out["lambda_home"] > 0.0
        assert out["lambda_away"] > 0.0
        assert out["lambda3"] >= 0.0
