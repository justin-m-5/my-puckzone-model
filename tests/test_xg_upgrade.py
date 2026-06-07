import pickle

import pandas as pd

from features.plays import build_xg_features
from models.xg import XG_FEATURE_COLS, build_xg_feature_matrix
from scripts.train import xg as train_xg


def _fixture_shots():
    rows = []
    for i in range(60):
        season = 20242025 if i < 40 else 20252026
        rows.append({
            "id": i + 1,
            "game_id": 1000 + (i // 4),
            "sort_order": i + 1,
            "type_desc_key": "goal" if i % 7 == 0 else "shot-on-goal",
            "period": 1 + (i % 3),
            "period_type": "REG",
            "situation_code": "1551",
            "x_coord": 55 + (i % 30),
            "y_coord": (-20 + (i % 40)),
            "shot_type": "wrist" if i % 2 == 0 else "slap",
            "event_owner_team_id": 1 if i % 2 == 0 else 2,
            "shooting_player_id": 10 + i,
            "goalie_in_net_id": 99,
            "home_team_id": 1,
            "date": "2025-01-01",
            "season": season,
            "time_in_period_seconds": (i % 20) * 7,
        })
    return pd.DataFrame(rows)


def test_build_xg_features_includes_v2_context_columns():
    feat = build_xg_features(_fixture_shots())
    expected = {
        "seconds_since_last_event",
        "event_gap",
        "is_rush_proxy",
        "is_one_timer_proxy",
        "is_5v5",
        "strength_state_code",
        "score_state",
        "x_abs_norm",
        "xy_interaction",
        "lane_center_mid_angle",
    }
    assert expected.issubset(set(feat.columns))
    assert feat["seconds_since_last_event"].ge(0).all()


def test_train_xg_writes_payload_with_schema_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(train_xg, "get_shot_events", lambda: _fixture_shots())
    monkeypatch.chdir(tmp_path)
    train_xg.train()

    with open(tmp_path / "xg_model.pkl", "rb") as f:
        payload = pickle.load(f)

    assert payload["payload_version"] == 2
    assert payload["feature_cols"] == XG_FEATURE_COLS
    assert {"auc", "log_loss", "brier", "calibration_bins"} <= set(payload["metrics"].keys())
    assert isinstance(payload["metrics"]["calibration_bins"], list)


def test_feature_matrix_reindexes_missing_optional_columns():
    feat = build_xg_features(_fixture_shots().drop(columns=["time_in_period_seconds"]))
    payload_cols = XG_FEATURE_COLS + ["future_optional_column"]
    X = build_xg_feature_matrix(feat, payload_cols)
    assert list(X.columns) == payload_cols
    assert (X["future_optional_column"] == 0.0).all()
