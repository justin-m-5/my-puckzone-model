import datetime
import json
import numpy as np
import pandas as pd

from features.materialized import parse_materialized_rows
from features.pipeline import DataContext, build_features_batch
from features.training import builder as training_builder
from features.training import playoff_builder
from scripts.materialize.run import build_form_snapshot_rows, build_materialized_rows, _clean_rows
from models.game import FEATURE_COLS
from tests.conftest import HOME_TEAM, AWAY_TEAM, ALT_AWAY_TEAM


def _ctx_with_playoffs(ctx):
    playoff_games = pd.DataFrame(
        [
            {
                "id": 9901,
                "season": 20232024,
                "game_type": 3,
                "date": datetime.date(2024, 4, 20),
                "home_team_id": HOME_TEAM,
                "away_team_id": AWAY_TEAM,
                "home_score": 3,
                "away_score": 2,
            },
            {
                "id": 9902,
                "season": 20232024,
                "game_type": 3,
                "date": datetime.date(2024, 4, 22),
                "home_team_id": AWAY_TEAM,
                "away_team_id": HOME_TEAM,
                "home_score": 1,
                "away_score": 4,
            },
            {
                "id": 9903,
                "season": 20232024,
                "game_type": 2,
                "date": datetime.date(2024, 4, 24),
                "home_team_id": HOME_TEAM,
                "away_team_id": ALT_AWAY_TEAM,
                "home_score": 5,
                "away_score": 1,
            },
        ]
    )

    game_rows = pd.concat([ctx.games, playoff_games], ignore_index=True)

    def _append_rows(df, cols, game_df):
        rows = []
        for _, g in game_df.iterrows():
            for team_id, is_home in [
                (int(g["home_team_id"]), True),
                (int(g["away_team_id"]), False),
            ]:
                base = {"game_id": g["id"], "team_id": team_id, "is_home": is_home, "date": g["date"]}
                base.update(cols(team_id))
                rows.append(base)
        return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    goalie_df = _append_rows(
        ctx.goalie_df,
        lambda team_id: {"player_id": 1000 + team_id, "saves": 28, "shots_against": 31},
        playoff_games,
    )
    gsax_df = pd.concat(
        [
            ctx.gsax_df,
            pd.DataFrame(
                [
                    {"game_id": g["id"], "player_id": 1000 + t, "team_id": t, "date": g["date"], "gsax": 0.2}
                    for _, g in playoff_games.iterrows()
                    for t in [int(g["home_team_id"]), int(g["away_team_id"])]
                ]
            ),
        ],
        ignore_index=True,
    )
    team_stats_df = _append_rows(
        ctx.team_stats_df,
        lambda _: {
            "pp_pctg": 0.21,
            "faceoff_winning_pctg": 0.5,
            "sog": 30,
            "hits": 20,
            "blocked_shots": 12,
        },
        playoff_games,
    )
    advanced_df = pd.concat(
        [
            ctx.advanced_df,
            pd.DataFrame(
                [
                    {
                        "game_id": g["id"],
                        "team_id": t,
                        "season": 20232024,
                        "date": g["date"],
                        "cf_pct": 0.51,
                        "xgf_pct": 0.5,
                        "hdcf_pct": 0.49,
                        "cf_pct_5v5": 0.5,
                        "xgf_pct_5v5": 0.49,
                        "hdcf_pct_5v5": 0.48,
                    }
                    for _, g in playoff_games.iterrows()
                    for t in [int(g["home_team_id"]), int(g["away_team_id"])]
                ]
            ),
        ],
        ignore_index=True,
    )

    return DataContext(
        games=game_rows,
        standings=ctx.standings,
        goalie_df=goalie_df,
        gsax_df=gsax_df,
        team_stats_df=team_stats_df,
        advanced_df=advanced_df,
    )


def test_materialized_round_trip_parity_regular(ctx):
    live_df = build_features_batch(ctx, game_type=2).sort_values("game_id").reset_index(drop=True)
    rows = build_materialized_rows(ctx)
    parsed = parse_materialized_rows(rows)
    parsed_reg = parsed[parsed["game_type"] == 2].sort_values("game_id").reset_index(drop=True)

    common = sorted(set(live_df["game_id"]).intersection(set(parsed_reg["game_id"])))
    live_cmp = live_df[live_df["game_id"].isin(common)].sort_values("game_id").reset_index(drop=True)
    mat_cmp = parsed_reg[parsed_reg["game_id"].isin(common)].sort_values("game_id").reset_index(drop=True)

    for col in FEATURE_COLS:
        assert np.allclose(
            pd.to_numeric(live_cmp[col], errors="coerce").values,
            pd.to_numeric(mat_cmp[col], errors="coerce").values,
            equal_nan=True,
            atol=1e-9,
        ), col


def test_materialized_rows_playoff_columns(monkeypatch, ctx):
    ctx_playoff = _ctx_with_playoffs(ctx)
    monkeypatch.setattr(playoff_builder, "_get_seeds_lookup", lambda: {(2024, HOME_TEAM): 2, (2024, AWAY_TEAM): 7})
    monkeypatch.setattr(
        playoff_builder,
        "_get_series_round_lookup",
        lambda: {(2024, frozenset((HOME_TEAM, AWAY_TEAM))): 1},
    )

    parsed = parse_materialized_rows(build_materialized_rows(ctx_playoff))
    playoff = parsed[parsed["game_type"] == 3]
    regular = parsed[parsed["game_type"] == 2]

    assert not playoff.empty
    assert playoff["series_game_number"].notna().all()
    assert playoff["home_series_wins"].notna().all()
    assert playoff["away_series_wins"].notna().all()
    assert playoff["seed_diff"].notna().all()

    assert regular["series_game_number"].isna().all()
    assert regular["home_series_wins"].isna().all()
    assert regular["away_series_wins"].isna().all()
    assert regular["seed_diff"].isna().all()


def test_parse_empty_and_training_fallback(monkeypatch, ctx):
    assert parse_materialized_rows([]).empty

    monkeypatch.setattr(training_builder, "get_materialized_game_features", lambda game_type=2: pd.DataFrame())
    monkeypatch.setattr(training_builder.DataContext, "from_supabase", lambda: ctx)

    df = training_builder.build_features(use_materialized=True)
    live = build_features_batch(ctx, game_type=2)
    assert len(df) == len(live)
    assert set(df["game_id"]) == set(live["game_id"])


def test_build_form_snapshot_rows_returns_expected_snapshot_keys(ctx):
    materialized_rows = build_materialized_rows(ctx)

    team_rows, goalie_rows = build_form_snapshot_rows(ctx, materialized_rows)

    assert isinstance(team_rows, list)
    assert isinstance(goalie_rows, list)
    assert team_rows
    assert goalie_rows

    team_row = team_rows[0]
    assert {"team_id", "as_of_date", "pp_pctg"}.issubset(team_row)


def test_clean_rows_normalizes_numpy_nan_and_dates():
    cleaned = _clean_rows(
        [
            {
                "int_value": np.int64(7),
                "float_value": np.float64(1.25),
                "nan_value": np.nan,
                "date_value": datetime.date(2024, 1, 2),
            }
        ]
    )

    row = cleaned[0]
    assert row == {
        "int_value": 7,
        "float_value": 1.25,
        "nan_value": None,
        "date_value": "2024-01-02",
    }
    assert json.dumps(row)
