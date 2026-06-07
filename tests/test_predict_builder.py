import datetime
import pandas as pd

from features.pipeline import DataContext
from scripts.predict import builder as predict_builder
from tests.conftest import HOME_TEAM, AWAY_TEAM, TARGET_DATE

MOCK_STALE_SERIES_WINS = 99


def _ctx_with_games(ctx, games_df):
    return DataContext(
        games=games_df,
        standings=ctx.standings,
        goalie_df=ctx.goalie_df,
        gsax_df=ctx.gsax_df,
        team_stats_df=ctx.team_stats_df,
        advanced_df=ctx.advanced_df,
        skater_df=ctx.skater_df,
    )


def test_playoff_prediction_row_adds_series_columns_and_uses_bracket_year(ctx, monkeypatch):
    playoff_games = pd.DataFrame(
        [
            {
                "id": 9001,
                "season": 20232024,
                "game_type": 3,
                "date": datetime.date(2024, 1, 2),
                "home_team_id": HOME_TEAM,
                "away_team_id": AWAY_TEAM,
                "home_score": 3,
                "away_score": 1,
            },
            {
                "id": 9002,
                "season": 20232024,
                "game_type": 3,
                "date": datetime.date(2024, 1, 5),
                "home_team_id": AWAY_TEAM,
                "away_team_id": HOME_TEAM,
                "home_score": 4,
                "away_score": 2,
            },
        ]
    )
    ctx_playoff = _ctx_with_games(ctx, pd.concat([ctx.games, playoff_games], ignore_index=True))
    monkeypatch.setattr(predict_builder.DataContext, "from_supabase", lambda: ctx_playoff)

    captured = {}

    def _mock_series_context(home_team_id, away_team_id, bracket_year):
        captured["bracket_year"] = bracket_year
        return {
            "round_number": 1,
            "series_title": "Round 1",
            "series_abbrev": "R1",
            "team_a_wins": MOCK_STALE_SERIES_WINS,
            "team_b_wins": MOCK_STALE_SERIES_WINS,
            "team_a_seed": 2,
            "team_b_seed": 7,
            "series_clinched": False,
            "series_winner_id": None,
        }

    monkeypatch.setattr(predict_builder, "get_series_context", _mock_series_context)

    row, debug = predict_builder.build_prediction_row(HOME_TEAM, AWAY_TEAM, TARGET_DATE, is_playoff=True)

    assert captured["bracket_year"] == 2024
    assert row["home_series_wins"] == 1
    assert row["away_series_wins"] == 1
    assert row["series_game_number"] == 3
    assert row["seed_diff"] == -5

    assert debug["series"]["team_a_wins"] == 1
    assert debug["series"]["team_b_wins"] == 1


def test_regular_prediction_row_sets_neutral_playoff_defaults(ctx, monkeypatch):
    monkeypatch.setattr(predict_builder.DataContext, "from_supabase", lambda: ctx)

    row, debug = predict_builder.build_prediction_row(HOME_TEAM, AWAY_TEAM, TARGET_DATE, is_playoff=False)

    assert row["home_series_wins"] == 0
    assert row["away_series_wins"] == 0
    assert row["series_game_number"] == 1
    assert row["seed_diff"] == 0
    assert debug["series"] is None


def test_regular_prediction_h2h_uses_regular_season_only(ctx, monkeypatch):
    playoff_games = pd.DataFrame(
        [
            {
                "id": 9011,
                "season": 20232024,
                "game_type": 3,
                "date": datetime.date(2024, 1, 3),
                "home_team_id": HOME_TEAM,
                "away_team_id": AWAY_TEAM,
                "home_score": 1,
                "away_score": 4,
            },
            {
                "id": 9012,
                "season": 20232024,
                "game_type": 3,
                "date": datetime.date(2024, 1, 8),
                "home_team_id": AWAY_TEAM,
                "away_team_id": HOME_TEAM,
                "home_score": 3,
                "away_score": 0,
            },
        ]
    )
    ctx_mixed = _ctx_with_games(ctx, pd.concat([ctx.games, playoff_games], ignore_index=True))
    monkeypatch.setattr(predict_builder.DataContext, "from_supabase", lambda: ctx_mixed)

    row, _ = predict_builder.build_prediction_row(HOME_TEAM, AWAY_TEAM, TARGET_DATE, is_playoff=False)

    # Regular-season prior meetings are 12, with 10 HOME wins.
    assert row["h2h_home_win_pctg"] == 10 / 12
