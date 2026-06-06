# tests/conftest.py
"""
Shared pytest fixtures for the PuckZone model test suite.

All fixtures provide in-memory DataContext objects so the pipeline can be
tested without a live Supabase connection.  The data is small and synthetic
but covers the structural patterns the pipeline exercises:

  - Two teams (home=1, away=2) with enough prior history to satisfy rolling
    window min_periods (3).
  - Goalie stats keyed to the correct (game_id, player_id) pairs.
  - Advanced share stats that differ between teams.
  - Standing snapshots dated before the target game.

Game IDs and dates are chosen so:
  - Games 1..10 pre-date the "target" game (game_id=11, date=2024-01-15).
  - Games 12+ post-date the target game and must NOT affect features.
"""

import datetime
import pandas as pd
import pytest

from features.pipeline import DataContext


# ---------------------------------------------------------------------------
# Constants used across fixtures
# ---------------------------------------------------------------------------

HOME_TEAM = 1
AWAY_TEAM = 2
GOALIE_HOME = 10   # player_id for the home team's starter
GOALIE_AWAY = 20   # player_id for the away team's starter
TARGET_GAME_ID = 11
TARGET_DATE = datetime.date(2024, 1, 15)
FUTURE_DATE = datetime.date(2024, 2, 1)
SEASON = 20232024


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_games(n=10, start_date=datetime.date(2023, 10, 15)):
    """Generate n completed games alternating home/away wins, plus the target."""
    rows = []
    d = start_date
    for i in range(1, n + 1):
        rows.append({
            "id": i,
            "season": SEASON,
            "game_type": 2,
            "date": d,
            "home_team_id": HOME_TEAM,
            "away_team_id": AWAY_TEAM,
            "home_score": 3,
            "away_score": 2,
        })
        d = d + datetime.timedelta(days=4)

    # The target game (game 11) — on TARGET_DATE, result known for training tests
    rows.append({
        "id": TARGET_GAME_ID,
        "season": SEASON,
        "game_type": 2,
        "date": TARGET_DATE,
        "home_team_id": HOME_TEAM,
        "away_team_id": AWAY_TEAM,
        "home_score": 4,
        "away_score": 1,
    })
    return pd.DataFrame(rows)


def _make_goalie_df(games_df):
    """One starter row per game for each team."""
    rows = []
    for _, g in games_df.iterrows():
        rows.append({
            "game_id": g["id"],
            "player_id": GOALIE_HOME,
            "team_id": HOME_TEAM,
            "is_home": True,
            "saves": 28,
            "shots_against": 30,
            "date": g["date"],
        })
        rows.append({
            "game_id": g["id"],
            "player_id": GOALIE_AWAY,
            "team_id": AWAY_TEAM,
            "is_home": False,
            "saves": 25,
            "shots_against": 30,
            "date": g["date"],
        })
    return pd.DataFrame(rows)


def _make_gsax_df(games_df):
    """GSAx entries keyed to game_id + player_id."""
    rows = []
    for _, g in games_df.iterrows():
        rows.append({
            "game_id": g["id"],
            "player_id": GOALIE_HOME,
            "team_id": HOME_TEAM,
            "date": g["date"],
            "gsax": 0.5,
        })
        rows.append({
            "game_id": g["id"],
            "player_id": GOALIE_AWAY,
            "team_id": AWAY_TEAM,
            "date": g["date"],
            "gsax": -0.3,
        })
    return pd.DataFrame(rows)


def _make_team_stats_df(games_df):
    """Simple team stats rows."""
    rows = []
    for _, g in games_df.iterrows():
        rows.append({
            "game_id": g["id"],
            "team_id": HOME_TEAM,
            "is_home": True,
            "date": g["date"],
            "pp_pctg": 0.22,
            "faceoff_winning_pctg": 0.51,
            "sog": 32,
            "hits": 22,
            "blocked_shots": 14,
        })
        rows.append({
            "game_id": g["id"],
            "team_id": AWAY_TEAM,
            "is_home": False,
            "date": g["date"],
            "pp_pctg": 0.18,
            "faceoff_winning_pctg": 0.49,
            "sog": 28,
            "hits": 20,
            "blocked_shots": 12,
        })
    return pd.DataFrame(rows)


def _make_advanced_df(games_df):
    """Advanced share stats per team per game."""
    rows = []
    for _, g in games_df.iterrows():
        rows.append({
            "game_id": g["id"],
            "team_id": HOME_TEAM,
            "season": SEASON,
            "date": g["date"],
            "cf_pct": 0.54,
            "xgf_pct": 0.55,
            "hdcf_pct": 0.53,
            "cf_pct_5v5": 0.53,
            "xgf_pct_5v5": 0.54,
            "hdcf_pct_5v5": 0.52,
        })
        rows.append({
            "game_id": g["id"],
            "team_id": AWAY_TEAM,
            "season": SEASON,
            "date": g["date"],
            "cf_pct": 0.46,
            "xgf_pct": 0.45,
            "hdcf_pct": 0.47,
            "cf_pct_5v5": 0.47,
            "xgf_pct_5v5": 0.46,
            "hdcf_pct_5v5": 0.48,
        })
    return pd.DataFrame(rows)


def _make_standings_df():
    """Daily standing snapshots for both teams, spanning the season."""
    rows = []
    base = datetime.date(2023, 10, 14)
    for i in range(100):
        d = base + datetime.timedelta(days=i)
        for team, pts, gf, ga in [(HOME_TEAM, 60, 120, 100), (AWAY_TEAM, 50, 110, 115)]:
            rows.append({
                "team_id": team,
                "season_id": SEASON,
                "as_of_date": d,
                "games_played": max(i // 4, 1),
                "point_pctg": 0.60 if team == HOME_TEAM else 0.50,
                "win_pctg": 0.55 if team == HOME_TEAM else 0.45,
                "regulation_win_pctg": 0.50 if team == HOME_TEAM else 0.40,
                "goal_for": gf,
                "goal_against": ga,
                "goal_differential": gf - ga,
                "l10_points": 12 if team == HOME_TEAM else 10,
                "home_wins": 15 if team == HOME_TEAM else 12,
                "home_losses": 8,
                "road_wins": 12 if team == HOME_TEAM else 10,
                "road_losses": 10,
                "points": pts,
                "wins": 30 if team == HOME_TEAM else 25,
                "losses": 15,
                "ot_losses": 5,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_games():
    """Ten historical games + the target game (game 11)."""
    return _make_games(n=10)


@pytest.fixture
def ctx(base_games):
    """DataContext built from in-memory fixtures (no Supabase required)."""
    goalie_df = _make_goalie_df(base_games)
    gsax_df = _make_gsax_df(base_games)
    team_stats_df = _make_team_stats_df(base_games)
    advanced_df = _make_advanced_df(base_games)
    standings = _make_standings_df()

    return DataContext(
        games=base_games,
        standings=standings,
        goalie_df=goalie_df,
        gsax_df=gsax_df,
        team_stats_df=team_stats_df,
        advanced_df=advanced_df,
    )


@pytest.fixture
def ctx_with_future(base_games):
    """DataContext that includes extra rows dated after TARGET_DATE.
    Used by the leakage test to verify that future data doesn't affect features.
    """
    # Extra games dated after TARGET_DATE
    future_games = pd.DataFrame([{
        "id": 100,
        "season": SEASON,
        "game_type": 2,
        "date": FUTURE_DATE,
        "home_team_id": HOME_TEAM,
        "away_team_id": AWAY_TEAM,
        "home_score": 5,
        "away_score": 0,
    }])
    all_games = pd.concat([base_games, future_games], ignore_index=True)

    goalie_df = _make_goalie_df(all_games)
    gsax_df = _make_gsax_df(all_games)
    team_stats_df = _make_team_stats_df(all_games)
    advanced_df = _make_advanced_df(all_games)
    standings = _make_standings_df()

    return DataContext(
        games=all_games,
        standings=standings,
        goalie_df=goalie_df,
        gsax_df=gsax_df,
        team_stats_df=team_stats_df,
        advanced_df=advanced_df,
    )
