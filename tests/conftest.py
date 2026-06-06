# tests/conftest.py
"""
Shared pytest fixtures for the PuckZone model test suite.

The fixture data intentionally varies game-to-game (goalie, team stats, and
advanced metrics) so rolling windows and shift/off-by-one behavior are
meaningfully tested.
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
ALT_AWAY_TEAM = 3

GOALIE_HOME = 10
GOALIE_AWAY = 20
GOALIE_ALT_AWAY = 30

TARGET_GAME_ID = 13
TARGET_DATE = datetime.date(2024, 1, 15)

ALT_TARGET_GAME_ID = 105
ALT_TARGET_DATE = datetime.date(2024, 1, 12)

FUTURE_DATE = datetime.date(2024, 2, 1)
SEASON = 20232024

GOALIE_BY_TEAM = {
    HOME_TEAM: GOALIE_HOME,
    AWAY_TEAM: GOALIE_AWAY,
    ALT_AWAY_TEAM: GOALIE_ALT_AWAY,
}


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_games(start_date=datetime.date(2023, 10, 15)):
    """Build synthetic schedule with varied prior-meeting counts."""
    rows = []

    # 12 prior HOME vs AWAY meetings before TARGET_DATE.
    # First 2 are away wins, next 10 are home wins. This intentionally creates
    # a 10/12 uncapped H2H baseline that differs from a capped last-10 (1.0)
    # baseline, so cap-vs-uncap regressions are detectable.
    d = start_date
    for i in range(1, 13):
        home_score = 2 if i <= 2 else 4
        away_score = 3 if i <= 2 else 2
        rows.append({
            "id": i,
            "season": SEASON,
            "game_type": 2,
            "date": d,
            "home_team_id": HOME_TEAM,
            "away_team_id": AWAY_TEAM,
            "home_score": home_score,
            "away_score": away_score,
        })
        d += datetime.timedelta(days=4)

    # 4 prior HOME vs ALT_AWAY meetings before ALT_TARGET_DATE.
    for i, date in enumerate(
        [
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 4),
            datetime.date(2024, 1, 7),
            datetime.date(2024, 1, 10),
        ],
        start=1,
    ):
        rows.append({
            "id": 100 + i,
            "season": SEASON,
            "game_type": 2,
            "date": date,
            "home_team_id": HOME_TEAM,
            "away_team_id": ALT_AWAY_TEAM,
            "home_score": 3 if i != 2 else 1,
            "away_score": 1 if i != 2 else 2,
        })

    # Alternate target game (different prior-meeting count).
    rows.append({
        "id": ALT_TARGET_GAME_ID,
        "season": SEASON,
        "game_type": 2,
        "date": ALT_TARGET_DATE,
        "home_team_id": HOME_TEAM,
        "away_team_id": ALT_AWAY_TEAM,
        "home_score": 4,
        "away_score": 2,
    })

    # Main target game.
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
    """One starter row per team per game with varied saves/shots."""
    rows = []
    ordered = games_df.sort_values("date").reset_index(drop=True)
    for idx, g in ordered.iterrows():
        game_idx = idx + 1
        home_team_id = int(g["home_team_id"])
        away_team_id = int(g["away_team_id"])
        rows.append({
            "game_id": g["id"],
            "player_id": GOALIE_BY_TEAM[home_team_id],
            "team_id": home_team_id,
            "is_home": True,
            "saves": 24 + (game_idx % 7),
            "shots_against": 29 + (game_idx % 4),
            "date": g["date"],
        })
        rows.append({
            "game_id": g["id"],
            "player_id": GOALIE_BY_TEAM[away_team_id],
            "team_id": away_team_id,
            "is_home": False,
            "saves": 23 + ((game_idx + away_team_id) % 6),
            "shots_against": 28 + ((game_idx + away_team_id) % 5),
            "date": g["date"],
        })
    return pd.DataFrame(rows)


def _make_gsax_df(games_df):
    """GSAx entries keyed to game_id + player_id with game-to-game variation."""
    rows = []
    ordered = games_df.sort_values("date").reset_index(drop=True)
    for idx, g in ordered.iterrows():
        game_idx = idx + 1
        for team_id, is_home in [(int(g["home_team_id"]), True), (int(g["away_team_id"]), False)]:
            base = 0.15 if team_id == HOME_TEAM else (-0.05 if team_id == AWAY_TEAM else -0.12)
            trend = 0.03 * ((game_idx % 5) - 2)
            rows.append({
                "game_id": g["id"],
                "player_id": GOALIE_BY_TEAM[team_id],
                "team_id": team_id,
                "date": g["date"],
                "gsax": base + trend + (0.02 if is_home else -0.01),
            })
    return pd.DataFrame(rows)


def _make_team_stats_df(games_df):
    """Team stats rows with varied values across games."""
    rows = []
    ordered = games_df.sort_values("date").reset_index(drop=True)
    for idx, g in ordered.iterrows():
        game_idx = idx + 1
        for team_id, is_home in [(int(g["home_team_id"]), True), (int(g["away_team_id"]), False)]:
            if team_id == HOME_TEAM:
                pp = 0.19 + 0.005 * (game_idx % 6)
                fo = 0.49 + 0.004 * (game_idx % 5)
                sog = 30 + (game_idx % 6)
                hits = 18 + (game_idx % 7)
                blocks = 11 + (game_idx % 5)
            elif team_id == AWAY_TEAM:
                pp = 0.16 + 0.006 * ((game_idx + 1) % 5)
                fo = 0.47 + 0.003 * ((game_idx + 2) % 4)
                sog = 27 + ((game_idx + 1) % 5)
                hits = 19 + ((game_idx + 2) % 6)
                blocks = 10 + ((game_idx + 1) % 4)
            else:
                pp = 0.17 + 0.004 * ((game_idx + 3) % 5)
                fo = 0.48 + 0.003 * ((game_idx + 1) % 5)
                sog = 28 + ((game_idx + 2) % 5)
                hits = 17 + ((game_idx + 3) % 6)
                blocks = 9 + ((game_idx + 2) % 4)

            rows.append({
                "game_id": g["id"],
                "team_id": team_id,
                "is_home": is_home,
                "date": g["date"],
                "pp_pctg": pp,
                "faceoff_winning_pctg": fo,
                "sog": sog,
                "hits": hits,
                "blocked_shots": blocks,
            })
    return pd.DataFrame(rows)


def _make_advanced_df(games_df):
    """Advanced share stats per team per game with varied values."""
    rows = []
    ordered = games_df.sort_values("date").reset_index(drop=True)
    for idx, g in ordered.iterrows():
        game_idx = idx + 1
        for team_id in [int(g["home_team_id"]), int(g["away_team_id"])]:
            if team_id == HOME_TEAM:
                cf = 0.50 + 0.008 * ((game_idx % 6) - 2)
                xgf = 0.51 + 0.007 * ((game_idx % 5) - 2)
                hdcf = 0.50 + 0.009 * ((game_idx % 4) - 1)
            elif team_id == AWAY_TEAM:
                cf = 0.47 + 0.006 * ((game_idx % 5) - 2)
                xgf = 0.46 + 0.005 * ((game_idx % 4) - 1)
                hdcf = 0.48 + 0.007 * ((game_idx % 6) - 3)
            else:
                cf = 0.49 + 0.005 * ((game_idx % 4) - 1)
                xgf = 0.48 + 0.006 * ((game_idx % 6) - 2)
                hdcf = 0.47 + 0.005 * ((game_idx % 5) - 2)

            rows.append({
                "game_id": g["id"],
                "team_id": team_id,
                "season": SEASON,
                "date": g["date"],
                "cf_pct": cf,
                "xgf_pct": xgf,
                "hdcf_pct": hdcf,
                "cf_pct_5v5": cf - 0.01,
                "xgf_pct_5v5": xgf - 0.01,
                "hdcf_pct_5v5": hdcf - 0.01,
            })
    return pd.DataFrame(rows)


def _make_standings_df():
    """Daily standing snapshots for all fixture teams."""
    rows = []
    base = datetime.date(2023, 10, 14)
    team_meta = {
        HOME_TEAM: {"points": 62, "gf": 122, "ga": 101},
        AWAY_TEAM: {"points": 54, "gf": 114, "ga": 117},
        ALT_AWAY_TEAM: {"points": 50, "gf": 110, "ga": 116},
    }

    for i in range(120):
        d = base + datetime.timedelta(days=i)
        for team_id, meta in team_meta.items():
            rows.append({
                "team_id": team_id,
                "season_id": SEASON,
                "as_of_date": d,
                "games_played": max(i // 4, 1),
                "point_pctg": (meta["points"] / 100) + (0.001 * (i % 3)),
                "win_pctg": 0.45 + (0.02 if team_id == HOME_TEAM else 0.0),
                "regulation_win_pctg": 0.40 + (0.02 if team_id == HOME_TEAM else 0.0),
                "goal_for": meta["gf"] + (i % 6),
                "goal_against": meta["ga"] + (i % 5),
                "goal_differential": (meta["gf"] - meta["ga"]) + ((i % 6) - (i % 5)),
                "l10_points": 9 + (i % 5) + (1 if team_id == HOME_TEAM else 0),
                "home_wins": 14 + (1 if team_id == HOME_TEAM else 0),
                "home_losses": 8 + (0 if team_id == HOME_TEAM else 1),
                "road_wins": 11 + (1 if team_id == HOME_TEAM else 0),
                "road_losses": 10 + (0 if team_id == HOME_TEAM else 1),
                "points": meta["points"] + (i % 6),
                "wins": 28 + (2 if team_id == HOME_TEAM else 0),
                "losses": 16 + (0 if team_id == HOME_TEAM else 1),
                "ot_losses": 5,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_games():
    """Synthetic completed regular-season games with two target matchups."""
    return _make_games()


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
