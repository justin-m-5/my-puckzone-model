# tests/test_player_projection.py

import numpy as np
import pandas as pd

from features import player_projection as pp


def skater_rows(player_id, position, games, toi=18.0, goals=0, team_id=1, start="2025-10-01"):
    dates = pd.date_range(start, periods=games, freq="2D")
    return pd.DataFrame({
        "game_id": range(player_id * 1000, player_id * 1000 + games), "player_id": player_id, "team_id": team_id,
        "position": position, "date": dates, "toi_min": toi, "goals": goals, "assists": 0, "sog": 2, "hits": 1,
        "blocked_shots": 1, "pim": 0, "ixg": 0.2,
    })


def league_rows(n=60):
    # A big pool so the position prior is ~0.5 goals / 60 for forwards.
    rows = [skater_rows(100 + i, "C", n, goals=np.tile([1, 0, 0, 0], n // 4)) for i in range(20)]
    rows += [skater_rows(200 + i, "D", n, toi=22.0) for i in range(10)]
    return pd.concat(rows, ignore_index=True)


def teams():
    t = pd.DataFrame({"gf": [3.0, 3.0], "ga": [3.0, 3.0], "sf": [30.0, 30.0], "sa": [30.0, 30.0], "n": [20, 20]},
                     index=pd.Index([1, 2], name="team_id"))
    return t, {"goals": 3.0, "shots": 30.0}


def test_short_hot_streak_is_pulled_toward_the_position_average():
    hist = pd.concat([league_rows(), skater_rows(1, "C", 3, goals=2)], ignore_index=True)
    state = pp.skater_state(hist)
    hot = state.loc[1, "goals_per60"]
    assert 0.6 < hot < 6.0          # 2 goals a game raw = 6.7 / 60; shrunk well below
    assert hot > state.loc[100, "goals_per60"]


def test_expected_goals_above_usual_scales_scoring_but_not_hits():
    hist = league_rows()
    state = pp.skater_state(hist)
    t, league = teams()
    lineup = pd.DataFrame({"game_id": [9, 9], "player_id": [100, 100], "team_id": [1, 1],
                           "opponent_team_id": [2, 2], "is_home": [True, True], "position": ["C", "C"]})
    base = pp.project_skaters(lineup.iloc[:1], state, t, league)
    boosted = pp.project_skaters(lineup.iloc[:1], state, t, league, {(9, 1): 3.6})
    assert np.isclose(boosted["goals"].iloc[0] / base["goals"].iloc[0], 1.2)
    assert np.isclose(boosted["hits"].iloc[0], base["hits"].iloc[0])
    assert 0 < boosted["p_goal"].iloc[0] < 1


def test_unknown_player_gets_position_defaults():
    state = pp.skater_state(league_rows())
    t, league = teams()
    lineup = pd.DataFrame({"game_id": [9], "player_id": [999], "team_id": [1], "opponent_team_id": [2],
                           "is_home": [True], "position": ["D"]})
    out = pp.project_skaters(lineup, state, t, league)
    assert np.isclose(out["toi_min"].iloc[0], 22.0)
    assert out["shots"].iloc[0] > 0


def test_goalie_uses_model_goals_against_and_scaled_shutout_chance():
    t, league = teams()
    gstate = pd.DataFrame({"save_pct": [0.91], "starts": [30]}, index=pd.Index([50], name="player_id"))
    starters = pd.DataFrame({"game_id": [9], "player_id": [50], "team_id": [1], "opponent_team_id": [2], "is_home": [True]})
    out = pp.project_goalies(starters, gstate, 0.9, t, league, {(9, 1): 2.0})
    assert np.isclose(out["shots_against"].iloc[0], 30.0)
    assert np.isclose(out["saves"].iloc[0], 28.0)
    assert np.isclose(out["p_shutout"].iloc[0], np.exp(-2.0) * pp.SHUTOUT_SCALE)
