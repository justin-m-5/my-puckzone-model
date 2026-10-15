# features/player_projection.py

"""
Per-player projections for one game: goals, assists, points, shots, hits,
blocks, PIM and individual xG for skaters; shots against, saves, goals
against and shutout chance for goalies.

How a skater's number is built:
  1. Ice time: recent games weighted heavily (half-life TOI_HALFLIFE games).
  2. Rate per 60: a long, recency-weighted history (half-life RATE_HALFLIFE
     games) pulled toward the position average by PRIOR_MINUTES of average
     play, so a hot week or a short sample doesn't run away.
  3. Expected count = rate x projected minutes, then adjusted for the game:
     scoring stats scale with the team's expected goals tonight (from the
     goals model when a prediction exists) vs the team's usual output, and
     shots with how many shots the opponent usually allows.
Chances (to score, to get a point, 3+ shots) treat each count as Poisson.

Everything here is plain pandas on history before the game, so the same
code serves the daily run (scripts.predict.players) and the backtest
(scripts.backtest.players).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from db import fetch_all, supabase

SKATER_STATS = ["goals", "assists", "sog", "hits", "blocked_shots", "pim", "ixg"]

TOI_HALFLIFE = 8          # games
RATE_HALFLIFE = 60        # games
TEAM_HALFLIFE = 20        # games
GOALIE_HALFLIFE = 40      # starts

# Minutes of position-average play mixed into each player's rate.
PRIOR_MINUTES = {"goals": 500, "assists": 400, "sog": 150, "hits": 80, "blocked_shots": 150, "pim": 400, "ixg": 300}
# Shots of league-average save % mixed into each goalie's.
GOALIE_PRIOR_SHOTS = 1500

# Shutouts happen about 60% as often as Poisson(goals against) says (empty-net
# goals, pulled goalies); measured on the 2025-26 backtest.
SHUTOUT_SCALE = 0.6

OFFENSE_CLIP = (0.75, 1.33)
SHOTS_CLIP = (0.85, 1.15)

FORWARDS = {"C", "L", "R"}


# --------------------------------------------------------------------------- data

def _parse_toi(value) -> float:
    if not value or pd.isna(value):
        return 0.0
    try:
        minutes, seconds = str(value).split(":")
        return int(minutes) + int(seconds) / 60
    except ValueError:
        return 0.0


def load_history(first_season: int):
    """Completed regular-season and playoff games from first_season on."""
    start_id = (first_season // 10000) * 1_000_000
    games = pd.DataFrame(fetch_all("games", (
        supabase.table("games")
        .select("id,season,game_type,date,start_time_utc,home_team_id,away_team_id,home_score,away_score,home_sog,away_sog")
        .gte("id", start_id).in_("game_type", [2, 3]).in_("game_state", ["OFF", "FINAL"])
    ))).rename(columns={"id": "game_id"})
    games["date"] = pd.to_datetime(games["date"])
    game_ids = set(games["game_id"])

    skaters = pd.DataFrame(fetch_all("game_skater_stats", (
        supabase.table("game_skater_stats")
        .select("game_id,player_id,team_id,position,toi,goals,assists,sog,hits,blocked_shots,pim")
        .gte("game_id", start_id)
    )))
    skaters = skaters[skaters["game_id"].isin(game_ids) & (skaters["position"] != "G")]
    ixg = pd.DataFrame(fetch_all("game_skater_advanced_stats", (
        supabase.table("game_skater_advanced_stats").select("game_id,player_id,ixg").gte("game_id", start_id)
    )))
    skaters = skaters.merge(ixg, on=["game_id", "player_id"], how="left")
    skaters["ixg"] = skaters["ixg"].fillna(0.0)
    skaters["toi_min"] = skaters["toi"].map(_parse_toi)
    skaters = skaters[skaters["toi_min"] > 0]
    skaters = skaters.merge(games[["game_id", "date", "season"]], on="game_id")

    goalies = pd.DataFrame(fetch_all("game_goalie_stats", (
        supabase.table("game_goalie_stats")
        .select("game_id,player_id,team_id,starter,toi,shots_against,saves,goals_against,decision")
        .gte("game_id", start_id)
    )))
    goalies = goalies[goalies["game_id"].isin(game_ids)]
    goalies["toi_min"] = goalies["toi"].map(_parse_toi)
    goalies = goalies[goalies["toi_min"] > 0].merge(games[["game_id", "date", "season"]], on="game_id")

    return games, skaters, goalies


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: goals and shots for and against."""
    home = games.assign(team_id=games["home_team_id"], gf=games["home_score"], ga=games["away_score"],
                        sf=games["home_sog"], sa=games["away_sog"])
    away = games.assign(team_id=games["away_team_id"], gf=games["away_score"], ga=games["home_score"],
                        sf=games["away_sog"], sa=games["home_sog"])
    cols = ["game_id", "date", "team_id", "gf", "ga", "sf", "sa"]
    return pd.concat([home[cols], away[cols]]).sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- state

def _ewm_last(df: pd.DataFrame, key: str, cols: list[str], halflife: float) -> pd.DataFrame:
    """Recency-weighted mean of cols per key (most recent row weighs most), plus games counted."""
    df = df.sort_values([key, "date", "game_id"])
    means = df.groupby(key)[cols].ewm(halflife=halflife).mean()
    last = means.groupby(level=0).tail(1).droplevel(1)
    last["n"] = df.groupby(key).size()
    return last


def _weight_total(n, halflife):
    """Sum of the ewm weights over n games: (1 - d^n) / (1 - d)."""
    d = 0.5 ** (1 / halflife)
    return (1 - d ** n) / (1 - d)


def league_rates(skaters: pd.DataFrame) -> dict:
    """Per-60 averages by position group (F/D), the prior every player is pulled toward."""
    group = np.where(skaters["position"].isin(FORWARDS), "F", "D")
    totals = skaters.groupby(group)[SKATER_STATS + ["toi_min"]].sum()
    rates = {g: {s: totals.loc[g, s] / totals.loc[g, "toi_min"] * 60 for s in SKATER_STATS} for g in totals.index}
    per_game_toi = skaters.groupby(group)["toi_min"].median()
    for g in rates:
        rates[g]["toi_min"] = float(per_game_toi[g])
    return rates


def skater_state(skaters: pd.DataFrame) -> pd.DataFrame:
    """Per player: projected minutes and shrunk per-60 rates, as of the end of `skaters`."""
    prior = league_rates(skaters)
    toi = _ewm_last(skaters, "player_id", ["toi_min"], TOI_HALFLIFE)
    rate = _ewm_last(skaters, "player_id", SKATER_STATS + ["toi_min"], RATE_HALFLIFE)
    weight = _weight_total(rate["n"], RATE_HALFLIFE)

    last = skaters.sort_values(["date", "game_id"]).groupby("player_id").tail(1).set_index("player_id")
    state = pd.DataFrame(index=rate.index)
    state["position"] = last["position"]
    state["last_team_id"] = last["team_id"]
    state["games"] = rate["n"]
    state["toi_min"] = toi["toi_min"]
    group = np.where(state["position"].isin(FORWARDS), "F", "D")
    minutes = rate["toi_min"] * weight          # weighted minutes played
    for s in SKATER_STATS:
        prior_rate = np.array([prior[g][s] for g in group])
        events = rate[s] * weight
        k = PRIOR_MINUTES[s]
        state[f"{s}_per60"] = (events + prior_rate * k / 60) / (minutes + k) * 60
    state.attrs["prior"] = prior
    return state


def team_state(team_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per team recency-weighted goals and shots for/against per game, and league averages."""
    state = _ewm_last(team_rows, "team_id", ["gf", "ga", "sf", "sa"], TEAM_HALFLIFE)
    league = {"goals": float(team_rows["gf"].mean()), "shots": float(team_rows["sf"].mean())}
    return state, league


def goalie_state(goalies: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per goalie: shrunk save % from recency-weighted shots and saves."""
    league_sv = float(goalies["saves"].sum() / max(goalies["shots_against"].sum(), 1))
    rate = _ewm_last(goalies, "player_id", ["shots_against", "saves"], GOALIE_HALFLIFE)
    weight = _weight_total(rate["n"], GOALIE_HALFLIFE)
    shots, saves = rate["shots_against"] * weight, rate["saves"] * weight
    state = pd.DataFrame(index=rate.index)
    state["save_pct"] = (saves + league_sv * GOALIE_PRIOR_SHOTS) / (shots + GOALIE_PRIOR_SHOTS)
    state["starts"] = rate["n"]
    return state, league_sv


# --------------------------------------------------------------------------- projections

def _poisson_at_least(lam, k):
    """P(X >= k) for Poisson(lam), k small."""
    lam = np.asarray(lam, dtype=float)
    term = np.exp(-lam)
    below = np.zeros_like(lam)
    for i in range(k):
        below += term
        term = term * lam / (i + 1)
    return 1 - below


def project_skaters(lineups: pd.DataFrame, state: pd.DataFrame, teams: pd.DataFrame, league: dict,
                    team_expected_goals: dict | None = None) -> pd.DataFrame:
    """
    lineups: game_id, player_id, team_id, opponent_team_id, is_home, position (one row per skater).
    team_expected_goals: {(game_id, team_id): goals} from the goals model; when a
    team is missing, the opponent's usual goals against stands in.
    """
    prior = state.attrs["prior"]
    out = lineups.reset_index(drop=True).copy()
    s = state.reindex(out["player_id"]).reset_index(drop=True)
    out["position"] = s["position"].fillna(out["position"])
    group = np.where(out["position"].isin(FORWARDS), "F", "D")
    out["toi_min"] = s["toi_min"].values
    missing_toi = out["toi_min"].isna()
    out.loc[missing_toi, "toi_min"] = [prior[g]["toi_min"] for g in group[missing_toi.values]]

    t = teams.reindex(out["team_id"]).reset_index(drop=True)
    o = teams.reindex(out["opponent_team_id"]).reset_index(drop=True)
    team_gf = t["gf"].fillna(league["goals"]).values
    opp_ga = o["ga"].fillna(league["goals"]).values
    opp_sa = o["sa"].fillna(league["shots"]).values

    proxy = team_gf * opp_ga / league["goals"]
    expected = np.array([
        (team_expected_goals or {}).get((g, tm), np.nan) for g, tm in zip(out["game_id"], out["team_id"])
    ], dtype=float)
    expected = np.where(np.isnan(expected), proxy, expected)
    offense = np.clip(expected / team_gf, *OFFENSE_CLIP)
    shots = np.clip(opp_sa / league["shots"], *SHOTS_CLIP)

    minutes = out["toi_min"].values
    for stat, name in [("goals", "goals"), ("assists", "assists"), ("sog", "shots"), ("hits", "hits"),
                       ("blocked_shots", "blocks"), ("pim", "pim"), ("ixg", "ixg")]:
        rate = s[f"{stat}_per60"].values
        fallback = np.array([prior[g][stat] for g in group])
        rate = np.where(np.isnan(rate), fallback, rate)
        factor = offense if stat in ("goals", "assists", "ixg") else shots if stat == "sog" else 1.0
        out[name] = rate * minutes / 60 * factor

    out["points"] = out["goals"] + out["assists"]
    out["p_goal"] = _poisson_at_least(out["goals"], 1)
    out["p_point"] = _poisson_at_least(out["points"], 1)
    out["p_multi_point"] = _poisson_at_least(out["points"], 2)
    out["p_shots_3plus"] = _poisson_at_least(out["shots"], 3)
    out["team_expected_goals"] = expected
    return out


def project_goalies(starters: pd.DataFrame, gstate: pd.DataFrame, league_sv: float, teams: pd.DataFrame,
                    league: dict, goals_against: dict | None = None) -> pd.DataFrame:
    """
    starters: game_id, player_id, team_id, opponent_team_id, is_home.
    goals_against: {(game_id, team_id): goals this team is expected to allow} from
    the goals model; otherwise expected shots x (1 - the goalie's save %).
    """
    out = starters.reset_index(drop=True).copy()
    t = teams.reindex(out["team_id"]).reset_index(drop=True)
    o = teams.reindex(out["opponent_team_id"]).reset_index(drop=True)
    sa = t["sa"].fillna(league["shots"]).values * o["sf"].fillna(league["shots"]).values / league["shots"]
    sv = gstate["save_pct"].reindex(out["player_id"]).fillna(league_sv).values

    ga = np.array([(goals_against or {}).get((g, tm), np.nan) for g, tm in zip(out["game_id"], out["team_id"])],
                  dtype=float)
    ga = np.where(np.isnan(ga), sa * (1 - sv), ga)

    out["shots_against"] = sa
    out["goals_against"] = ga
    out["saves"] = np.maximum(sa - ga, 0)
    out["save_pct"] = out["saves"] / out["shots_against"]
    out["p_shutout"] = np.exp(-ga) * SHUTOUT_SCALE
    return out
