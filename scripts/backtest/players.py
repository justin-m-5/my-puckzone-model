# scripts/backtest/players.py

"""
Walk-forward test of the player projections on a past season: every game day
is projected using only games before it, then compared with what happened
and with a simple baseline (the player's per-game average this season, or
last season's for the first few games).

Lineups are the players who actually dressed, and team context uses the
proxy (team goals for x opponent goals against), not the goals model, so
these numbers are a fair out-of-sample read on the player side alone.

Usage:
    PYTHONPATH=. python3 -m scripts.backtest.players                       # 2025-26
    PYTHONPATH=. python3 -m scripts.backtest.players --season 20252026 --every 3
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from features import player_projection as pp

STATS = [("goals", "goals"), ("assists", "assists"), ("points", "points"), ("shots", "sog"),
         ("hits", "hits"), ("blocks", "blocked_shots"), ("pim", "pim")]
MIN_SEASON_GAMES = 5


def _history(first_season, cache):
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    data = pp.load_history(first_season)
    if cache:
        with open(cache, "wb") as f:
            pickle.dump(data, f)
    return data


def _opponents(games):
    home = games[["game_id", "home_team_id", "away_team_id"]].rename(
        columns={"home_team_id": "team_id", "away_team_id": "opponent_team_id"})
    away = games[["game_id", "away_team_id", "home_team_id"]].rename(
        columns={"away_team_id": "team_id", "home_team_id": "opponent_team_id"})
    return pd.concat([home, away])


def _baseline(hist, players, season, cols):
    """Per-game average this season (MIN_SEASON_GAMES+), else last season, else position mean."""
    this = hist[hist["season"] == season].groupby("player_id")
    last = hist[hist["season"] == season - 10001].groupby("player_id")
    this_n = this.size()
    out = pd.DataFrame(index=players)
    group = hist.assign(g=np.where(hist["position"].isin(pp.FORWARDS), "F", "D")).groupby("g")[cols].mean()
    for c in cols:
        val = this[c].mean().where(this_n >= MIN_SEASON_GAMES).reindex(players)
        val = val.fillna(last[c].mean().reindex(players))
        out[c] = val
    return out, group


def run(season, first_season, every, cache):
    games, skaters, goalies = _history(first_season, cache)
    skaters = skaters.assign(points=skaters["goals"] + skaters["assists"])
    teams_all = pp.team_games(games)
    opp = _opponents(games)
    days = sorted(games.loc[games["season"] == season, "date"].unique())[::every]

    sk_rows, g_rows = [], []
    for day in days:
        hist = skaters[skaters["date"] < day]
        today = skaters[skaters["date"] == day].merge(opp, on=["game_id", "team_id"])
        if hist.empty or today.empty:
            continue
        state = pp.skater_state(hist)
        tstate, league = pp.team_state(teams_all[teams_all["date"] < day])
        proj = pp.project_skaters(today[["game_id", "player_id", "team_id", "opponent_team_id", "position"]],
                                  state, tstate, league)
        base, group_mean = _baseline(hist, today["player_id"], season, [c for _, c in STATS])
        g = np.where(today["position"].isin(pp.FORWARDS), "F", "D")
        actual = today.reset_index(drop=True)
        row = pd.DataFrame({"player_id": actual["player_id"], "games_before": state["games"].reindex(actual["player_id"]).fillna(0).values})
        for name, col in STATS:
            b = base[col].values
            b = np.where(np.isnan(b), group_mean.loc[g, col].values, b)
            row[f"{name}_pred"] = proj[name].values
            row[f"{name}_base"] = b
            row[f"{name}_actual"] = actual[col].values
        sk_rows.append(row)

        # Goalies: the actual starters.
        ghist = goalies[goalies["date"] < day]
        starts = goalies[(goalies["date"] == day) & goalies["starter"]].merge(opp, on=["game_id", "team_id"])
        if not ghist.empty and not starts.empty:
            gstate, league_sv = pp.goalie_state(ghist)
            gp = pp.project_goalies(starts[["game_id", "player_id", "team_id", "opponent_team_id"]],
                                    gstate, league_sv, tstate, league)
            this = ghist[(ghist["season"] == season) & ghist["starter"]].groupby("player_id")
            n = this.size()
            gbase = this[["shots_against", "saves", "goals_against"]].mean().where(n >= 3, axis=0)
            gbase = gbase.reindex(starts["player_id"])
            fill = ghist[ghist["starter"]][["shots_against", "saves", "goals_against"]].mean()
            gbase = gbase.fillna(fill)
            s = starts.reset_index(drop=True)
            g_rows.append(pd.DataFrame({
                "sa_pred": gp["shots_against"], "sa_base": gbase["shots_against"].values, "sa_actual": s["shots_against"],
                "sv_pred": gp["saves"], "sv_base": gbase["saves"].values, "sv_actual": s["saves"],
                "ga_pred": gp["goals_against"], "ga_base": gbase["goals_against"].values, "ga_actual": s["goals_against"],
                "so_pred": gp["p_shutout"], "so_actual": (s["goals_against"] == 0).astype(float),
            }))
        print(f"  {pd.Timestamp(day).date()}  {len(today)} skaters", end="\r")

    return pd.concat(sk_rows, ignore_index=True), pd.concat(g_rows, ignore_index=True)


def _poisson_dev(pred, actual):
    pred = np.maximum(pred, 1e-6)
    term = np.where(actual > 0, actual * np.log(np.maximum(actual, 1e-12) / pred), 0.0)
    return float(np.mean(2 * (term - (actual - pred))))


def report(sk, g):
    print(f"\nSkaters: {len(sk):,} player-games")
    print(f"  {'stat':<8}{'avg actual':>11}{'avg proj':>10}{'MAE proj':>10}{'MAE base':>10}{'dev proj':>10}{'dev base':>10}")
    for name, _ in STATS:
        a, p, b = sk[f"{name}_actual"], sk[f"{name}_pred"], sk[f"{name}_base"]
        print(f"  {name:<8}{a.mean():>11.3f}{p.mean():>10.3f}{np.abs(p - a).mean():>10.3f}{np.abs(b - a).mean():>10.3f}"
              f"{_poisson_dev(p, a):>10.4f}{_poisson_dev(b, a):>10.4f}")

    for name, label in [("goals", "scores"), ("points", "gets a point")]:
        actual = (sk[f"{name}_actual"] > 0).astype(float)
        p_model = 1 - np.exp(-sk[f"{name}_pred"])
        p_base = 1 - np.exp(-sk[f"{name}_base"])
        print(f"\n  Chance a player {label}: Brier proj {np.mean((p_model - actual) ** 2):.4f}"
              f"  base {np.mean((p_base - actual) ** 2):.4f}")
        bins = pd.cut(p_model, [0, .1, .2, .3, .4, .5, .6, .8, 1])
        table = pd.DataFrame({"bin": bins, "p": p_model, "hit": actual}).groupby("bin", observed=True).agg(
            n=("hit", "size"), projected=("p", "mean"), actual=("hit", "mean"))
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    print(f"\nGoalies: {len(g):,} starts")
    for key, label in [("sa", "shots against"), ("sv", "saves"), ("ga", "goals against")]:
        a, p, b = g[f"{key}_actual"], g[f"{key}_pred"], g[f"{key}_base"]
        print(f"  {label:<14} avg {a.mean():6.2f}  proj {p.mean():6.2f}  MAE proj {np.abs(p - a).mean():.3f}  base {np.abs(b - a).mean():.3f}")
    print(f"  shutouts       actual {g['so_actual'].mean():.3f}  projected {g['so_pred'].mean():.3f}"
          f"  Brier {np.mean((g['so_pred'] - g['so_actual']) ** 2):.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backtest player projections on a past season.")
    parser.add_argument("--season", type=int, default=20252026)
    parser.add_argument("--history-from", type=int, default=None, help="First season of history (default: two before).")
    parser.add_argument("--every", type=int, default=1, help="Test every Nth game day (faster).")
    parser.add_argument("--cache", default=None, help="Pickle path to reuse loaded history between runs.")
    args = parser.parse_args(argv)
    first = args.history_from or args.season - 20002
    sk, g = run(args.season, first, args.every, args.cache)
    report(sk, g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
