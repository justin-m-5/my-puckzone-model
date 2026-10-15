# scripts/predict/players.py

"""
Project every player in a day's games and save to public.player_projections.

The slate runs this after it publishes, so goalie picks and each team's
expected goals come from the predictions you just made. Run it by hand to
refresh (it replaces the rows for those games):

    PYTHONPATH=. python3 -m scripts.predict.players                    # today
    PYTHONPATH=. python3 -m scripts.predict.players --date 2026-10-16
    PYTHONPATH=. python3 -m scripts.predict.players --game-id 2026020090 --dry-run

Lineups are the skaters who dressed in each team's most recent game; goalies
are the starters picked in the slate (no prediction = no goalie row).
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np
import pandas as pd

from db import supabase, upsert_rows
from features import player_projection as pp

TABLE = "player_projections"
BATCH = 500


def _season_of(day: datetime.date) -> int:
    start = day.year if day.month >= 7 else day.year - 1
    return start * 10000 + start + 1


def _fetch_games(day, game_ids):
    query = supabase.table("games").select("id,date,home_team_id,away_team_id,game_state")
    query = query.in_("id", game_ids) if game_ids else query.eq("date", day.isoformat()).in_("game_type", [2, 3])
    return pd.DataFrame(query.execute().data or [])


def _fetch_predictions(game_ids):
    res = (supabase.table("game_predictions")
           .select("game_id,home_goalie_id,away_goalie_id,home_win_prob,away_win_prob,pred_home_goals,pred_away_goals")
           .in_("game_id", game_ids).execute())
    return {r["game_id"]: r for r in res.data or []}


def _lineups(games, skaters):
    """Skaters from each team's most recent game before tonight."""
    rows = []
    for g in games.itertuples():
        for team, opp, home in [(g.home_team_id, g.away_team_id, True), (g.away_team_id, g.home_team_id, False)]:
            played = skaters[skaters["team_id"] == team]
            if played.empty:
                continue
            last_game = played.sort_values(["date", "game_id"])["game_id"].iloc[-1]
            dressed = played[played["game_id"] == last_game]
            rows.append(pd.DataFrame({
                "game_id": g.id, "player_id": dressed["player_id"].values, "team_id": team,
                "opponent_team_id": opp, "is_home": home, "position": dressed["position"].values,
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def project(day, game_ids=None, quiet=False):
    """Build projection rows for the day's games. Returns (skater_df, goalie_df)."""
    say = (lambda *a: None) if quiet else print
    games = _fetch_games(day, game_ids)
    if games.empty:
        say(f"No games on {day}.")
        return pd.DataFrame(), pd.DataFrame()
    day = pd.Timestamp(games["date"].iloc[0])
    preds = _fetch_predictions(games["id"].tolist())

    say("Loading player history...")
    hist_games, skaters, goalies = pp.load_history(_season_of(day.date()) - 10001)
    skaters, goalies = skaters[skaters["date"] < day], goalies[goalies["date"] < day]
    teams, league = pp.team_state(pp.team_games(hist_games[hist_games["date"] < day]))

    expected, allowed, win = {}, {}, {}
    for gid, p in preds.items():
        g = games.set_index("id").loc[gid]
        expected[(gid, g.home_team_id)], expected[(gid, g.away_team_id)] = p["pred_home_goals"], p["pred_away_goals"]
        allowed[(gid, g.home_team_id)], allowed[(gid, g.away_team_id)] = p["pred_away_goals"], p["pred_home_goals"]
        win[(gid, g.home_team_id)], win[(gid, g.away_team_id)] = p["home_win_prob"], p["away_win_prob"]

    state = pp.skater_state(skaters)
    lineups = _lineups(games, skaters)
    sk = pp.project_skaters(lineups, state, teams, league, expected)
    sk["games_sample"] = state["games"].reindex(sk["player_id"]).fillna(0).astype(int).values

    starters = []
    for gid, p in preds.items():
        g = games.set_index("id").loc[gid]
        for goalie, team, opp, home in [(p["home_goalie_id"], g.home_team_id, g.away_team_id, True),
                                        (p["away_goalie_id"], g.away_team_id, g.home_team_id, False)]:
            if goalie:
                starters.append({"game_id": gid, "player_id": int(goalie), "team_id": int(team),
                                 "opponent_team_id": int(opp), "is_home": home})
    gl = pd.DataFrame()
    if starters:
        gstate, league_sv = pp.goalie_state(goalies)
        gl = pp.project_goalies(pd.DataFrame(starters), gstate, league_sv, teams, league, allowed)
        gl["win_prob"] = [win.get((r.game_id, r.team_id)) for r in gl.itertuples()]
        gl["games_sample"] = gstate["starts"].reindex(gl["player_id"]).fillna(0).astype(int).values
        gl["team_expected_goals"] = [expected.get((r.game_id, r.team_id)) for r in gl.itertuples()]

    for df in (sk, gl):
        if not df.empty:
            df["game_date"] = day.date().isoformat()
    return sk, gl


SKATER_COLS = ["toi_min", "goals", "assists", "points", "shots", "hits", "blocks", "pim", "ixg",
               "p_goal", "p_point", "p_multi_point", "p_shots_3plus", "team_expected_goals"]
GOALIE_COLS = ["win_prob", "shots_against", "saves", "goals_against", "save_pct", "p_shutout", "team_expected_goals"]
KEY_COLS = ["game_id", "player_id", "game_date", "team_id", "opponent_team_id", "is_home", "games_sample"]


def _records(df, cols, kind):
    out = []
    for r in df.to_dict("records"):
        rec = {k: (int(r[k]) if k in ("game_id", "player_id", "team_id", "opponent_team_id", "games_sample")
                   else bool(r[k]) if k == "is_home" else r[k]) for k in KEY_COLS}
        rec["kind"] = kind
        rec["position"] = r.get("position") or "G"
        for c in cols:
            v = r.get(c)
            rec["toi_minutes" if c == "toi_min" else c] = None if v is None or pd.isna(v) else round(float(v), 4)
        out.append(rec)
    return out


def save(sk, gl):
    records = _records(sk, SKATER_COLS, "skater") + _records(gl, GOALIE_COLS, "goalie")
    game_ids = sorted({r["game_id"] for r in records})
    if not game_ids:
        return 0
    # Replace, so players no longer in a lineup drop out.
    supabase.table(TABLE).delete().in_("game_id", game_ids).execute()
    for i in range(0, len(records), BATCH):
        upsert_rows(TABLE, records[i:i + BATCH], on_conflict="game_id,player_id")
    return len(records)


def run(day, game_ids=None, dry_run=False, quiet=False):
    sk, gl = project(day, game_ids, quiet=quiet)
    if sk.empty and gl.empty:
        return 0
    if not quiet:
        top = sk.sort_values("points", ascending=False).head(10)
        names = _names(top["player_id"].tolist())
        print(f"\n{sk['game_id'].nunique()} games · {len(sk)} skaters · {len(gl)} goalies. Top projected points:")
        for r in top.itertuples():
            print(f"  {names.get(r.player_id, r.player_id):<24} {r.points:.2f} pts  {r.p_point * 100:.0f}% point"
                  f"  {r.p_goal * 100:.0f}% goal  {r.shots:.1f} SOG")
    if dry_run:
        print("Dry run: nothing saved.")
        return len(sk) + len(gl)
    n = save(sk, gl)
    if not quiet:
        print(f"Saved {n} projections.")
    return n


def _names(ids):
    if not ids:
        return {}
    res = supabase.table("players").select("id,first_name,last_name").in_("id", ids).execute()
    return {r["id"]: f"{r['first_name']} {r['last_name']}" for r in res.data or []}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Project players for a day's games.")
    parser.add_argument("--date", type=datetime.date.fromisoformat, default=None)
    parser.add_argument("--game-id", type=int, action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run(args.date or datetime.date.today(), args.game_id, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
