# scripts/predict/slate.py

"""
Predict a whole day's slate in one sitting, then publish after review.

For each game you pick the starting goalies (Enter = auto, which is the team's
most recent starter), every prediction is shown in one table, and nothing is
written to public.game_predictions until you confirm. Re-running a game
overwrites its prediction (upsert on game_id), so you can redo one after a
late goalie change with --game-id.

Usage:
    PYTHONPATH=. python3 -m scripts.predict.slate                     # today
    PYTHONPATH=. python3 -m scripts.predict.slate --date 2026-10-01
    PYTHONPATH=. python3 -m scripts.predict.slate --game-id 2026020003
"""

import argparse
import datetime

from db import fetch_all, supabase
from features.pipeline import DataContext
from scripts.predict.run import (
    GAME_SELECT_FIELDS,
    TEAM_BY_ID,
    compute_prediction,
    fetch_game_by_id,
    load_payloads,
    parse_game_date,
    persist_prediction,
    resolve_team_display,
)

REGULAR_SEASON_GAME_TYPE = 2
PLAYOFF_GAME_TYPE = 3
GOALIE_CHOICES = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict and publish a day's NHL slate.")
    parser.add_argument("--date", type=datetime.date.fromisoformat, default=None,
                        help="Slate date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--game-id", type=int, action="append", default=None,
                        help="Only these games (repeatable).")
    return parser.parse_args(argv)


def fetch_slate(target_date, game_ids=None):
    if game_ids:
        games = [fetch_game_by_id(gid) for gid in game_ids]
        return [g for g in games if g is not None]
    query = (
        supabase.table("games")
        .select(GAME_SELECT_FIELDS)
        .eq("date", target_date.isoformat())
        .in_("game_type", [REGULAR_SEASON_GAME_TYPE, PLAYOFF_GAME_TYPE])
        .order("id")
    )
    return fetch_all("games", query)


def fetch_player_names(player_ids):
    names = {}
    ids = [int(p) for p in player_ids]
    for i in range(0, len(ids), 200):
        rows = (
            supabase.table("players").select("id,first_name,last_name")
            .in_("id", ids[i:i + 200]).execute().data or []
        )
        for r in rows:
            names[r["id"]] = f"{r.get('first_name') or ''} {r.get('last_name') or ''}".strip()
    return names


class GoalieBook:
    """Goalie lookups for the prompts, built once from the loaded DataContext."""

    def __init__(self, goalie_df, as_of_date):
        self.prior = goalie_df[goalie_df["date"] < as_of_date]
        self.names = fetch_player_names(self.prior["player_id"].unique())
        last = self.prior.sort_values("date").groupby("player_id").tail(1)
        # player_id -> team of their most recent start
        self.current_team = dict(zip(last["player_id"], last["team_id"]))

    def name(self, player_id):
        return self.names.get(int(player_id)) or f"ID {player_id}"

    def team_options(self, team_id):
        """Recent starters for this team, most recent first: [(player_id, last_date, starts)]."""
        rows = self.prior[self.prior["team_id"] == team_id]
        if rows.empty:
            return []
        grouped = rows.groupby("player_id").agg(last=("date", "max"), starts=("date", "size"))
        grouped = grouped.sort_values("last", ascending=False).head(GOALIE_CHOICES)
        return [(int(pid), r["last"], int(r["starts"])) for pid, r in grouped.iterrows()]

    def search(self, text):
        text = text.lower()
        return [pid for pid, n in self.names.items() if text in n.lower()]


def team_abbr(team_id):
    team = TEAM_BY_ID.get(int(team_id))
    return team["abbr"] if team else str(team_id)


def confirm_goalie(book, pid, team_id, abbr):
    last_team = book.current_team.get(pid)
    if last_team is not None and int(last_team) != int(team_id):
        print(f"    Note: {book.name(pid)} is new to {abbr} (last started for {team_abbr(last_team)}).")
    return pid


def pick_goalie(book, team_id, abbr, label):
    options = book.team_options(team_id)
    print(f"\n  {label} goalie ({abbr}):")
    for i, (pid, last, starts) in enumerate(options, 1):
        auto = "  ← auto" if i == 1 else ""
        print(f"    {i}) {book.name(pid):<24} last start {last}  ({starts} starts){auto}")
    if not options:
        print("    (no prior starts for this team — auto falls back to league average)")

    while True:
        raw = input("    Enter=auto, number, player id, or name search: ").strip()
        if raw == "":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        if raw.isdigit():
            pid = int(raw)
            if pid in book.names or pid in book.current_team:
                return confirm_goalie(book, pid, team_id, abbr)
            print(f"    No prior starts found for id {pid}.")
            continue
        matches = book.search(raw)
        if len(matches) == 1:
            pid = matches[0]
            print(f"    → {book.name(pid)}")
            return confirm_goalie(book, pid, team_id, abbr)
        if not matches:
            print(f"    No goalie matching '{raw}'.")
            continue
        for pid in matches[:8]:
            print(f"      {pid}  {book.name(pid)}")
        print("    Several matches — enter the player id.")


def auto_goalie(book, team_id):
    options = book.team_options(team_id)
    return options[0][0] if options else None


def season_for(game_date):
    y = game_date.year
    return int(f"{y}{y + 1}") if game_date.month >= 10 else int(f"{y - 1}{y}")


def opening_note(standings, team_id, abbr, game_date):
    """Mirror of the pipeline's prior-season standings fallback, for display."""
    season = season_for(game_date)
    rows = standings[
        (standings["team_id"] == team_id)
        & (standings["season_id"] == season)
        & (standings["as_of_date"] < game_date)
    ]
    if rows.empty or rows.sort_values("as_of_date").iloc[-1]["games_played"] == 0:
        start = season // 10000
        print(f"  Note: {abbr} hasn't played in {start}-{str(start + 1)[2:]} yet; "
              f"using {start - 1}-{str(start)[2:]} final standings.")


def build_inputs(game, book, standings):
    game_type = int(game.get("game_type") or REGULAR_SEASON_GAME_TYPE)
    home_team_id = int(game["home_team_id"])
    away_team_id = int(game["away_team_id"])
    home_name, home_abbr = resolve_team_display(home_team_id)
    away_name, away_abbr = resolve_team_display(away_team_id)

    game_date = parse_game_date(game["date"])
    print(f"\n{'─' * 60}\n  {game['id']}  {away_abbr} @ {home_abbr}")
    opening_note(standings, home_team_id, home_abbr, game_date)
    opening_note(standings, away_team_id, away_abbr, game_date)
    home_goalie_id = pick_goalie(book, home_team_id, home_abbr, "Home")
    away_goalie_id = pick_goalie(book, away_team_id, away_abbr, "Away")

    return {
        "game_id": int(game["id"]),
        "game_date": game_date,
        "game_type": game_type,
        "game_type_label": "playoffs" if game_type == PLAYOFF_GAME_TYPE else "regular",
        "is_playoff": game_type == PLAYOFF_GAME_TYPE,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_name": home_name,
        "away_name": away_name,
        "home_abbr": home_abbr,
        "away_abbr": away_abbr,
        "home_goalie_id": home_goalie_id,
        "away_goalie_id": away_goalie_id,
    }


def goalie_label(book, inputs, side):
    pid = inputs[f"{side}_goalie_id"]
    if pid is not None:
        return book.name(pid)
    auto = auto_goalie(book, inputs[f"{side}_team_id"])
    return f"{book.name(auto)} (auto)" if auto is not None else "auto"


def print_slate(entries, book):
    print(f"\n{'=' * 96}")
    print(f"  {'Game':<11}{'Matchup':<12}{'Home win':>9}{'Away win':>10}  {'Score':<14}{'Pick':<6}Goalies (home / away)")
    print(f"{'-' * 96}")
    for i, (inputs, result, _) in enumerate(entries, 1):
        score = f"{inputs['home_abbr']} {result['home_gf']:.1f}-{result['away_gf']:.1f}"
        pick = inputs["home_abbr"] if result["winner_team_id"] == inputs["home_team_id"] else inputs["away_abbr"]
        matchup = f"{inputs['away_abbr']} @ {inputs['home_abbr']}"
        goalies = f"{goalie_label(book, inputs, 'home')} / {goalie_label(book, inputs, 'away')}"
        print(f"  {inputs['game_id']:<11}{matchup:<12}{result['home_prob'] * 100:>8.1f}%"
              f"{result['away_prob'] * 100:>9.1f}%  {score:<14}{pick:<6}{goalies}")
    print(f"{'=' * 96}")


def slate(argv=None):
    args = parse_args(argv)
    target_date = args.date or datetime.date.today()

    games = fetch_slate(target_date, args.game_id)
    if not games:
        print(f"No games found for {target_date}. Has ingest loaded the schedule?")
        return 1
    print(f"{len(games)} game(s) for {games[0]['date'] if args.game_id else target_date}")

    print("Loading data from Supabase (once for the whole slate)...")
    ctx = DataContext.from_supabase()
    book = GoalieBook(ctx.goalie_df, parse_game_date(games[0]["date"]))

    payloads = {}
    entries = []
    skipped = []
    for game in games:
        inputs = build_inputs(game, book, ctx.standings)
        is_playoff = inputs["is_playoff"]
        if is_playoff not in payloads:
            payloads[is_playoff] = load_payloads(is_playoff)
        try:
            result, debug = compute_prediction(inputs, *payloads[is_playoff], ctx=ctx)
        except SystemExit:
            # build_prediction_row exits when standings are missing for a team.
            skipped.append(inputs["game_id"])
            print(f"  Skipping {inputs['game_id']}: not enough data to build features.")
            continue
        entries.append((inputs, result, debug))

    if not entries:
        print("\nNo predictions could be made.")
        return 1

    print_slate(entries, book)
    if skipped:
        print(f"  Skipped (no prediction): {', '.join(map(str, skipped))}")

    answer = input(f"\nPublish {len(entries)} prediction(s) to the app? [y/N]: ").strip().lower()
    if answer != "y":
        print("Nothing published.")
        return 0

    failed = 0
    for inputs, result, _ in entries:
        try:
            persist_prediction(inputs, result)
            print(f"  published {inputs['game_id']}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED {inputs['game_id']}: {exc}")
    print(f"\nDone — {len(entries) - failed} published, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(slate())
