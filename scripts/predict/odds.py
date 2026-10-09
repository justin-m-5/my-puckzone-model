# scripts/predict/odds.py

"""
Betting lines saved with each prediction (game_predictions.market_odds).

The slate fetches the Odds API once when you publish (3 credits: one region,
three markets) and stores the lines next to each prediction, so the web app
never calls the Odds API for predictions. The stored shape matches the app's
ClosingGameOdds type, plus "source": "publish" or "closing" (backfilled).

Backfill closing lines for games predicted before this existed (historical
snapshots cost 10 credits per market, so 30 per distinct start time):
    PYTHONPATH=. python3 -m scripts.predict.odds --backfill --season 20262027 --dry-run
    PYTHONPATH=. python3 -m scripts.predict.odds --backfill --season 20262027
"""

import argparse
import datetime
import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request

from db import fetch_all, supabase
from scripts.predict.teams import TEAMS

BASE_URL = os.getenv("THE_ODDS_API_BASE_URL", "https://api.the-odds-api.com").rstrip("/")
PARAMS = {"regions": "us", "oddsFormat": "american", "markets": "h2h,spreads,totals"}
PREFERRED_BOOKMAKER = "draftkings"
# An event matches a game only if it starts within this window of the game.
MATCH_WINDOW = datetime.timedelta(hours=6)
# Historical snapshot this long before puck drop, on a 5-minute grid.
CLOSING_LEAD = datetime.timedelta(minutes=5)

NAME_BY_TEAM_ID = {team_id: name for team_id, name in TEAMS.values()}


class OddsUnavailable(Exception):
    pass


def _api_key():
    key = os.getenv("THE_ODDS_API_KEY")
    if not key:
        raise OddsUnavailable("THE_ODDS_API_KEY is not set in .env")
    return key


def _get(path, **extra):
    query = urllib.parse.urlencode({**PARAMS, **extra, "apiKey": _api_key()})
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}?{query}", timeout=20) as res:
            remaining = res.headers.get("x-requests-remaining")
            return json.load(res), remaining
    except urllib.error.HTTPError as exc:
        raise OddsUnavailable(f"Odds API {exc.code}: {exc.read().decode(errors='replace')[:200]}") from exc
    except urllib.error.URLError as exc:
        raise OddsUnavailable(f"Odds API unreachable: {exc.reason}") from exc


def fetch_current_odds():
    """Upcoming (and in-progress) NHL events with lines. Returns (events, credits_remaining)."""
    return _get("/v4/sports/icehockey_nhl/odds")


def fetch_historical_odds(snapshot_iso):
    """Events as listed at snapshot_iso. Returns (events, credits_remaining)."""
    body, remaining = _get("/v4/historical/sports/icehockey_nhl/odds", date=snapshot_iso)
    return body.get("data") or [], remaining


def _norm(name):
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", plain.lower())


def _parse_time(value):
    if not value:
        return None
    return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def find_event(events, home_team_id, away_team_id, start_time_utc):
    """The event for this matchup closest to the game's start time, or None."""
    home = _norm(NAME_BY_TEAM_ID.get(home_team_id, ""))
    away = _norm(NAME_BY_TEAM_ID.get(away_team_id, ""))
    start = _parse_time(start_time_utc)
    best, best_gap = None, None
    for ev in events:
        if _norm(ev.get("home_team", "")) != home or _norm(ev.get("away_team", "")) != away:
            continue
        commence = _parse_time(ev.get("commence_time"))
        gap = abs(commence - start) if (commence and start) else datetime.timedelta(0)
        if gap > MATCH_WINDOW:
            continue
        if best_gap is None or gap < best_gap:
            best, best_gap = ev, gap
    return best


def to_market_odds(event, source):
    """One event -> the ClosingGameOdds-shaped dict the app reads."""
    books = event.get("bookmakers") or []
    book = next((b for b in books if b.get("key") == PREFERRED_BOOKMAKER), books[0] if books else None)
    if not book:
        return None
    markets = {m["key"]: m.get("outcomes") or [] for m in book.get("markets") or []}
    home, away = _norm(event["home_team"]), _norm(event["away_team"])

    def side(outcomes, team):
        return next((o for o in outcomes if _norm(o.get("name", "")) == team), None)

    ml_home, ml_away = side(markets.get("h2h", []), home), side(markets.get("h2h", []), away)
    sp_home, sp_away = side(markets.get("spreads", []), home), side(markets.get("spreads", []), away)
    over = next((o for o in markets.get("totals", []) if o.get("name") == "Over"), None)
    under = next((o for o in markets.get("totals", []) if o.get("name") == "Under"), None)

    return {
        "bookmaker": book.get("title") or book.get("key"),
        "lastUpdated": book.get("last_update"),
        "source": source,
        "moneyline": {"home": ml_home["price"], "away": ml_away["price"]} if ml_home and ml_away else None,
        "spread": {
            "homeLine": sp_home["point"], "homePrice": sp_home["price"],
            "awayLine": sp_away["point"], "awayPrice": sp_away["price"],
        } if sp_home and sp_away and "point" in sp_home and "point" in sp_away else None,
        "totals": {
            "overLine": over["point"], "overPrice": over["price"],
            "underLine": under["point"], "underPrice": under["price"],
        } if over and under and "point" in over and "point" in under else None,
    }


def fetch_start_times(game_ids):
    if not game_ids:
        return {}
    res = supabase.table("games").select("id,start_time_utc").in_("id", list(game_ids)).execute()
    return {row["id"]: row["start_time_utc"] for row in res.data or []}


def lines_for_games(games):
    """
    games: [{game_id, home_team_id, away_team_id}]. One API call for all of
    them. Returns ({game_id: market_odds}, credits_remaining). Games that
    already started, or that the feed doesn't list, are left out so a re-run
    after puck drop keeps the lines saved earlier.
    """
    events, remaining = fetch_current_odds()
    starts = fetch_start_times({g["game_id"] for g in games})
    now = datetime.datetime.now(datetime.timezone.utc)
    found = {}
    for g in games:
        start = starts.get(g["game_id"])
        start_dt = _parse_time(start)
        if start_dt and start_dt <= now:
            continue
        ev = find_event(events, g["home_team_id"], g["away_team_id"], start)
        odds = to_market_odds(ev, "publish") if ev else None
        if odds:
            found[g["game_id"]] = odds
    return found, remaining


def closing_snapshot_iso(start_time_utc):
    start = _parse_time(start_time_utc)
    snap = start - CLOSING_LEAD
    snap = snap.replace(minute=snap.minute - snap.minute % 5, second=0, microsecond=0)
    return snap.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def backfill(season, dry_run=False):
    """Closing lines for this season's predictions that have none yet."""
    start_year = season // 10000
    query = (
        supabase.table("game_predictions")
        .select("game_id,home_team_id,away_team_id,market_odds")
        .gte("game_id", start_year * 1_000_000)
        .lt("game_id", (start_year + 1) * 1_000_000)
        .is_("market_odds", "null")
    )
    rows = fetch_all("game_predictions", query)
    starts = fetch_start_times({r["game_id"] for r in rows})
    now = datetime.datetime.now(datetime.timezone.utc)
    by_snapshot = {}
    for r in rows:
        start = starts.get(r["game_id"])
        if start and _parse_time(start) < now:
            by_snapshot.setdefault(closing_snapshot_iso(start), []).append({**r, "start": start})

    games = sum(len(v) for v in by_snapshot.values())
    print(f"{games} started game(s) without lines, {len(by_snapshot)} snapshot(s) "
          f"≈ {len(by_snapshot) * 30} credits")
    if dry_run or not by_snapshot:
        return 0

    saved = missing = 0
    remaining = None
    for snapshot, group in sorted(by_snapshot.items()):
        try:
            events, remaining = fetch_historical_odds(snapshot)
        except OddsUnavailable as exc:
            print(f"  {snapshot}: {exc}")
            missing += len(group)
            continue
        for r in group:
            ev = find_event(events, r["home_team_id"], r["away_team_id"], r["start"])
            odds = to_market_odds(ev, "closing") if ev else None
            if not odds:
                missing += 1
                print(f"  {r['game_id']}: not in snapshot {snapshot}")
                continue
            supabase.table("game_predictions").update(
                {"market_odds": odds, "odds_fetched_at": snapshot}
            ).eq("game_id", r["game_id"]).execute()
            saved += 1
    print(f"Saved {saved}, missing {missing}. Credits remaining: {remaining}")
    return 1 if missing else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Odds stored with predictions.")
    parser.add_argument("--backfill", action="store_true", help="Fill closing lines for past predictions.")
    parser.add_argument("--season", type=int, default=20262027)
    parser.add_argument("--dry-run", action="store_true", help="Count snapshots and credits; no API calls.")
    args = parser.parse_args(argv)
    if not args.backfill:
        parser.error("nothing to do (pass --backfill)")
    return backfill(args.season, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
