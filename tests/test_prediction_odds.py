# tests/test_prediction_odds.py

import datetime

from scripts.predict import odds

MTL, STL, UTA, VGK = 8, 19, 68, 54


def event(home, away, commence, ml=(-150, 130)):
    return {
        "home_team": home, "away_team": away, "commence_time": commence,
        "bookmakers": [
            {"key": "fanduel", "title": "FanDuel", "last_update": "x", "markets": []},
            {"key": "draftkings", "title": "DraftKings", "last_update": "2026-10-09T15:00:00Z", "markets": [
                {"key": "h2h", "outcomes": [{"name": away, "price": ml[1]}, {"name": home, "price": ml[0]}]},
                {"key": "spreads", "outcomes": [
                    {"name": away, "price": -180, "point": 1.5}, {"name": home, "price": 150, "point": -1.5}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": -110, "point": 6.5}, {"name": "Under", "price": -110, "point": 6.5}]},
            ]},
        ],
    }


def test_matches_accented_and_punctuated_names_and_prefers_draftkings():
    events = [event("Montréal Canadiens", "St Louis Blues", "2026-10-09T23:00:00Z")]
    ev = odds.find_event(events, MTL, STL, "2026-10-09T23:00:00+00:00")
    got = odds.to_market_odds(ev, "publish")
    assert got["bookmaker"] == "DraftKings"
    assert got["source"] == "publish"
    assert got["moneyline"] == {"home": -150, "away": 130}
    assert got["spread"] == {"homeLine": -1.5, "homePrice": 150, "awayLine": 1.5, "awayPrice": -180}
    assert got["totals"]["overLine"] == 6.5


def test_home_and_away_must_match_and_start_time_must_be_close():
    events = [event("St Louis Blues", "Montréal Canadiens", "2026-10-09T23:00:00Z")]
    assert odds.find_event(events, MTL, STL, "2026-10-09T23:00:00Z") is None
    events = [event("Montréal Canadiens", "St Louis Blues", "2026-10-11T23:00:00Z")]
    assert odds.find_event(events, MTL, STL, "2026-10-09T23:00:00Z") is None


def test_lines_for_games_skips_started_and_unlisted_games(monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc)
    later = (now + datetime.timedelta(hours=3)).isoformat()
    earlier = (now - datetime.timedelta(minutes=30)).isoformat()
    events = [event("Utah Mammoth", "Vegas Golden Knights", later), event("Montréal Canadiens", "St Louis Blues", earlier)]
    monkeypatch.setattr(odds, "fetch_current_odds", lambda: (events, "500"))
    monkeypatch.setattr(odds, "fetch_start_times", lambda ids: {1: later, 2: earlier, 3: later})
    found, remaining = odds.lines_for_games([
        {"game_id": 1, "home_team_id": UTA, "away_team_id": VGK},
        {"game_id": 2, "home_team_id": MTL, "away_team_id": STL},   # already started
        {"game_id": 3, "home_team_id": VGK, "away_team_id": UTA},   # not in the feed
    ])
    assert list(found) == [1] and remaining == "500"


def test_closing_snapshot_is_five_minutes_before_on_a_five_minute_grid():
    assert odds.closing_snapshot_iso("2026-10-09T23:07:00+00:00") == "2026-10-09T23:00:00Z"
    assert odds.closing_snapshot_iso("2026-10-10T00:00:00+00:00") == "2026-10-09T23:55:00Z"
