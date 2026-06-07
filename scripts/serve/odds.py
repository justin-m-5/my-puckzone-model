from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

from db import fetch_all, supabase


@dataclass
class NormalizedOdds:
    game_id: int
    date: datetime.date | None
    provider: str
    opening_home_prob: float | None
    opening_away_prob: float | None
    closing_home_prob: float | None
    closing_away_prob: float | None


def american_to_implied_prob(odds: float | int | None) -> float | None:
    if odds is None:
        return None
    odds = float(odds)
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def _devig(home_prob: float | None, away_prob: float | None) -> tuple[float | None, float | None]:
    if home_prob is None or away_prob is None:
        return home_prob, away_prob
    total = home_prob + away_prob
    if total <= 0:
        return None, None
    return home_prob / total, away_prob / total


def _extract_pair(row: dict) -> tuple[float | None, float | None]:
    home_prob = row.get("home_implied_prob")
    away_prob = row.get("away_implied_prob")
    if home_prob is None or away_prob is None:
        home_prob = american_to_implied_prob(row.get("home_moneyline"))
        away_prob = american_to_implied_prob(row.get("away_moneyline"))
    return _devig(home_prob, away_prob)


def normalize_market_odds(rows: list[dict]) -> dict[int, NormalizedOdds]:
    grouped: dict[int, dict] = {}
    for row in rows:
        game_id = row.get("game_id")
        if game_id is None:
            continue
        game_id = int(game_id)
        typ = (row.get("snapshot_type") or "closing").lower()
        home_prob, away_prob = _extract_pair(row)

        slot = grouped.setdefault(
            game_id,
            {
                "game_id": game_id,
                "date": row.get("date"),
                "provider": row.get("provider") or "unknown",
                "opening_home_prob": None,
                "opening_away_prob": None,
                "closing_home_prob": None,
                "closing_away_prob": None,
            },
        )
        if typ == "opening":
            slot["opening_home_prob"] = home_prob
            slot["opening_away_prob"] = away_prob
        else:
            slot["closing_home_prob"] = home_prob
            slot["closing_away_prob"] = away_prob

    result: dict[int, NormalizedOdds] = {}
    for game_id, row in grouped.items():
        result[game_id] = NormalizedOdds(**row)
    return result


class OddsProvider:
    """Odds adapter interface."""

    def fetch_for_date(self, target_date: datetime.date) -> list[dict]:
        raise NotImplementedError


class SupabaseOddsProvider(OddsProvider):
    def __init__(self, table: str | None = None):
        self.table = table or os.getenv("ODDS_TABLE", "market_odds")

    def fetch_for_date(self, target_date: datetime.date) -> list[dict]:
        query = (
            supabase.table(self.table)
            .select(
                "game_id,date,provider,snapshot_type,home_moneyline,away_moneyline,home_implied_prob,away_implied_prob"
            )
            .eq("date", target_date.isoformat())
        )
        return fetch_all(self.table, query)
