# features/playoffs.py

from typing import Optional
from db import supabase


def get_series_context(team_a_id: int, team_b_id: int, bracket_year: int) -> Optional[dict]:
    """
    Fetch the current playoff series between two teams for the given bracket year.
    Returns a dict with series context, or None if no series found.
    """
    result = (
        supabase.table("playoff_series")
        .select(
            "round_number, series_title, series_abbrev, "
            "top_seed_team_id, top_seed_abbrev, top_seed_wins, top_seed_rank, "
            "bottom_seed_team_id, bottom_seed_abbrev, bottom_seed_wins, bottom_seed_rank, "
            "winning_team_id"
        )
        .eq("bracket_year", bracket_year)
        .or_(
            f"and(top_seed_team_id.eq.{team_a_id},bottom_seed_team_id.eq.{team_b_id}),"
            f"and(top_seed_team_id.eq.{team_b_id},bottom_seed_team_id.eq.{team_a_id})"
        )
        .maybe_single()
        .execute()
    )

    if not result or not result.data:
        return None

    s = result.data

    # Normalise so home_team perspective is consistent with the caller
    def wins_for(team_id):
        if s["top_seed_team_id"] == team_id:
            return s["top_seed_wins"] or 0
        if s["bottom_seed_team_id"] == team_id:
            return s["bottom_seed_wins"] or 0
        return 0

    def seed_for(team_id):
        if s["top_seed_team_id"] == team_id:
            return s["top_seed_rank"]
        if s["bottom_seed_team_id"] == team_id:
            return s["bottom_seed_rank"]
        return None

    return {
        "round_number": s["round_number"],
        "series_title": s["series_title"],
        "series_abbrev": s["series_abbrev"],
        "team_a_wins": wins_for(team_a_id),
        "team_b_wins": wins_for(team_b_id),
        "team_a_seed": seed_for(team_a_id),
        "team_b_seed": seed_for(team_b_id),
        "series_clinched": s["winning_team_id"] is not None,
        "series_winner_id": s["winning_team_id"],
    }
