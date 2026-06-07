import datetime
import pandas as pd

from db import supabase, fetch_all
from models.game import FEATURE_COLS
from models.playoff import PLAYOFF_EXTRA_COLS


_NUMERIC_COLS = FEATURE_COLS + PLAYOFF_EXTRA_COLS + [
    "game_id",
    "season",
    "game_type",
    "home_team_id",
    "away_team_id",
    "target",
]


def parse_materialized_rows(rows: list[dict]) -> pd.DataFrame:
    """Parse raw materialized rows into a typed DataFrame."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_materialized_game_features(game_type=None) -> pd.DataFrame:
    """
    Read model_game_features and return parsed rows.
    Returns empty DataFrame on any error so callers can fall back to live features.
    """
    try:
        query = supabase.table("model_game_features").select("*")
        if game_type is not None:
            query = query.eq("game_type", game_type)
        rows = fetch_all("model_game_features", query)
        return parse_materialized_rows(rows)
    except Exception as exc:
        print(f"  (model_game_features not readable: {exc})")
        return pd.DataFrame()


def get_latest_team_form(team_id: int, as_of_date) -> dict | None:
    """Latest team_form_snapshots row where as_of_date <= target date."""
    if isinstance(as_of_date, (datetime.datetime, pd.Timestamp)):
        as_of_date = as_of_date.date()
    if isinstance(as_of_date, datetime.date):
        as_of_date = as_of_date.isoformat()

    try:
        query = (
            supabase.table("team_form_snapshots")
            .select("*")
            .eq("team_id", team_id)
            .lte("as_of_date", as_of_date)
            .order("as_of_date", desc=True)
            .limit(1)
        )
        rows = fetch_all("team_form_snapshots", query)
    except Exception as exc:
        print(f"  (team_form_snapshots not readable: {exc})")
        return None
    return rows[0] if rows else None


def get_latest_goalie_form(player_id: int, as_of_date) -> dict | None:
    """Latest goalie_form_snapshots row where as_of_date <= target date."""
    if isinstance(as_of_date, (datetime.datetime, pd.Timestamp)):
        as_of_date = as_of_date.date()
    if isinstance(as_of_date, datetime.date):
        as_of_date = as_of_date.isoformat()

    try:
        query = (
            supabase.table("goalie_form_snapshots")
            .select("*")
            .eq("player_id", player_id)
            .lte("as_of_date", as_of_date)
            .order("as_of_date", desc=True)
            .limit(1)
        )
        rows = fetch_all("goalie_form_snapshots", query)
    except Exception as exc:
        print(f"  (goalie_form_snapshots not readable: {exc})")
        return None
    return rows[0] if rows else None
