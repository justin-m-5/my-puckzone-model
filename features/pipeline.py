# features/pipeline.py
"""
PuckZone v2.0 — Unified, point-in-time-safe feature pipeline.

Single entry point used by ALL three callers:
  - features/training/builder.py          (regular-season training batch)
  - features/training/playoff_builder.py  (playoff training batch + 4 series cols)
  - scripts/predict/builder.py            (live single-matchup serving)

WHY this module exists
----------------------
v1.x had three independent feature builders that drifted apart and produced
train/serve skew:

  1. Goalie skew — training looked up the *actual* starting goalie's rolling
     sv%/GSAx for each historical game; serving used the team's *last-seen*
     starter. The model was trained on truth and served on a guess.

  2. Advanced rolling window skew — training used shift(1).rolling(window) so
     the current game was never included; serving used tail(ROLL_WINDOW).mean()
     over all available data, which could include same-day games and did not
     respect the as_of_date cutoff.

  3. Neutral fills — NEUTRAL_FILLS from models/game.py is the agreed imputation
     but was applied inconsistently.

This module fixes all three by being the *only* place where features are
computed.  Every computation is parameterized by ``as_of_date`` and uses
**only** rows with ``date < as_of_date``.

Data access is injectable via DataContext so the pipeline can be unit-tested
with small in-memory DataFrames (no live Supabase connection required).

Public API
----------
  DataContext            — injectable data container
  build_feature_row()    — point-in-time: one matchup, any as_of_date
  build_features_batch() — efficient batch: all games of a given type
"""

import datetime
import numpy as np
import pandas as pd

from features.elo import STARTING_ELO, K, HOME_ADV

# Rolling window sizes — match v1.x behaviour exactly so existing .pkl files
# continue to produce the same probability distributions.
_GOALIE_WINDOW = 10
_TEAM_STATS_WINDOW = 10
_ADVANCED_WINDOW = 10

# Share columns kept in the advanced rolling table.
_ADVANCED_COLS = [
    "cf_pct", "xgf_pct", "hdcf_pct",
    "cf_pct_5v5", "xgf_pct_5v5", "hdcf_pct_5v5",
]


# ---------------------------------------------------------------------------
# Injectable data container
# ---------------------------------------------------------------------------

class DataContext:
    """
    Holds all pre-loaded DataFrames required by the feature pipeline.

    Pass a DataContext built from tiny in-memory fixtures to unit-test the
    pipeline without a live Supabase connection (see tests/conftest.py).

    All ``date`` columns are normalised to ``datetime.date`` objects on
    construction so downstream code can rely on consistent comparisons.
    """

    def __init__(
        self,
        *,
        games: pd.DataFrame,
        standings: pd.DataFrame,
        goalie_df: pd.DataFrame,
        gsax_df: pd.DataFrame,
        team_stats_df: pd.DataFrame,
        advanced_df: pd.DataFrame,
    ):
        self.games = _coerce_date(games.copy(), "date")
        self.standings = standings.copy()
        if not self.standings.empty and "as_of_date" in self.standings.columns:
            self.standings["as_of_date"] = pd.to_datetime(
                self.standings["as_of_date"]
            ).dt.date
        self.goalie_df = _coerce_date(goalie_df.copy(), "date")
        self.gsax_df = _coerce_date(gsax_df.copy(), "date") if not gsax_df.empty else gsax_df.copy()
        self.team_stats_df = _coerce_date(team_stats_df.copy(), "date") if not team_stats_df.empty else team_stats_df.copy()
        self.advanced_df = _coerce_date(advanced_df.copy(), "date") if not advanced_df.empty else advanced_df.copy()

    @classmethod
    def from_supabase(cls) -> "DataContext":
        """Load all required data from Supabase (live connection required)."""
        from features.games import get_all_games
        from features.standings import get_standings
        from features.goalies import get_goalie_stats, get_goalie_advanced_stats
        from features.team_stats import get_team_stats
        from features.advanced import get_materialized_team_games, build_advanced_team_games

        print("Loading games (regular season + playoffs)...")
        games = get_all_games()

        print("Loading standings...")
        standings = get_standings()

        print("Loading goalie stats...")
        goalie_df = get_goalie_stats(games_df=games)
        gsax_df = get_goalie_advanced_stats()

        print("Loading team stats...")
        team_stats_df = get_team_stats(games_df=games)

        print("Loading advanced metrics...")
        advanced_df = get_materialized_team_games()
        if advanced_df.empty:
            print("  Falling back to play-by-play computation...")
            advanced_df = build_advanced_team_games(games_df=games)

        return cls(
            games=games,
            standings=standings,
            goalie_df=goalie_df,
            gsax_df=gsax_df,
            team_stats_df=team_stats_df,
            advanced_df=advanced_df,
        )


# ---------------------------------------------------------------------------
# Point-in-time helpers (used by build_feature_row)
# ---------------------------------------------------------------------------

def _coerce_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a column to datetime.date in-place, returning the DataFrame."""
    if col in df.columns and not df.empty:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def _as_date(val) -> datetime.date:
    """Normalise val to a datetime.date."""
    if isinstance(val, datetime.date) and not isinstance(val, datetime.datetime):
        return val
    if isinstance(val, datetime.datetime):
        return val.date()
    if isinstance(val, str):
        return datetime.date.fromisoformat(val)
    if hasattr(val, "date"):
        return val.date()
    return val


def _goalie_sv_as_of(
    goalie_df: pd.DataFrame,
    team_id: int,
    as_of_date: datetime.date,
    game_id=None,
    is_home=None,
    window: int = _GOALIE_WINDOW,
):
    """
    Rolling save% for a team's goalie, using only data strictly before as_of_date.

    Training mode (game_id provided): identifies the *actual* starter of game_id
    then computes their rolling sv% from the window of prior starts.

    Serving mode (game_id=None): uses the most recent starter before as_of_date
    (best available guess when the starter is not yet announced).

    In both modes the rolling window is computed identically — last ``window``
    starts before as_of_date — so the skew between training and serving is
    eliminated.
    """
    if goalie_df.empty:
        return None

    if game_id is not None:
        # Training: look up the actual starter for this game.
        row = goalie_df[
            (goalie_df["game_id"] == game_id)
            & (goalie_df["team_id"] == team_id)
            & (goalie_df["is_home"] == is_home)
        ]
        if row.empty:
            return None
        player_id = row.iloc[0]["player_id"]
    else:
        # Serving: most recent starter before as_of_date.
        recent = goalie_df[
            (goalie_df["team_id"] == team_id)
            & (goalie_df["date"] < as_of_date)
        ].sort_values("date")
        if recent.empty:
            return None
        player_id = recent.iloc[-1]["player_id"]

    # Rolling sv% = prior starts strictly before as_of_date.
    # Identical to shift(1).rolling(window) used in the batch builder.
    prior = goalie_df[
        (goalie_df["player_id"] == player_id)
        & (goalie_df["date"] < as_of_date)
    ].sort_values("date")

    if len(prior) < 3:
        return None

    tail = prior.tail(window)
    saves = tail["saves"].sum()
    shots = tail["shots_against"].sum()
    return float(saves / shots) if shots > 0 else None


def _goalie_gsax_as_of(
    gsax_df: pd.DataFrame,
    goalie_df: pd.DataFrame,
    team_id: int,
    as_of_date: datetime.date,
    game_id=None,
    is_home=None,
    window: int = _GOALIE_WINDOW,
):
    """
    Rolling GSAx for a team's goalie, strictly prior to as_of_date.
    Uses the same player-identification logic as _goalie_sv_as_of.
    """
    if gsax_df.empty or goalie_df.empty:
        return None

    if game_id is not None:
        row = goalie_df[
            (goalie_df["game_id"] == game_id)
            & (goalie_df["team_id"] == team_id)
            & (goalie_df["is_home"] == is_home)
        ]
        if row.empty:
            return None
        player_id = row.iloc[0]["player_id"]
    else:
        recent = goalie_df[
            (goalie_df["team_id"] == team_id)
            & (goalie_df["date"] < as_of_date)
        ].sort_values("date")
        if recent.empty:
            return None
        player_id = recent.iloc[-1]["player_id"]

    prior = gsax_df[
        (gsax_df["player_id"] == player_id)
        & (gsax_df["date"] < as_of_date)
    ].sort_values("date")

    if len(prior) < 3:
        return None

    tail = prior.tail(window)
    vals = pd.to_numeric(tail["gsax"], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) >= 3 else None


def _team_stats_as_of(
    team_stats_df: pd.DataFrame,
    team_id: int,
    as_of_date: datetime.date,
    window: int = _TEAM_STATS_WINDOW,
) -> dict:
    """
    Rolling average of key team stats, using data strictly before as_of_date.
    Equivalent to shift(1).rolling(window) used in the batch builder.
    """
    cols = ["pp_pctg", "faceoff_winning_pctg", "sog", "hits", "blocked_shots"]
    if team_stats_df.empty:
        return {c: None for c in cols}

    prior = team_stats_df[
        (team_stats_df["team_id"] == team_id)
        & (team_stats_df["date"] < as_of_date)
    ].sort_values("date")

    if len(prior) < 3:
        return {c: None for c in cols}

    tail = prior.tail(window)
    return {
        c: (float(tail[c].mean()) if c in tail.columns and tail[c].notna().any() else None)
        for c in cols
    }


def _advanced_as_of(
    advanced_df: pd.DataFrame,
    team_id: int,
    as_of_date: datetime.date,
    window: int = _ADVANCED_WINDOW,
) -> dict:
    """
    Rolling advanced shot-share stats, using data strictly before as_of_date.

    This is the serving-mode equivalent of the shift(1).rolling(window) logic
    in features/advanced.py::build_advanced_rolling.  Filtering to
    ``date < as_of_date`` excludes the current game (same effect as shift(1)),
    so training and serving produce the same distribution.
    """
    if advanced_df.empty:
        return {}

    cols = [c for c in _ADVANCED_COLS if c in advanced_df.columns]
    if not cols:
        return {}

    prior = advanced_df[
        (advanced_df["team_id"] == team_id)
        & (advanced_df["date"] < as_of_date)
    ].sort_values("date")

    if prior.empty:
        return {}

    tail = prior.tail(window)
    result = {}
    for c in cols:
        vals = pd.to_numeric(tail[c], errors="coerce").dropna()
        result[c] = float(vals.mean()) if len(vals) >= 3 else None
    return result


def _rest_days_as_of(
    games_df: pd.DataFrame,
    team_id: int,
    as_of_date: datetime.date,
):
    """Days of rest a team has before as_of_date (None if no prior game found)."""
    prior = games_df[
        (
            (games_df["home_team_id"] == team_id)
            | (games_df["away_team_id"] == team_id)
        )
        & (games_df["date"] < as_of_date)
    ].sort_values("date")

    if prior.empty:
        return None

    last_date = _as_date(prior.iloc[-1]["date"])
    if isinstance(as_of_date, str):
        as_of_date = datetime.date.fromisoformat(as_of_date)
    return (as_of_date - last_date).days


def _h2h_as_of(
    games_df: pd.DataFrame,
    home_id: int,
    away_id: int,
    as_of_date: datetime.date,
):
    """
    Head-to-head home-win% from the last 10 meetings before as_of_date.
    Returns None if no prior meetings exist.
    """
    prior = games_df[
        (games_df["date"] < as_of_date)
        & (
            (
                (games_df["home_team_id"] == home_id)
                & (games_df["away_team_id"] == away_id)
            )
            | (
                (games_df["home_team_id"] == away_id)
                & (games_df["away_team_id"] == home_id)
            )
        )
    ].tail(10)

    if prior.empty:
        return None

    home_wins = sum(
        1
        for _, g in prior.iterrows()
        if (
            g["home_team_id"] == home_id and g["home_score"] > g["away_score"]
        )
        or (
            g["away_team_id"] == home_id and g["away_score"] > g["home_score"]
        )
    )
    return home_wins / len(prior)


def _elo_as_of(
    games_df: pd.DataFrame,
    home_id: int,
    away_id: int,
    as_of_date: datetime.date,
    season: int,
):
    """
    Replay Elo ratings from scratch up to (but not including) as_of_date.
    Resets at the start of each season, matching build_elo_lookup in features/elo.py.
    Returns (home_elo, away_elo).
    """
    prior = games_df[games_df["date"] < as_of_date].sort_values("date")

    elo: dict = {}
    current_season = None

    for _, game in prior.iterrows():
        s = game["season"]
        if s != current_season:
            elo = {}
            current_season = s

        h_id = game["home_team_id"]
        a_id = game["away_team_id"]
        h_elo = elo.get(h_id, STARTING_ELO)
        a_elo = elo.get(a_id, STARTING_ELO)

        expected_home = 1 / (1 + 10 ** ((a_elo - h_elo - HOME_ADV) / 400))
        result = 1 if game["home_score"] > game["away_score"] else 0

        elo[h_id] = h_elo + K * (result - expected_home)
        elo[a_id] = a_elo + K * ((1 - result) - (1 - expected_home))

    return elo.get(home_id, STARTING_ELO), elo.get(away_id, STARTING_ELO)


def _standings_as_of(standings_df, team_id, season, as_of_date):
    """Most recent standings snapshot before as_of_date (delegates to existing helper)."""
    from features.standings import get_latest_standings_before
    return get_latest_standings_before(standings_df, team_id, season, as_of_date)


# ---------------------------------------------------------------------------
# Single row assembler (shared by both entry points)
# ---------------------------------------------------------------------------

def _assemble_row(
    *,
    home_std,
    away_std,
    home_sv,
    away_sv,
    home_gsax,
    away_gsax,
    home_rest,
    away_rest,
    home_ts: dict,
    away_ts: dict,
    home_adv: dict,
    away_adv: dict,
    h2h_home_win_pctg,
    home_elo: float,
    away_elo: float,
    game_id=None,
    season=None,
    game_date=None,
    home_team_id=None,
    away_team_id=None,
    home_win=None,
) -> dict:
    """
    Assemble the canonical feature dict from pre-computed components.

    This is the **single source of truth** for the feature schema: every caller
    (training batch, playoff training, live serving) must produce its feature
    row by calling this function.  Moving any computation here (or removing a
    column) will automatically take effect everywhere.
    """
    home_pk = (1 - away_ts["pp_pctg"]) if away_ts.get("pp_pctg") is not None else None
    away_pk = (1 - home_ts["pp_pctg"]) if home_ts.get("pp_pctg") is not None else None

    home_gf_pg = (home_std["goal_for"] or 0) / max(home_std["games_played"], 1)
    home_ga_pg = (home_std["goal_against"] or 0) / max(home_std["games_played"], 1)
    away_gf_pg = (away_std["goal_for"] or 0) / max(away_std["games_played"], 1)
    away_ga_pg = (away_std["goal_against"] or 0) / max(away_std["games_played"], 1)

    home_home_win_pctg = (home_std["home_wins"] or 0) / max(
        (home_std["home_wins"] or 0) + (home_std["home_losses"] or 0), 1
    )
    away_road_win_pctg = (away_std["road_wins"] or 0) / max(
        (away_std["road_wins"] or 0) + (away_std["road_losses"] or 0), 1
    )

    row: dict = {}

    # --- metadata (not model features) ---
    if game_id is not None:
        row["game_id"] = game_id
    if season is not None:
        row["season"] = season
    if game_date is not None:
        row["date"] = game_date
    if home_team_id is not None:
        row["home_team_id"] = home_team_id
    if away_team_id is not None:
        row["away_team_id"] = away_team_id
    if home_win is not None:
        row["target"] = home_win

    # --- standings features ---
    row.update({
        "home_games_played":          home_std["games_played"] or 0,
        "home_point_pctg":            home_std["point_pctg"] or 0.5,
        "home_win_pctg":              home_std["win_pctg"] or 0.5,
        "home_reg_win_pctg":          home_std["regulation_win_pctg"] or 0.5,
        "home_goal_diff":             home_std["goal_differential"] or 0,
        "home_l10_points":            home_std["l10_points"] or 0,
        "home_points":                home_std["points"] or 0,
        "home_goals_for_per_game":    home_gf_pg,
        "home_goals_against_per_game": home_ga_pg,

        "away_games_played":          away_std["games_played"] or 0,
        "away_point_pctg":            away_std["point_pctg"] or 0.5,
        "away_win_pctg":              away_std["win_pctg"] or 0.5,
        "away_reg_win_pctg":          away_std["regulation_win_pctg"] or 0.5,
        "away_goal_diff":             away_std["goal_differential"] or 0,
        "away_l10_points":            away_std["l10_points"] or 0,
        "away_points":                away_std["points"] or 0,
        "away_goals_for_per_game":    away_gf_pg,
        "away_goals_against_per_game": away_ga_pg,
    })

    # --- goalie features ---
    row.update({
        "home_goalie_sv_pctg": home_sv,
        "home_goalie_gsax":    home_gsax,
        "away_goalie_sv_pctg": away_sv,
        "away_goalie_gsax":    away_gsax,
    })

    # --- rest features ---
    row.update({
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "home_is_b2b":    1 if home_rest == 1 else 0,
        "away_is_b2b":    1 if away_rest == 1 else 0,
        "rest_advantage": (home_rest or 2) - (away_rest or 2),
    })

    # --- team rolling stats ---
    row.update({
        "home_pp_pctg":       home_ts.get("pp_pctg"),
        "home_pk_pctg":       home_pk,
        "home_faceoff_pctg":  home_ts.get("faceoff_winning_pctg"),
        "home_sog":           home_ts.get("sog"),
        "home_hits":          home_ts.get("hits"),
        "home_blocked_shots": home_ts.get("blocked_shots"),

        "away_pp_pctg":       away_ts.get("pp_pctg"),
        "away_pk_pctg":       away_pk,
        "away_faceoff_pctg":  away_ts.get("faceoff_winning_pctg"),
        "away_sog":           away_ts.get("sog"),
        "away_hits":          away_ts.get("hits"),
        "away_blocked_shots": away_ts.get("blocked_shots"),
    })

    # --- differential features ---
    row.update({
        "diff_point_pctg":   (home_std["point_pctg"] or 0.5) - (away_std["point_pctg"] or 0.5),
        "diff_goal_diff":    (home_std["goal_differential"] or 0) - (away_std["goal_differential"] or 0),
        "diff_l10_points":   (home_std["l10_points"] or 0) - (away_std["l10_points"] or 0),
        "diff_points":       (home_std["points"] or 0) - (away_std["points"] or 0),
        "diff_goalie_sv_pctg": (home_sv or 0) - (away_sv or 0),
        "diff_goalie_gsax":  (home_gsax or 0) - (away_gsax or 0),
        "diff_pp_pctg":      (home_ts.get("pp_pctg") or 0) - (away_ts.get("pp_pctg") or 0),
        "diff_pk_pctg":      (home_pk or 0) - (away_pk or 0),
        "diff_faceoff_pctg": (home_ts.get("faceoff_winning_pctg") or 0) - (away_ts.get("faceoff_winning_pctg") or 0),
        "diff_sog":          (home_ts.get("sog") or 0) - (away_ts.get("sog") or 0),
        "diff_goals_for_per_game":    home_gf_pg - away_gf_pg,
        "diff_goals_against_per_game": home_ga_pg - away_ga_pg,
    })

    # --- advanced shot-share diffs (neutral 0.5 fill → diff of 0 when missing) ---
    row.update({
        "diff_cf_pct":       (home_adv.get("cf_pct") or 0.5) - (away_adv.get("cf_pct") or 0.5),
        "diff_xgf_pct":      (home_adv.get("xgf_pct") or 0.5) - (away_adv.get("xgf_pct") or 0.5),
        "diff_hdcf_pct":     (home_adv.get("hdcf_pct") or 0.5) - (away_adv.get("hdcf_pct") or 0.5),
        "diff_cf_pct_5v5":   (home_adv.get("cf_pct_5v5") or 0.5) - (away_adv.get("cf_pct_5v5") or 0.5),
        "diff_xgf_pct_5v5":  (home_adv.get("xgf_pct_5v5") or 0.5) - (away_adv.get("xgf_pct_5v5") or 0.5),
        "diff_hdcf_pct_5v5": (home_adv.get("hdcf_pct_5v5") or 0.5) - (away_adv.get("hdcf_pct_5v5") or 0.5),
    })

    # --- home/away split win% ---
    row.update({
        "home_home_win_pctg":  home_home_win_pctg,
        "away_road_win_pctg":  away_road_win_pctg,
        "diff_home_road_pctg": home_home_win_pctg - away_road_win_pctg,
    })

    # --- H2H and Elo ---
    row.update({
        "h2h_home_win_pctg": h2h_home_win_pctg,
        "home_elo":          home_elo,
        "away_elo":          away_elo,
        "elo_diff":          home_elo - away_elo,
    })

    return row


# ---------------------------------------------------------------------------
# Public API — point-in-time (serving mode)
# ---------------------------------------------------------------------------

def build_feature_row(
    home_team_id: int,
    away_team_id: int,
    as_of_date,
    ctx: DataContext,
    game_id=None,
    season: int = None,
) -> dict:
    """
    Build a single feature row for a matchup using **only** data strictly
    before ``as_of_date``.

    Parameters
    ----------
    home_team_id : int
    away_team_id : int
    as_of_date   : datetime.date  (the game date)
    ctx          : DataContext    (injectable data; use DataContext.from_supabase()
                                   for production)
    game_id      : int | None     If provided (training mode), the actual starting
                                   goalie for that game is used for goalie features.
                                   If None (serving mode), the most recent starter
                                   before as_of_date is used.
    season       : int | None     If None, inferred from as_of_date (Oct+ = season start).

    Returns
    -------
    dict  Feature row (None for any feature that cannot be computed).
    Returns None if standings are not available for one or both teams.
    """
    as_of_date = _as_date(as_of_date)

    if season is None:
        y = as_of_date.year
        season = int(f"{y}{y + 1}") if as_of_date.month >= 10 else int(f"{y - 1}{y}")

    home_std = _standings_as_of(ctx.standings, home_team_id, season, as_of_date)
    away_std = _standings_as_of(ctx.standings, away_team_id, season, as_of_date)
    if home_std is None or away_std is None:
        return None
    if home_std["games_played"] == 0 and away_std["games_played"] == 0:
        return None

    home_sv = _goalie_sv_as_of(
        ctx.goalie_df, home_team_id, as_of_date, game_id=game_id, is_home=True
    )
    away_sv = _goalie_sv_as_of(
        ctx.goalie_df, away_team_id, as_of_date, game_id=game_id, is_home=False
    )
    home_gsax = _goalie_gsax_as_of(
        ctx.gsax_df, ctx.goalie_df, home_team_id, as_of_date, game_id=game_id, is_home=True
    )
    away_gsax = _goalie_gsax_as_of(
        ctx.gsax_df, ctx.goalie_df, away_team_id, as_of_date, game_id=game_id, is_home=False
    )

    home_rest = _rest_days_as_of(ctx.games, home_team_id, as_of_date)
    away_rest = _rest_days_as_of(ctx.games, away_team_id, as_of_date)

    home_ts = _team_stats_as_of(ctx.team_stats_df, home_team_id, as_of_date)
    away_ts = _team_stats_as_of(ctx.team_stats_df, away_team_id, as_of_date)

    home_adv = _advanced_as_of(ctx.advanced_df, home_team_id, as_of_date)
    away_adv = _advanced_as_of(ctx.advanced_df, away_team_id, as_of_date)

    home_elo, away_elo = _elo_as_of(
        ctx.games, home_team_id, away_team_id, as_of_date, season
    )

    h2h = _h2h_as_of(ctx.games, home_team_id, away_team_id, as_of_date)

    return _assemble_row(
        home_std=home_std,
        away_std=away_std,
        home_sv=home_sv,
        away_sv=away_sv,
        home_gsax=home_gsax,
        away_gsax=away_gsax,
        home_rest=home_rest,
        away_rest=away_rest,
        home_ts=home_ts,
        away_ts=away_ts,
        home_adv=home_adv,
        away_adv=away_adv,
        h2h_home_win_pctg=h2h,
        home_elo=home_elo,
        away_elo=away_elo,
        game_id=game_id,
        season=season,
        game_date=as_of_date,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )


# ---------------------------------------------------------------------------
# Public API — batch (training mode)
# ---------------------------------------------------------------------------

def _build_advanced_lookup_from_df(
    advanced_df: pd.DataFrame,
    window: int = _ADVANCED_WINDOW,
) -> dict:
    """
    Build the (game_id, team_id) -> rolling-stats lookup from a pre-loaded
    advanced DataFrame, using shift(1).rolling(window) exactly as
    features/advanced.py::build_advanced_rolling does.

    Keeping this here means the pipeline owns the rolling logic, so the batch
    and point-in-time paths share the same formula.
    """
    from features.advanced import ROLL_COLS

    if advanced_df.empty:
        return {}

    roll_cols = [c for c in ROLL_COLS if c in advanced_df.columns]
    lookup: dict = {}

    for team_id, group in advanced_df.groupby("team_id"):
        group = group.sort_values("date").reset_index(drop=True)
        for col in roll_cols:
            group[f"_r_{col}"] = (
                group[col].shift(1).rolling(window, min_periods=3).mean()
            )
        for _, row in group.iterrows():
            lookup[(row["game_id"], team_id)] = {
                col: (row[f"_r_{col}"] if pd.notna(row[f"_r_{col}"]) else None)
                for col in roll_cols
            }

    return lookup


def build_features_batch(
    ctx: DataContext,
    game_type: int = 2,
) -> pd.DataFrame:
    """
    Build feature rows for **all completed games** of the given type,
    using pre-built lookups for efficiency (O(n) not O(n²)).

    This is the correct entry point for training pipelines.  It is equivalent
    to calling ``build_feature_row`` for each game but dramatically faster
    because lookups are built once and reused.

    Parameters
    ----------
    ctx       : DataContext
    game_type : int   2 = regular season, 3 = playoffs, None = both

    Returns
    -------
    pd.DataFrame  One row per game, including metadata columns (game_id,
                  season, date, home_team_id, away_team_id, target).
    """
    from features.goalies import (
        build_goalie_rolling,
        build_gsax_rolling,
        get_goalie_sv_for_game,
        get_goalie_gsax_for_game,
    )
    from features.team_stats import build_team_stats_rolling, get_team_stats_for_game
    from features.advanced import get_advanced_for_game
    from features.elo import build_elo_lookup
    from features.games import build_rest_days_lookup, build_h2h_lookup

    games = ctx.games.copy()
    if game_type is not None and "game_type" in games.columns:
        games = games[games["game_type"] == game_type].copy()

    if games.empty:
        return pd.DataFrame()

    print(f"  Pre-building lookups for {len(games)} games...")
    rest_lookup = build_rest_days_lookup(games)

    goalie_lookup = build_goalie_rolling(ctx.goalie_df)
    gsax_lookup = build_gsax_rolling(ctx.gsax_df) if not ctx.gsax_df.empty else {}

    team_stats_lookup = build_team_stats_rolling(ctx.team_stats_df) if not ctx.team_stats_df.empty else {}

    # Build advanced lookup from the pre-loaded DataFrame (avoids re-fetching from
    # Supabase and uses the identical shift(1).rolling formula).
    advanced_lookup = _build_advanced_lookup_from_df(ctx.advanced_df)

    # Use all games (including playoffs) for Elo and H2H so the lookups have
    # the full picture; filter to the target game_type later.
    elo_lookup = build_elo_lookup(ctx.games)
    h2h_lookup = build_h2h_lookup(ctx.games)

    rows = []
    skipped = 0

    for _, game in games.iterrows():
        game_date = _as_date(game["date"])

        home_std = _standings_as_of(
            ctx.standings, game["home_team_id"], game["season"], game_date
        )
        away_std = _standings_as_of(
            ctx.standings, game["away_team_id"], game["season"], game_date
        )
        if home_std is None or away_std is None:
            skipped += 1
            continue
        if home_std["games_played"] == 0 and away_std["games_played"] == 0:
            skipped += 1
            continue

        # Use actual starter lookups for each historical game (training-mode goalie).
        home_sv = get_goalie_sv_for_game(
            ctx.goalie_df, goalie_lookup, game["id"], game["home_team_id"], True
        )
        away_sv = get_goalie_sv_for_game(
            ctx.goalie_df, goalie_lookup, game["id"], game["away_team_id"], False
        )
        home_gsax = get_goalie_gsax_for_game(
            ctx.goalie_df, gsax_lookup, game["id"], game["home_team_id"], True
        )
        away_gsax = get_goalie_gsax_for_game(
            ctx.goalie_df, gsax_lookup, game["id"], game["away_team_id"], False
        )

        home_rest = rest_lookup.get((game["id"], game["home_team_id"]))
        away_rest = rest_lookup.get((game["id"], game["away_team_id"]))

        home_ts = get_team_stats_for_game(team_stats_lookup, game["id"], game["home_team_id"])
        away_ts = get_team_stats_for_game(team_stats_lookup, game["id"], game["away_team_id"])

        home_adv = get_advanced_for_game(advanced_lookup, game["id"], game["home_team_id"])
        away_adv = get_advanced_for_game(advanced_lookup, game["id"], game["away_team_id"])

        elo = elo_lookup.get(game["id"], {})
        h2h_entry = h2h_lookup.get(game["id"], {})

        home_win = 1 if game["home_score"] > game["away_score"] else 0

        row = _assemble_row(
            home_std=home_std,
            away_std=away_std,
            home_sv=home_sv,
            away_sv=away_sv,
            home_gsax=home_gsax,
            away_gsax=away_gsax,
            home_rest=home_rest,
            away_rest=away_rest,
            home_ts=home_ts,
            away_ts=away_ts,
            home_adv=home_adv,
            away_adv=away_adv,
            h2h_home_win_pctg=h2h_entry.get("h2h_home_win_pctg"),
            home_elo=elo.get("home_elo", STARTING_ELO),
            away_elo=elo.get("away_elo", STARTING_ELO),
            game_id=game["id"],
            season=game["season"],
            game_date=game_date,
            home_team_id=game["home_team_id"],
            away_team_id=game["away_team_id"],
            home_win=home_win,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Built {len(df)} feature rows ({skipped} skipped — no prior standings)")
    return df
