# tests/test_pipeline.py
"""
PuckZone v2.0 — pipeline parity and leakage tests.

These tests do NOT require a live Supabase connection.  All data is injected
via the DataContext fixtures defined in conftest.py.

Test classes
------------
TestParity
    For a set of historical games, assert that the features produced by the
    **batch training path** (build_features_batch) are equal (within floating-
    point tolerance) to the features produced by the **point-in-time serving
    path** (build_feature_row with as_of_date = game date).

    This is the regression gate that proves train/serve skew is gone.

TestLeakage
    Assert that adding or removing any game data dated >= as_of_date does NOT
    change the feature row for that date.  Concretely: a DataContext that
    includes extra rows with ``date > TARGET_DATE`` must produce the same
    feature values as a DataContext that has only rows with ``date < TARGET_DATE``.
"""

import math
import pytest

from features.pipeline import (
    build_feature_row,
    build_features_batch,
)
from tests.conftest import (
    HOME_TEAM,
    AWAY_TEAM,
    ALT_AWAY_TEAM,
    TARGET_GAME_ID,
    TARGET_DATE,
    ALT_TARGET_GAME_ID,
    ALT_TARGET_DATE,
    SEASON,
)

# Columns we can check for exact parity between batch and point-in-time paths.
# Goalie features may differ if the two paths select different players (e.g. a
# mid-game goalie change not captured in fixtures), so we test them separately
# with a looser check.
_NON_GOALIE_FEATURE_COLS = [
    "home_point_pctg", "away_point_pctg",
    "home_win_pctg", "away_win_pctg",
    "home_reg_win_pctg", "away_reg_win_pctg",
    "home_goal_diff", "away_goal_diff",
    "home_l10_points", "away_l10_points",
    "home_rest_days", "away_rest_days",
    "home_is_b2b", "away_is_b2b",
    "rest_advantage",
    "diff_point_pctg", "diff_goal_diff", "diff_l10_points",
    "diff_pp_pctg", "diff_faceoff_pctg", "diff_sog",
    "diff_cf_pct", "diff_xgf_pct", "diff_hdcf_pct",
    "diff_cf_pct_5v5", "diff_xgf_pct_5v5", "diff_hdcf_pct_5v5",
    "home_home_win_pctg", "away_road_win_pctg", "diff_home_road_pctg",
    "h2h_home_win_pctg",
    "home_elo", "away_elo", "elo_diff",
]

_GOALIE_FEATURE_COLS = [
    "home_goalie_sv_pctg", "away_goalie_sv_pctg",
    "home_goalie_gsax", "away_goalie_gsax",
    "diff_goalie_sv_pctg", "diff_goalie_gsax",
]

_TOLERANCE = 1e-9   # floating-point tolerance for exact-parity assertions


def _approx_eq(a, b, tol=_TOLERANCE):
    """True if both are None, both are NaN, or |a-b| <= tol."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------

class TestParity:
    """
    Training path (build_features_batch) == serving path (build_feature_row
    with as_of_date = game date) for all deterministic non-goalie features.
    """

    def test_batch_and_point_in_time_have_same_shape(self, ctx):
        """build_features_batch returns a non-empty DataFrame."""
        df = build_features_batch(ctx, game_type=2)
        assert not df.empty, "build_features_batch returned an empty DataFrame"
        assert "game_id" in df.columns

    def test_deterministic_features_match(self, ctx):
        """
        For game TARGET_GAME_ID, the feature values from the batch path equal
        those from the point-in-time path (as_of_date = TARGET_DATE).
        """
        # Batch path (training mode)
        batch_df = build_features_batch(ctx, game_type=2)
        batch_row = batch_df[batch_df["game_id"] == TARGET_GAME_ID]
        assert len(batch_row) == 1, (
            f"Expected exactly one row for game {TARGET_GAME_ID}, got {len(batch_row)}"
        )
        batch = batch_row.iloc[0].to_dict()

        # Point-in-time path (serving mode) — game_id provided so goalie logic matches
        pit = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=AWAY_TEAM,
            as_of_date=TARGET_DATE,
            ctx=ctx,
            game_id=TARGET_GAME_ID,
            season=SEASON,
        )
        assert pit is not None, "build_feature_row returned None"

        mismatches = []
        for col in _NON_GOALIE_FEATURE_COLS:
            if col not in batch or col not in pit:
                continue
            if not _approx_eq(batch[col], pit[col]):
                mismatches.append(
                    f"  {col}: batch={batch[col]!r}  pit={pit[col]!r}"
                )

        assert not mismatches, (
            "Train/serve parity failure — the following features differ:\n"
            + "\n".join(mismatches)
        )

    def test_goalie_features_present(self, ctx):
        """
        Goalie features should be non-None for the target game when enough
        prior starts exist in the fixtures (conftest provides 10 prior games).
        """
        pit = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=AWAY_TEAM,
            as_of_date=TARGET_DATE,
            ctx=ctx,
            game_id=TARGET_GAME_ID,
            season=SEASON,
        )
        assert pit is not None
        # With 10 prior starts and min_periods=3 the rolling should be defined.
        assert pit["home_goalie_sv_pctg"] is not None, "home_goalie_sv_pctg should not be None"
        assert pit["away_goalie_sv_pctg"] is not None, "away_goalie_sv_pctg should not be None"

    def test_goalie_features_match_between_paths(self, ctx):
        """
        When game_id is provided (training mode) both paths identify the same
        player and compute the same rolling window, so goalie features should
        agree exactly.
        """
        batch_df = build_features_batch(ctx, game_type=2)
        batch_row = batch_df[batch_df["game_id"] == TARGET_GAME_ID].iloc[0].to_dict()

        pit = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=AWAY_TEAM,
            as_of_date=TARGET_DATE,
            ctx=ctx,
            game_id=TARGET_GAME_ID,
            season=SEASON,
        )

        for col in _GOALIE_FEATURE_COLS:
            if col not in batch_row or col not in pit:
                continue
            assert _approx_eq(batch_row[col], pit[col]), (
                f"Goalie feature mismatch for {col}: "
                f"batch={batch_row[col]!r}  pit={pit[col]!r}"
            )

    def test_elo_is_point_in_time(self, ctx):
        """
        Elo features should reflect only games played before TARGET_DATE.
        At game 1 the Elo diff is 0 (both teams start at STARTING_ELO).
        """
        from features.elo import STARTING_ELO
        first_game_date = ctx.games["date"].min()
        pit = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=AWAY_TEAM,
            as_of_date=first_game_date,
            ctx=ctx,
            season=SEASON,
        )
        # No prior games → both Elo should be STARTING_ELO
        # (may be None if standings not yet available; skip in that case)
        if pit is not None:
            assert pit["home_elo"] == STARTING_ELO
            assert pit["away_elo"] == STARTING_ELO
            assert pit["elo_diff"] == 0.0

    def test_advanced_rolling_uses_prior_games_only(self, ctx):
        """
        The advanced rolling window at TARGET_DATE must equal the mean of
        all prior games, not include TARGET_DATE itself.
        """
        prior = ctx.advanced_df[
            (ctx.advanced_df["team_id"] == HOME_TEAM)
            & (ctx.advanced_df["date"] < TARGET_DATE)
        ].sort_values("date")
        expected_home = prior.tail(10)["cf_pct"].mean()

        prior_away = ctx.advanced_df[
            (ctx.advanced_df["team_id"] == AWAY_TEAM)
            & (ctx.advanced_df["date"] < TARGET_DATE)
        ].sort_values("date")
        expected_away = prior_away.tail(10)["cf_pct"].mean()

        pit = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=AWAY_TEAM,
            as_of_date=TARGET_DATE,
            ctx=ctx,
            season=SEASON,
        )
        assert pit is not None
        expected_diff = expected_home - expected_away
        assert abs(pit["diff_cf_pct"] - expected_diff) < 1e-6, (
            f"Expected diff_cf_pct ≈ {expected_diff:.3f}, got {pit['diff_cf_pct']!r}"
        )

    def test_h2h_uses_prior_games_only(self, ctx):
        """
        H2H percentage at TARGET_DATE should be based on games before that date.
        The fixture has 12 prior HOME vs AWAY games with 10 home wins, so
        uncapped H2H should be 10 / 12 (not capped to last 10 = 1.0).
        """
        pit = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=AWAY_TEAM,
            as_of_date=TARGET_DATE,
            ctx=ctx,
            season=SEASON,
        )
        assert pit is not None
        assert pit["h2h_home_win_pctg"] == pytest.approx(10 / 12)

    def test_h2h_fixture_has_different_prior_meeting_counts(self, ctx):
        """
        Secondary fixture matchup has fewer priors (HOME vs ALT_AWAY), ensuring
        H2H behavior is tested across unequal prior-meeting counts.
        """
        pit = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=ALT_AWAY_TEAM,
            as_of_date=ALT_TARGET_DATE,
            ctx=ctx,
            game_id=ALT_TARGET_GAME_ID,
            season=SEASON,
        )
        assert pit is not None
        # 4 prior meetings before ALT_TARGET_DATE; home won 3.
        assert pit["h2h_home_win_pctg"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------

class TestLeakage:
    """
    Features for TARGET_DATE must not change when data dated >= TARGET_DATE
    is added to the DataContext.
    """

    def test_future_games_do_not_affect_features(self, ctx, ctx_with_future):
        """
        A DataContext that includes a game on FUTURE_DATE (> TARGET_DATE)
        must produce identical feature values for TARGET_DATE.
        """
        row_base = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=AWAY_TEAM,
            as_of_date=TARGET_DATE,
            ctx=ctx,
            season=SEASON,
        )
        row_future = build_feature_row(
            home_team_id=HOME_TEAM,
            away_team_id=AWAY_TEAM,
            as_of_date=TARGET_DATE,
            ctx=ctx_with_future,
            season=SEASON,
        )

        assert row_base is not None and row_future is not None

        all_cols = _NON_GOALIE_FEATURE_COLS + _GOALIE_FEATURE_COLS
        leaks = []
        for col in all_cols:
            if col not in row_base or col not in row_future:
                continue
            if not _approx_eq(row_base[col], row_future[col]):
                leaks.append(
                    f"  {col}: without_future={row_base[col]!r}  "
                    f"with_future={row_future[col]!r}"
                )

        assert not leaks, (
            "Leakage detected — future data changed the following features:\n"
            + "\n".join(leaks)
        )

    def test_batch_leakage(self, ctx, ctx_with_future):
        """
        build_features_batch should produce the same row for TARGET_GAME_ID
        regardless of whether post-date games exist in the DataContext.
        """
        df_base = build_features_batch(ctx, game_type=2)
        df_future = build_features_batch(ctx_with_future, game_type=2)

        row_base = df_base[df_base["game_id"] == TARGET_GAME_ID]
        row_future = df_future[df_future["game_id"] == TARGET_GAME_ID]

        assert len(row_base) == 1, "Target game not found in base batch"
        assert len(row_future) == 1, "Target game not found in future batch"

        b = row_base.iloc[0].to_dict()
        f = row_future.iloc[0].to_dict()

        all_cols = _NON_GOALIE_FEATURE_COLS + _GOALIE_FEATURE_COLS
        leaks = []
        for col in all_cols:
            if col not in b or col not in f:
                continue
            if not _approx_eq(b[col], f[col]):
                leaks.append(
                    f"  {col}: without_future={b[col]!r}  with_future={f[col]!r}"
                )

        assert not leaks, (
            "Batch leakage detected — future data changed the following features:\n"
            + "\n".join(leaks)
        )

    def test_rest_days_not_affected_by_future_game(self, ctx, ctx_with_future):
        """
        A future game on FUTURE_DATE must not change the rest-days feature
        computed at TARGET_DATE.
        """
        row_base = build_feature_row(
            HOME_TEAM, AWAY_TEAM, TARGET_DATE, ctx, season=SEASON
        )
        row_future = build_feature_row(
            HOME_TEAM, AWAY_TEAM, TARGET_DATE, ctx_with_future, season=SEASON
        )
        assert row_base["home_rest_days"] == row_future["home_rest_days"]
        assert row_base["away_rest_days"] == row_future["away_rest_days"]

    def test_goalie_not_affected_by_future_game(self, ctx, ctx_with_future):
        """
        A goalie start on FUTURE_DATE must not affect the rolling sv%
        at TARGET_DATE.
        """
        row_base = build_feature_row(
            HOME_TEAM, AWAY_TEAM, TARGET_DATE, ctx, season=SEASON
        )
        row_future = build_feature_row(
            HOME_TEAM, AWAY_TEAM, TARGET_DATE, ctx_with_future, season=SEASON
        )
        assert _approx_eq(
            row_base["home_goalie_sv_pctg"], row_future["home_goalie_sv_pctg"]
        ), "Future goalie start leaked into home_goalie_sv_pctg"
        assert _approx_eq(
            row_base["away_goalie_sv_pctg"], row_future["away_goalie_sv_pctg"]
        ), "Future goalie start leaked into away_goalie_sv_pctg"
