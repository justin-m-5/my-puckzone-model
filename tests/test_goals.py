import numpy as np
from math import factorial

from features.pipeline import build_features_batch
from models import FEATURE_COLS, fill_features
from models.goals import (
    BivariatePoissonGoalsModel,
    final_home_win_probability,
    p_away_win,
    p_home_win,
    p_regulation_tie,
    score_matrix,
)


def _truncated_poisson(lmbda: float, max_goals: int = 10) -> np.ndarray:
    vals = np.array(
        [np.exp(-lmbda) * (lmbda ** k) / factorial(k) for k in range(max_goals + 1)],
        dtype=float,
    )
    return vals / vals.sum()


def test_score_matrix_nonnegative_and_normalized():
    mat = score_matrix(3.1, 2.7, 0.35, max_goals=10)
    assert mat.shape == (11, 11)
    assert np.all(mat >= 0)
    assert np.isclose(mat.sum(), 1.0, atol=1e-6)


def test_win_tie_probabilities_sum_to_one_and_final_home_prob_valid():
    mat = score_matrix(2.9, 2.4, 0.2, max_goals=10)
    total = p_home_win(mat) + p_away_win(mat) + p_regulation_tie(mat)
    assert np.isclose(total, 1.0, atol=1e-6)

    final_home = final_home_win_probability(2.9, 2.4, 0.2, max_goals=10)
    assert 0.0 <= final_home <= 1.0


def test_independent_case_matches_outer_product():
    lam_home, lam_away, max_goals = 2.6, 2.2, 10
    mat = score_matrix(lam_home, lam_away, lambda3=0.0, max_goals=max_goals)

    home = _truncated_poisson(lam_home, max_goals=max_goals)
    away = _truncated_poisson(lam_away, max_goals=max_goals)
    expected = np.outer(home, away)

    assert np.allclose(mat, expected, atol=1e-10)


def test_tiny_end_to_end_fit_returns_finite_rates_and_valid_matrix(ctx):
    df = build_features_batch(ctx, game_type=2)
    scores = ctx.games[["id", "home_score", "away_score"]].rename(columns={"id": "game_id"})
    df = df.merge(scores, on="game_id", how="left").dropna(subset=["home_score", "away_score"])

    X = fill_features(df[FEATURE_COLS])
    model = BivariatePoissonGoalsModel(use_shared_lambda3=True)
    model.fit(X, df["home_score"].values, df["away_score"].values)

    lam_home, lam_away, lam3 = model.predict_rates(X.head(1))
    assert np.isfinite(lam_home).all()
    assert np.isfinite(lam_away).all()
    assert np.isfinite(lam3).all()
    assert (lam_home > 0).all()
    assert (lam_away > 0).all()
    assert (lam3 >= 0).all()

    mat = score_matrix(lam_home[0], lam_away[0], lam3[0], max_goals=10)
    assert np.all(mat >= 0)
    assert np.isclose(mat.sum(), 1.0, atol=1e-6)
