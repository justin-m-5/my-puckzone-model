from math import comb

import numpy as np

from models.series_sim import simulate_series, series_win_prob_from_lambdas


def _analytic_best_of_7(p: float) -> float:
    # Win in 4+k games where opponent has k wins before final game.
    return sum(comb(3 + k, k) * (p ** 4) * ((1.0 - p) ** k) for k in range(4))


def test_symmetry_at_point_five_is_near_half():
    out = simulate_series(0.5, n_sims=200000, rng=np.random.default_rng(7))
    assert abs(out["p_team_a_wins"] - 0.5) < 0.01


def test_monotonicity_higher_per_game_prob_gives_higher_series_prob():
    rng = np.random.default_rng(123)
    p50 = simulate_series((0.50, 0.50), n_sims=120000, rng=np.random.default_rng(rng.integers(1_000_000)))["p_team_a_wins"]
    p55 = simulate_series((0.55, 0.45), n_sims=120000, rng=np.random.default_rng(rng.integers(1_000_000)))["p_team_a_wins"]
    p65 = simulate_series((0.65, 0.35), n_sims=120000, rng=np.random.default_rng(rng.integers(1_000_000)))["p_team_a_wins"]
    assert p65 > p55 > p50


def test_matches_closed_form_for_constant_team_a_strength():
    p = 0.57
    sim = simulate_series((p, 1.0 - p), n_sims=250000, rng=np.random.default_rng(42))
    exact = _analytic_best_of_7(p)
    assert abs(sim["p_team_a_wins"] - exact) < 0.01


def test_length_distribution_sums_to_one_and_reproducible():
    out1 = simulate_series((0.58, 0.42), n_sims=100000, rng=np.random.default_rng(99))
    out2 = simulate_series((0.58, 0.42), n_sims=100000, rng=np.random.default_rng(99))

    assert out1 == out2
    assert abs(sum(out1["length_distribution"].values()) - 1.0) < 1e-12
    assert set(out1["length_distribution"].keys()) == {4, 5, 6, 7}


def test_series_win_prob_from_lambdas_returns_valid_probability():
    out = series_win_prob_from_lambdas(
        lambdas_a_home=(3.0, 2.4, 0.2),
        lambdas_b_home=(2.6, 2.8, 0.2),
        n_sims=40000,
        rng=np.random.default_rng(5),
    )
    assert 0.0 <= out["p_team_a_wins"] <= 1.0
