"""
Bivariate Poisson goals model utilities (Phase 2.1, part 1).

This module uses a pragmatic estimator:
  1) fit two marginal Poisson regressions for expected home/away goals using
     the same leak-safe feature set as the win model;
  2) estimate a shared non-negative covariance term lambda3 from residual
     covariance on the training set.

The resulting (lambda_home, lambda_away, lambda3) parameterization is then
mapped to a joint score distribution P(home=i, away=j). Setting lambda3=0
reduces to an independent-Poisson fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import numpy as np
from sklearn.linear_model import PoissonRegressor


EPS = 1e-9
OT_TIE_RULE = "rate_proportional"


def _poisson_pmf_series(lmbda: float, max_goals: int) -> np.ndarray:
    lmbda = max(float(lmbda), EPS)
    vals = np.array(
        [exp(-lmbda) * (lmbda ** k) / factorial(k) for k in range(max_goals + 1)],
        dtype=float,
    )
    s = vals.sum()
    if s <= 0:
        vals[:] = 0.0
        vals[0] = 1.0
        return vals
    return vals / s


def _bounded_lambda3(lambda_home: float, lambda_away: float, lambda3: float) -> float:
    upper = max(min(float(lambda_home), float(lambda_away)) - EPS, 0.0)
    return float(min(max(float(lambda3), 0.0), upper))


def score_matrix(lambda_home: float, lambda_away: float, lambda3: float, max_goals: int = 10) -> np.ndarray:
    """Return normalized P(home=i, away=j) for i,j in [0, max_goals]."""
    lambda_home = max(float(lambda_home), EPS)
    lambda_away = max(float(lambda_away), EPS)
    lambda3 = _bounded_lambda3(lambda_home, lambda_away, lambda3)

    if lambda3 <= EPS:
        home = _poisson_pmf_series(lambda_home, max_goals)
        away = _poisson_pmf_series(lambda_away, max_goals)
        return np.outer(home, away)

    lambda1 = max(lambda_home - lambda3, EPS)
    lambda2 = max(lambda_away - lambda3, EPS)
    mat = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    expo = exp(-(lambda1 + lambda2 + lambda3))

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            kmax = min(i, j)
            mass = 0.0
            for k in range(kmax + 1):
                mass += (
                    (lambda1 ** (i - k)) / factorial(i - k)
                    * (lambda2 ** (j - k)) / factorial(j - k)
                    * (lambda3 ** k) / factorial(k)
                )
            mat[i, j] = expo * mass

    total = mat.sum()
    if total <= 0:
        mat[:, :] = 0.0
        mat[0, 0] = 1.0
        return mat
    return mat / total


def p_home_win(score_probs: np.ndarray) -> float:
    return float(np.tril(score_probs, k=-1).sum())


def p_away_win(score_probs: np.ndarray) -> float:
    return float(np.triu(score_probs, k=1).sum())


def p_regulation_tie(score_probs: np.ndarray) -> float:
    return float(np.trace(score_probs))


def tie_break_home_share(lambda_home: float, lambda_away: float) -> float:
    """Named NHL tie-resolution rule for OT/SO split."""
    if OT_TIE_RULE == "rate_proportional":
        denom = float(lambda_home) + float(lambda_away)
        if denom <= 0:
            return 0.5
        return float(lambda_home) / denom
    return 0.5


def final_home_win_probability(lambda_home: float, lambda_away: float, lambda3: float, max_goals: int = 10) -> float:
    probs = score_matrix(lambda_home, lambda_away, lambda3, max_goals=max_goals)
    tie_mass = p_regulation_tie(probs)
    final_home = p_home_win(probs) + tie_mass * tie_break_home_share(lambda_home, lambda_away)
    return float(np.clip(final_home, 0.0, 1.0))


def expected_home_goals(score_probs: np.ndarray) -> float:
    goals = np.arange(score_probs.shape[0], dtype=float)
    return float((score_probs * goals[:, None]).sum())


def expected_away_goals(score_probs: np.ndarray) -> float:
    goals = np.arange(score_probs.shape[1], dtype=float)
    return float((score_probs * goals[None, :]).sum())


def most_likely_scoreline(score_probs: np.ndarray) -> tuple[int, int]:
    idx = np.unravel_index(int(np.argmax(score_probs)), score_probs.shape)
    return int(idx[0]), int(idx[1])


@dataclass
class BivariatePoissonGoalsModel:
    use_shared_lambda3: bool = True
    fixed_lambda3: float | None = None
    alpha: float = 1e-4
    max_iter: int = 1000

    def fit(self, X, y_home, y_away) -> "BivariatePoissonGoalsModel":
        self.home_model_ = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.away_model_ = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.home_model_.fit(X, y_home)
        self.away_model_.fit(X, y_away)

        mu_home = np.clip(self.home_model_.predict(X), EPS, None)
        mu_away = np.clip(self.away_model_.predict(X), EPS, None)

        if self.fixed_lambda3 is not None:
            self.lambda3_ = max(float(self.fixed_lambda3), 0.0)
        elif self.use_shared_lambda3:
            cov = float(np.mean((np.asarray(y_home) - mu_home) * (np.asarray(y_away) - mu_away)))
            self.lambda3_ = max(cov, 0.0)
        else:
            self.lambda3_ = 0.0
        return self

    def predict_rates(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu_home = np.clip(self.home_model_.predict(X), EPS, None)
        mu_away = np.clip(self.away_model_.predict(X), EPS, None)
        lam3 = np.full(len(mu_home), max(float(getattr(self, "lambda3_", 0.0)), 0.0), dtype=float)
        return mu_home, mu_away, lam3

    def predict_home_win_prob(self, X, max_goals: int = 10) -> np.ndarray:
        mu_home, mu_away, lam3 = self.predict_rates(X)
        return np.array(
            [
                final_home_win_probability(h, a, c, max_goals=max_goals)
                for h, a, c in zip(mu_home, mu_away, lam3)
            ],
            dtype=float,
        )

    def predict_score_matrices(self, X, max_goals: int = 10) -> list[np.ndarray]:
        mu_home, mu_away, lam3 = self.predict_rates(X)
        return [score_matrix(h, a, c, max_goals=max_goals) for h, a, c in zip(mu_home, mu_away, lam3)]
