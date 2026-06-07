"""
Best-of-7 playoff series simulation utilities (Phase 2.1, part 2).

This module is pure math + RNG, so it is unit-testable without Supabase.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from models.goals import final_home_win_probability


_DEFAULT_PATTERN = ("A", "A", "B", "B", "A", "B", "A")


def _parse_home_ice_pattern(home_ice_pattern: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(home_ice_pattern, str):
        if home_ice_pattern != "2-2-1-1-1":
            raise ValueError(
                "String home_ice_pattern must be '2-2-1-1-1'. "
                "For custom patterns, pass a 7-item sequence of 'A'/'B'."
            )
        return _DEFAULT_PATTERN

    pattern = tuple(str(x).upper() for x in home_ice_pattern)
    if len(pattern) != 7 or any(x not in {"A", "B"} for x in pattern):
        raise ValueError("home_ice_pattern must contain 7 entries of 'A'/'B'.")
    return pattern


def _team_a_win_prob_for_game(
    p_home_game: float | tuple[float, float] | Callable[[int, str], float],
    *,
    game_number: int,
    home_team: str,
) -> float:
    if callable(p_home_game):
        p_home = float(p_home_game(game_number, home_team))
    elif isinstance(p_home_game, tuple):
        if len(p_home_game) != 2:
            raise ValueError("Tuple p_home_game must be (p_home_at_a_rink, p_home_at_b_rink).")
        p_home = float(p_home_game[0] if home_team == "A" else p_home_game[1])
    else:
        p_home = float(p_home_game)

    if not 0.0 <= p_home <= 1.0:
        raise ValueError("Per-game home-win probabilities must be in [0, 1].")
    return p_home if home_team == "A" else (1.0 - p_home)


def simulate_series(
    p_home_game: float | tuple[float, float] | Callable[[int, str], float],
    *,
    home_ice_pattern: str | Sequence[str] = "2-2-1-1-1",
    n_sims: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Simulate a best-of-7 series.

    Parameters
    ----------
    p_home_game:
        - float: home-team win probability applied in every game. Team A's per-game
          win chance is p when A is home and (1-p) when B is home.
        - tuple(p_home_at_a_rink, p_home_at_b_rink): allows asymmetric rink effects.
        - callable(game_number, home_team) -> p_home_win: full custom control where
          game_number is 1..7 and home_team is "A" or "B".
    home_ice_pattern:
        "2-2-1-1-1" (default), or an explicit 7-item sequence of "A"/"B" values
        indicating which team hosts each potential game.
    n_sims:
        Number of Monte Carlo simulations.
    rng:
        Optional numpy Generator. If None, uses numpy.random.default_rng().
    """
    if int(n_sims) <= 0:
        raise ValueError("n_sims must be positive.")
    n_sims = int(n_sims)
    rng = rng if rng is not None else np.random.default_rng()
    pattern = _parse_home_ice_pattern(home_ice_pattern)

    team_a_wins = np.zeros(n_sims, dtype=np.int8)
    team_b_wins = np.zeros(n_sims, dtype=np.int8)
    finished = np.zeros(n_sims, dtype=bool)
    games_played = np.zeros(n_sims, dtype=np.int8)

    for game_idx, home_team in enumerate(pattern, start=1):
        active = ~finished
        if not np.any(active):
            break

        p_a_win = _team_a_win_prob_for_game(
            p_home_game,
            game_number=game_idx,
            home_team=home_team,
        )
        draws = rng.random(int(active.sum()))
        a_won = draws < p_a_win

        active_idx = np.flatnonzero(active)
        team_a_wins[active_idx] += a_won.astype(np.int8)
        team_b_wins[active_idx] += (~a_won).astype(np.int8)
        games_played[active_idx] = game_idx

        finished[active_idx] = (team_a_wins[active_idx] >= 4) | (team_b_wins[active_idx] >= 4)

    length_distribution = {
        games: float(np.mean(games_played == games))
        for games in (4, 5, 6, 7)
    }
    p_team_a_wins = float(np.mean(team_a_wins >= 4))

    return {
        "p_team_a_wins": p_team_a_wins,
        "length_distribution": length_distribution,
        "expected_games": float(np.mean(games_played)),
    }


def series_win_prob_from_lambdas(
    *,
    lambdas_a_home: tuple[float, float, float],
    lambdas_b_home: tuple[float, float, float],
    home_ice_pattern: str | Sequence[str] = "2-2-1-1-1",
    n_sims: int = 10000,
    rng: np.random.Generator | None = None,
    max_goals: int = 10,
) -> dict:
    """
    Convert goals-model lambdas to series-win probability via Monte Carlo.

    lambdas_a_home is (lambda_home, lambda_away, lambda3) when Team A is home.
    lambdas_b_home is (lambda_home, lambda_away, lambda3) when Team B is home.
    """
    p_home_a_rink = final_home_win_probability(*lambdas_a_home, max_goals=max_goals)
    p_home_b_rink = final_home_win_probability(*lambdas_b_home, max_goals=max_goals)
    return simulate_series(
        (p_home_a_rink, p_home_b_rink),
        home_ice_pattern=home_ice_pattern,
        n_sims=n_sims,
        rng=rng,
    )
