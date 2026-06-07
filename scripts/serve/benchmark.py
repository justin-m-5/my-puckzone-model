from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from scripts.serve.odds import NormalizedOdds


def _safe_log_loss(y_true, p_true) -> float | None:
    if len(y_true) == 0:
        return None
    arr = np.clip(np.asarray(p_true, dtype=float), 1e-9, 1 - 1e-9)
    return float(log_loss(np.asarray(y_true, dtype=int), arr, labels=[0, 1]))


def _safe_brier(y_true, p_true) -> float | None:
    if len(y_true) == 0:
        return None
    arr = np.clip(np.asarray(p_true, dtype=float), 0.0, 1.0)
    return float(brier_score_loss(np.asarray(y_true, dtype=int), arr))


def calibration_slices(y_true, p_true, bins: int = 5) -> list[dict]:
    if len(y_true) == 0:
        return []
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_true, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            {
                "bucket_min": float(edges[b]),
                "bucket_max": float(edges[b + 1]),
                "n": n,
                "pred_mean": float(p[mask].mean()),
                "actual_mean": float(y[mask].mean()),
            }
        )
    return out


def build_benchmark_rows(
    serving_rows: list[dict],
    odds_by_game: dict[int, NormalizedOdds],
) -> tuple[list[dict], list[dict]]:
    game_rows = []
    y_true = []
    model_probs = []
    y_true_with_close = []
    closing_probs = []

    for row in serving_rows:
        game_id = int(row["game_id"])
        odds = odds_by_game.get(game_id)

        model_prob = row.get("home_win_probability")
        opening_prob = odds.opening_home_prob if odds else None
        closing_prob = odds.closing_home_prob if odds else None

        outcome = row.get("home_win_outcome")
        brier = None
        ll = None
        if outcome is not None:
            outcome_int = int(outcome)
            brier = _safe_brier([outcome_int], [model_prob])
            ll = _safe_log_loss([outcome_int], [model_prob])
            y_true.append(outcome_int)
            model_probs.append(model_prob)
            if closing_prob is not None:
                y_true_with_close.append(outcome_int)
                closing_probs.append(closing_prob)

        clv_log_edge = None
        if closing_prob is not None and model_prob is not None and closing_prob > 0 and model_prob > 0:
            clv_log_edge = float(np.log(model_prob / closing_prob))

        game_rows.append(
            {
                "game_id": game_id,
                "date": row.get("date"),
                "prediction_date": row.get("prediction_date"),
                "feature_version": row.get("feature_version"),
                "model_version": row.get("model_version"),
                "run_id": row.get("run_id"),
                "generated_at": row.get("generated_at"),
                "model_home_win_prob": model_prob,
                "market_open_home_prob": opening_prob,
                "market_close_home_prob": closing_prob,
                "edge_vs_open": (None if opening_prob is None or model_prob is None else float(model_prob - opening_prob)),
                "edge_vs_close": (None if closing_prob is None or model_prob is None else float(model_prob - closing_prob)),
                "clv_log_edge": clv_log_edge,
                "home_win_outcome": outcome,
                "brier": brier,
                "log_loss": ll,
            }
        )

    daily_rows = []
    if serving_rows:
        df = pd.DataFrame(game_rows)
        first = serving_rows[0]
        date_val = first.get("date")
        has_matching_closing_data = bool(y_true_with_close and len(y_true_with_close) == len(closing_probs))
        daily_rows.append(
            {
                "date": date_val,
                "feature_version": first.get("feature_version"),
                "model_version": first.get("model_version"),
                "run_id": first.get("run_id"),
                "generated_at": first.get("generated_at"),
                "n_games": int(len(df)),
                "n_with_closing_odds": int(df["market_close_home_prob"].notna().sum()),
                "n_with_outcomes": int(df["home_win_outcome"].notna().sum()),
                "mean_edge_vs_close": (None if df["edge_vs_close"].dropna().empty else float(df["edge_vs_close"].dropna().mean())),
                "mean_clv_log_edge": (None if df["clv_log_edge"].dropna().empty else float(df["clv_log_edge"].dropna().mean())),
                "model_brier": _safe_brier(y_true, model_probs),
                "model_log_loss": _safe_log_loss(y_true, model_probs),
                "market_closing_brier": _safe_brier(y_true_with_close, closing_probs) if has_matching_closing_data else None,
                "market_closing_log_loss": _safe_log_loss(y_true_with_close, closing_probs) if has_matching_closing_data else None,
                "calibration_slices": calibration_slices(y_true, model_probs),
            }
        )

    return game_rows, daily_rows
