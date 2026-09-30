"""
scripts/artifacts/report.py — compare the live models with the candidates.

Scores the live win and goals models on the candidate's holdout season (flagged
as in-sample when the live model already trained on it) next to the candidate's
own holdout metrics, and shows what each model trained on.

Usage:
    PYTHONPATH=. python3 -m scripts.artifacts.report
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

from models import fill_features

CANDIDATE_DIR = "candidates"
MODELS = ("win_model.pkl", "goals_model.pkl")


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _transform(payload, rows):
    X = fill_features(rows[payload["feature_cols"]])
    return payload["scaler"].transform(X) if payload.get("scaler") is not None else X


def _win_metrics(payload, rows):
    from scripts.backtest.metrics import compute_metrics
    home_prob = payload["model"].predict_proba(_transform(payload, rows))[:, 1]
    return compute_metrics(home_prob, rows["target"])


def _goals_metrics(payload, rows):
    home_rate, away_rate, _ = payload["model"].predict_rates(_transform(payload, rows))
    actual_total = rows["home_score"] + rows["away_score"]
    return {
        "n_games": int(len(rows)),
        "total_goals_mae": float(np.mean(np.abs((home_rate + away_rate) - actual_total))),
        "home_goals_mae": float(np.mean(np.abs(home_rate - rows["home_score"]))),
        "away_goals_mae": float(np.mean(np.abs(away_rate - rows["away_score"]))),
    }


def _holdout_rows(season: int) -> pd.DataFrame:
    from db import fetch_all, supabase
    from features.training import build_features

    df = build_features(use_materialized=True)
    df = df[df["season"] == season]
    query = (
        supabase.table("games").select("id, home_score, away_score")
        .eq("season", season).eq("game_type", 2).in_("game_state", ["OFF", "FINAL"])
    )
    scores = pd.DataFrame(fetch_all("games", query)).rename(columns={"id": "game_id"})
    return df.merge(scores, on="game_id", how="left").dropna(subset=["home_score", "away_score"])


def _describe(payload) -> str:
    t = (payload or {}).get("training")
    if not t:
        return "no training metadata (trained before evaluate-then-refit)"
    seasons = t.get("seasons") or []
    span = f"{seasons[0]}–{seasons[-1]}" if seasons else "?"
    return f"seasons {span}, through {t.get('trained_through')}, {t.get('n_rows')} rows, trained {t.get('trained_at')}"


def _row(label, live, cand, tolerance, lower_is_better=True, fmt="{:.4f}"):
    """tolerance: differences smaller than this are noise at holdout sample size."""
    if live is None or cand is None:
        return f"  {label:<18} live {fmt.format(live) if live is not None else '—':>8}   candidate {fmt.format(cand) if cand is not None else '—':>8}"
    diff = cand - live
    if abs(diff) < tolerance:
        mark = "same (within noise)"
    else:
        mark = "better" if (diff < 0) == lower_is_better else "worse"
    return f"  {label:<18} live {fmt.format(live):>8}   candidate {fmt.format(cand):>8}   ({mark})"


def main() -> int:
    live = {name: _load(name) for name in MODELS}
    cand = {name: _load(os.path.join(CANDIDATE_DIR, name)) for name in MODELS}
    if not any(cand.values()):
        print(f"No candidates in {CANDIDATE_DIR}/. Train first: scripts.train.win / scripts.train.goals")
        return 1

    holdout = next((c["training"]["holdout_season"] for c in cand.values() if c and c.get("training")), None)
    print(f"Holdout season: {holdout}")
    rows = _holdout_rows(holdout) if holdout else pd.DataFrame()
    print(f"Holdout games with final scores: {len(rows)}\n")

    for name in MODELS:
        lv, cd = live[name], cand[name]
        print(f"=== {name} ===")
        print(f"  live:      {_describe(lv)}")
        print(f"  candidate: {_describe(cd)}")
        if cd is None:
            print("  (no candidate)\n")
            continue

        in_sample = bool(lv and holdout in ((lv.get("training") or {}).get("seasons") or []))
        cand_m = cd["training"]["holdout_metrics"]
        live_m = None
        if lv is not None and len(rows):
            live_m = _win_metrics(lv, rows) if name == "win_model.pkl" else _goals_metrics(lv, rows)
        if in_sample:
            print(f"  ⚠ live model already trained on {holdout}; its numbers below are in-sample (optimistic).")

        if name == "win_model.pkl":
            # ~1 standard error of accuracy on a full season (~1,300 games) is 0.014.
            print(_row("accuracy", live_m and live_m["accuracy"], cand_m["accuracy"], 0.014, lower_is_better=False, fmt="{:.3f}"))
            print(_row("log loss", live_m and live_m["log_loss"], cand_m["log_loss"], 0.002))
            print(_row("brier", live_m and live_m["brier"], cand_m["brier"], 0.001))
        else:
            print(_row("total goals MAE", live_m and live_m["total_goals_mae"], cand_m["total_goals_mae"], 0.02, fmt="{:.3f}"))
            print(_row("home goals MAE", live_m and live_m["home_goals_mae"], cand_m["home_goals_mae"], 0.02, fmt="{:.3f}"))
            print(_row("away goals MAE", live_m and live_m["away_goals_mae"], cand_m["away_goals_mae"], 0.02, fmt="{:.3f}"))
        print()

    print("Candidate numbers come from a fit that excluded the holdout season; the")
    print("candidate file itself is refit on all seasons. Promote with:")
    print("  PYTHONPATH=. python3 -m scripts.artifacts.promote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
