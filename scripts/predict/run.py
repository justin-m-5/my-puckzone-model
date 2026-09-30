# scripts/predict/run.py

"""
Run this to predict the outcome of a game.
Loads the saved model and prompts you for the matchup details.

Usage:
    python3 -m scripts.predict.run
    python3 -m scripts.predict.run --game-id 123
"""

import argparse
import datetime
import pickle
import sys

import pandas as pd
from db import supabase
from models import fill_features
from models.goals import expected_away_goals, expected_home_goals, score_matrix
from scripts.predict.inputs import (
    pick_team,
    get_optional_goalie_id,
    get_game_date,
    get_game_type,
)
from scripts.predict.builder import build_prediction_row
from scripts.predict.teams import TEAMS
from scripts.predict.writer import build_prediction_record, write_game_prediction

REGULAR_SEASON_GAME_TYPE = 2
PLAYOFF_GAME_TYPE = 3
GAME_SELECT_FIELDS = "id,date,game_type,home_team_id,away_team_id"
TEAM_BY_ID = {
    team_id: {"name": team_name, "abbr": abbr}
    for abbr, (team_id, team_name) in TEAMS.items()
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict an NHL game outcome.")
    parser.add_argument(
        "--game-id",
        type=int,
        help="Load game context from Supabase public.games.",
    )
    return parser.parse_args(argv)


def parse_game_date(value):
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value).split("T")[0])


def fetch_game_by_id(game_id, client=supabase):
    result = (
        client.table("games")
        .select(GAME_SELECT_FIELDS)
        .eq("id", game_id)
        .limit(1)
        .execute()
    )
    rows = getattr(result, "data", None) or []
    return rows[0] if rows else None


def resolve_team_display(team_id, client=supabase):
    local = TEAM_BY_ID.get(team_id)
    if local:
        return local["name"], local["abbr"]

    try:
        result = client.table("teams").select("*").eq("id", team_id).limit(1).execute()
        rows = getattr(result, "data", None) or []
        if rows:
            row = rows[0]
            name = row.get("name") or row.get("team_name") or row.get("full_name")
            abbr = row.get("abbrev") or row.get("abbreviation") or row.get("abbr") or row.get("tri_code") or row.get("code")
            if name and abbr:
                return name, abbr
    except Exception:
        pass

    return f"Team {team_id}", str(team_id)

def get_goalie_name(goalie_id, client=supabase):
    """Fetch goalie name from players table by player_id."""
    if goalie_id is None:
        return "Auto (TBD)"
    
    try:
        result = client.table("players").select("first_name, last_name").eq("id", goalie_id).limit(1).execute()
        rows = getattr(result, "data", None) or []
        if rows:
            row = rows[0]
            first_name = row.get("first_name") or ""
            last_name = row.get("last_name") or ""
            name = f"{first_name} {last_name}".strip()
            if name:
                return name
    except Exception:
        pass
    
    return f"ID: {goalie_id}"

def load_goals_model(path="goals_model.pkl"):
    """Load the goals model, the source of predicted scores. Required."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"\nERROR: '{path}' not found (predicted scores come from the goals model).")
        print("Pull it with 'PYTHONPATH=. python3 -m scripts.artifacts.sync pull', or train and promote it.")
        exit(1)


def load_model(path="win_model.pkl"):
    """Load the trained win model from disk."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"\nERROR: '{path}' not found.")
        print("Run 'PYTHONPATH=. python3 -m scripts.train.win' to train the win model.")
        print("Tip: run 'PYTHONPATH=. python3 -m scripts.validate.artifacts' to check all required artifacts.")
        exit(1)


def load_playoff_model(path="playoff_model.pkl"):
    """Load the playoff model if it exists."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def get_prediction_inputs(game_id=None):
    if game_id is None:
        home_id, home_name, home_abbr = pick_team("home")
        home_goalie_id = get_optional_goalie_id("home", home_abbr)
        away_id, away_name, away_abbr = pick_team("away")
        away_goalie_id = get_optional_goalie_id("away", away_abbr)
        game_date = get_game_date()
        game_type_label, is_playoff = get_game_type()
        return {
            "game_id": None,
            "game_date": game_date,
            "game_type": PLAYOFF_GAME_TYPE if is_playoff else REGULAR_SEASON_GAME_TYPE,
            "game_type_label": game_type_label,
            "is_playoff": is_playoff,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_name": home_name,
            "away_name": away_name,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_goalie_id": home_goalie_id,
            "away_goalie_id": away_goalie_id,
        }

    game = fetch_game_by_id(game_id)
    if game is None:
        print(f"\nERROR: No game found for game_id {game_id}.")
        return None

    game_type = int(game.get("game_type") or REGULAR_SEASON_GAME_TYPE)
    is_playoff = game_type == PLAYOFF_GAME_TYPE
    home_team_id = int(game["home_team_id"])
    away_team_id = int(game["away_team_id"])
    game_date = parse_game_date(game["date"])
    home_name, home_abbr = resolve_team_display(home_team_id)
    away_name, away_abbr = resolve_team_display(away_team_id)
    home_goalie_id = get_optional_goalie_id("home", home_abbr, retry_on_invalid=False)
    away_goalie_id = get_optional_goalie_id("away", away_abbr, retry_on_invalid=False)
    
    # Display goalie names after input
    home_goalie_name = get_goalie_name(home_goalie_id)
    away_goalie_name = get_goalie_name(away_goalie_id)
    if home_goalie_id:
        print(f"  Home goalie: {home_goalie_name}")
    if away_goalie_id:
        print(f"  Away goalie: {away_goalie_name}")

    return {
        "game_id": int(game["id"]),
        "game_date": game_date,
        "game_type": game_type,
        "game_type_label": "playoffs" if is_playoff else "regular",
        "is_playoff": is_playoff,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_name": home_name,
        "away_name": away_name,
        "home_abbr": home_abbr,
        "away_abbr": away_abbr,
        "home_goalie_id": home_goalie_id,
        "away_goalie_id": away_goalie_id,
    }


def print_prediction_summary(
    prediction_inputs,
    *,
    home_prob,
    away_prob,
    home_gf,
    away_gf,
    threshold,
    winner,
    debug,
    model_name,
):
    home_name = prediction_inputs["home_name"]
    away_name = prediction_inputs["away_name"]
    home_abbr = prediction_inputs["home_abbr"]
    away_abbr = prediction_inputs["away_abbr"]
    game_type_label = prediction_inputs["game_type_label"]

    print(f"\n{'=' * 50}")
    print(f"  {game_type_label.title()} Game Prediction")
    print(f"{'=' * 50}")
    print(f"  {home_name:<25} (home): {home_prob * 100:.1f}%")
    print(f"  {away_name:<25} (away): {away_prob * 100:.1f}%")
    print(f"  Predicted winner: {winner}")
    if threshold != 0.5:
        print(f"  (home picked when home win% > {threshold * 100:.0f}% — "
            f"threshold tuned to correct playoff home-ice bias)")
    print(f"\n  Predicted score: {home_abbr} {home_gf:.1f} - {away_gf:.1f} {away_abbr}")
    print(f"  Rounded:         {home_abbr} {round(home_gf)} - {round(away_gf)} {away_abbr}")
    print(f"{'=' * 50}")

    print(f"\n--- Key inputs used ---")
    print(f"  {home_abbr} record: {debug['home_record']}")
    print(f"  {away_abbr} record: {debug['away_record']}")
    if debug["home_sv"] is not None:
        print(f"  {home_abbr} goalie sv% (last 10): {debug['home_sv']:.4f}")
    if debug["away_sv"] is not None:
        print(f"  {away_abbr} goalie sv% (last 10): {debug['away_sv']:.4f}")
    if debug["home_rest"] is not None:
        print(f"  {home_abbr} rest days: {debug['home_rest']}")
    if debug["away_rest"] is not None:
        print(f"  {away_abbr} rest days: {debug['away_rest']}")
    if debug["h2h"] is not None:
        print(f"  H2H ({home_abbr} win%): {debug['h2h'] * 100:.1f}%")
    print(f"  {home_abbr} Elo: {debug['home_elo']:.0f}")
    print(f"  {away_abbr} Elo: {debug['away_elo']:.0f}")

    series = debug.get("series")
    if series:
        round_names = {1: "First Round", 2: "Second Round", 3: "Conference Final", 4: "Stanley Cup Final"}
        round_label = round_names.get(series["round_number"], f"Round {series['round_number']}")
        home_sw = series["team_a_wins"]
        away_sw = series["team_b_wins"]
        print(f"\n--- Series context ({round_label}) ---")
        print(f"  {series['series_title'] or series['series_abbrev'] or ''}")
        print(f"  Series score: {home_abbr} {home_sw} - {away_sw} {away_abbr}")
        if series["team_a_seed"] and series["team_b_seed"]:
            print(f"  Seeding: {home_abbr} #{series['team_a_seed']} vs {away_abbr} #{series['team_b_seed']}")
        if series["series_clinched"]:
            print(f"  ⚠ Series already decided.")

    print(f"\n  Model used: {model_name}")


def persist_prediction(prediction_inputs, result):
    if prediction_inputs["game_id"] is None:
        return

    game_date = prediction_inputs.get("game_date")
    if game_date is None:
        raise ValueError("game_date is required when persisting a game-id prediction.")

    record = build_prediction_record(
        game_id=prediction_inputs["game_id"],
        game_date=game_date,
        game_type=prediction_inputs["game_type"],
        home_team_id=prediction_inputs["home_team_id"],
        away_team_id=prediction_inputs["away_team_id"],
        home_goalie_id=prediction_inputs["home_goalie_id"],
        away_goalie_id=prediction_inputs["away_goalie_id"],
        home_win_prob=result["home_prob"],
        away_win_prob=result["away_prob"],
        pred_home_goals=result["home_gf"],
        pred_away_goals=result["away_gf"],
        predicted_winner_team_id=result["winner_team_id"],
        model_versions=result["model_versions"],
        input_snapshot={
            "game_id": prediction_inputs["game_id"],
            "game_date": game_date.isoformat(),
            "game_type": prediction_inputs["game_type"],
            "home_team_id": prediction_inputs["home_team_id"],
            "away_team_id": prediction_inputs["away_team_id"],
            "home_team_abbr": prediction_inputs["home_abbr"],
            "away_team_abbr": prediction_inputs["away_abbr"],
            "home_goalie_id": prediction_inputs["home_goalie_id"],
            "away_goalie_id": prediction_inputs["away_goalie_id"],
        },
        output_snapshot={
            "home_win_prob": result["home_prob"],
            "away_win_prob": result["away_prob"],
            "pred_home_goals": result["home_gf"],
            "pred_away_goals": result["away_gf"],
            "predicted_winner": result["winner"],
            "predicted_winner_team_id": result["winner_team_id"],
            "decision_threshold": result["threshold"],
            "model_name": result["model_name"],
            "score_model_name": result["score_model_name"],
        },
    )
    write_game_prediction(record)


def load_payloads(is_playoff):
    """Load the win model payload and matching score model for a game type."""
    if is_playoff:
        playoff_payload = load_playoff_model()
        if playoff_payload is not None:
            payload = playoff_payload
        else:
            print("\nNote: playoff_model.pkl not found — using win_model.pkl.")
            print("Run 'PYTHONPATH=. python3 -m scripts.train.playoff' to train the playoff model.\n")
            payload = load_model()
    else:
        payload = load_model()

    return payload, load_goals_model()


def expected_goals(goals_payload, row):
    """Expected (home, away) goals from the bivariate Poisson goals model."""
    X = fill_features(pd.DataFrame([row])[goals_payload["feature_cols"]])
    if goals_payload.get("scaler") is not None:
        X = goals_payload["scaler"].transform(X)
    home_rate, away_rate, lambda3 = goals_payload["model"].predict_rates(X)
    matrix = score_matrix(float(home_rate[0]), float(away_rate[0]), float(lambda3[0]), max_goals=10)
    return expected_home_goals(matrix), expected_away_goals(matrix)


def compute_prediction(prediction_inputs, payload, goals_payload, ctx=None):
    """Build features and run the models for one game. Returns (result, debug)."""
    model = payload["model"]
    scaler = payload["scaler"]
    feature_cols = payload["feature_cols"]
    model_name = payload["model_name"]
    # Tuned decision threshold (set by scripts/train/playoff.py). Defaults to 0.5
    # for any model that doesn't carry one.
    threshold = payload.get("decision_threshold", 0.5)

    # --- build features ---
    row, debug = build_prediction_row(
        prediction_inputs["home_team_id"],
        prediction_inputs["away_team_id"],
        prediction_inputs["game_date"],
        prediction_inputs["is_playoff"],
        home_goalie_id=prediction_inputs["home_goalie_id"],
        away_goalie_id=prediction_inputs["away_goalie_id"],
        ctx=ctx,
    )

    X = fill_features(pd.DataFrame([row])[feature_cols])
    if scaler:
        X = scaler.transform(X)

    # --- predict ---
    prob = model.predict_proba(X)[0]
    away_prob, home_prob = prob[0], prob[1]
    winner = prediction_inputs["home_name"] if home_prob > threshold else prediction_inputs["away_name"]
    winner_team_id = prediction_inputs["home_team_id"] if home_prob > threshold else prediction_inputs["away_team_id"]

    # Predicted score from the goals model (same feature row as the win model).
    home_gf, away_gf = expected_goals(goals_payload, row)
    goals_name = goals_payload.get("model_name", "goals_model")

    result = {
        "home_prob": home_prob,
        "away_prob": away_prob,
        "home_gf": home_gf,
        "away_gf": away_gf,
        "winner": winner,
        "winner_team_id": winner_team_id,
        "threshold": threshold,
        "model_name": model_name,
        "score_model_name": goals_name,
        "model_versions": {
            "win_model": model_name,
            "win_model_trained_through": payload.get("training", {}).get("trained_through"),
            "score_model": goals_name,
            "score_model_trained_through": goals_payload.get("training", {}).get("trained_through"),
            "decision_threshold": threshold,
        },
    }
    return result, debug


def predict(argv=None):
    args = parse_args(argv)
    print("=" * 50)
    print("  NHL Game Predictor — Puck Zone Model")
    print("=" * 50)

    # --- user input ---
    try:
        prediction_inputs = get_prediction_inputs(game_id=args.game_id)
    except Exception as exc:
        print(f"\nERROR: Unable to load game context: {exc}")
        return 1
    if prediction_inputs is None:
        return 1

    payload, goals_payload = load_payloads(prediction_inputs["is_playoff"])
    result, debug = compute_prediction(prediction_inputs, payload, goals_payload)

    print_prediction_summary(
        prediction_inputs,
        home_prob=result["home_prob"],
        away_prob=result["away_prob"],
        home_gf=result["home_gf"],
        away_gf=result["away_gf"],
        threshold=result["threshold"],
        winner=result["winner"],
        debug=debug,
        model_name=result["model_name"],
    )

    try:
        persist_prediction(prediction_inputs, result)
    except Exception as exc:
        print(f"\nWarning: failed to persist prediction to public.game_predictions: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(predict())
