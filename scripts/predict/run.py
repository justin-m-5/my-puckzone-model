# scripts/predict/run.py

"""
Run this to predict the outcome of a game.
Loads the saved model and prompts you for the matchup details.

Usage:
    python3 -m scripts.predict.run
"""

import pickle
import pandas as pd
from scripts.predict.inputs import pick_team, get_game_date, get_game_type
from scripts.predict.builder import build_prediction_row


def load_score_model(path="score_model.pkl"):
    """Load the trained score regressors from disk."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def load_model(path="win_model.pkl"):
    """Load the trained win model from disk."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"\nERROR: '{path}' not found.")
        print("Run 'PYTHONPATH=. python3 -m scripts.train.win' to train the win model.")
        exit(1)


def load_playoff_model(path="playoff_model.pkl"):
    """Load the playoff model if it exists."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def predict():
    print("=" * 50)
    print("  NHL Game Predictor — Puck Zone Model")
    print("=" * 50)

    # --- user input ---
    home_id, home_name, home_abbr = pick_team("home")
    away_id, away_name, away_abbr = pick_team("away")
    game_date = get_game_date()
    game_type_label, is_playoff = get_game_type()

    # --- load models ---
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

    model = payload["model"]
    scaler = payload["scaler"]
    feature_cols = payload["feature_cols"]
    model_name = payload["model_name"]

    # Load appropriate score model for the game type
    score_payload = load_score_model("playoff_score_model.pkl" if is_playoff else "score_model.pkl")

    # --- build features ---
    row, debug = build_prediction_row(home_id, away_id, game_date, is_playoff)

    X = pd.DataFrame([row])[feature_cols].fillna(0.5)
    if scaler:
        X = scaler.transform(X)

    # --- predict ---
    prob = model.predict_proba(X)[0]
    away_prob, home_prob = prob[0], prob[1]
    winner = home_name if home_prob > 0.5 else away_name
    winner_abbr = home_abbr if home_prob > 0.5 else away_abbr

    # score estimate: use trained score model if available, else fall back to goals/game
    if score_payload is not None:
        X_score = pd.DataFrame([row])[score_payload["feature_cols"]].fillna(0.5)
        home_gf = score_payload["home_model"].predict(X_score)[0]
        away_gf = score_payload["away_model"].predict(X_score)[0]
    else:
        home_gf = debug["home_gf_pg"]
        away_gf = debug["away_gf_pg"]

    print(f"\n{'=' * 50}")
    print(f"  {game_type_label.title()} Game Prediction")
    print(f"{'=' * 50}")
    print(f"  {home_name:<25} (home): {home_prob * 100:.1f}%")
    print(f"  {away_name:<25} (away): {away_prob * 100:.1f}%")
    print(f"  Predicted winner: {winner}")
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


if __name__ == "__main__":
    predict()