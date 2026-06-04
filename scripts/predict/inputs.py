# scripts/predict/inputs.py

import datetime
from scripts.predict.teams import TEAMS


def pick_team(label):
    """Show a team list and let the user pick by abbreviation."""
    print(f"\nAvailable teams:")
    for abbr, (tid, name) in sorted(TEAMS.items()):
        print(f"  {abbr:<5} {name}")
    while True:
        abbr = input(f"\nEnter {label} team abbreviation (e.g. CAR): ").strip().upper()
        if abbr in TEAMS:
            team_id, team_name = TEAMS[abbr]
            print(f"  Selected: {team_name}")
            return team_id, team_name, abbr
        print(f"  '{abbr}' not found. Try again.")


def get_game_date():
    """Ask for a game date, defaulting to today."""
    today = datetime.date.today()
    raw = input(f"\nGame date (YYYY-MM-DD) [default: {today}]: ").strip()
    if not raw:
        return today
    try:
        return datetime.date.fromisoformat(raw)
    except ValueError:
        print("  Invalid date, using today.")
        return today


def get_game_type():
    """Ask whether this is a playoff or regular season game."""
    print("\nGame type:")
    print("  1 = Regular Season")
    print("  2 = Playoffs")
    while True:
        choice = input("Enter 1 or 2 [default: 1]: ").strip()
        if choice == "" or choice == "1":
            return "regular", False
        if choice == "2":
            return "playoffs", True
        print("  Please enter 1 or 2.")