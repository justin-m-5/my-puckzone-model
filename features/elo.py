# features/elo.py

K = 5
HOME_ADV = 50
STARTING_ELO = 1500


def build_elo_lookup(games_df):
    """
    Compute pre-game Elo ratings for every game.
    Processes games chronologically, updating ratings after each game.
    Resets to STARTING_ELO at the start of each season.
    Returns dict keyed by game_id.
    """
    games_df = games_df.sort_values("date").reset_index(drop=True)

    elo = {}        # team_id -> current elo
    lookup = {}     # game_id -> {home_elo, away_elo, elo_diff}

    current_season = None

    for _, game in games_df.iterrows():
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]
        season = game["season"]

        # reset at start of each new season
        if season != current_season:
            elo = {}
            current_season = season

        home_elo = elo.get(home_id, STARTING_ELO)
        away_elo = elo.get(away_id, STARTING_ELO)

        # store PRE-game elo as the feature
        lookup[game["id"]] = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
        }

        # expected win probability for home team
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - HOME_ADV) / 400))

        # result: 1 = home win, 0 = away win
        result = 1 if game["home_score"] > game["away_score"] else 0

        # update ratings
        elo[home_id] = home_elo + K * (result - expected_home)
        elo[away_id] = away_elo + K * ((1 - result) - (1 - expected_home))

    return lookup