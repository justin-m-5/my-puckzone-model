<!-- README.md -->

# my-puckzone-model

NHL analytics and prediction model built on historical game data spanning the 2017-18 through 2025-26 seasons.

Data is sourced from the NHL API and stored in Supabase via [my-puckzone-ingest](https://github.com/justin-m-5/my-puckzone-ingest).

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your SUPABASE_URL and SUPABASE_KEY
```

---

## Project structure

```
db.py                        Supabase client + fetch helper

features/
  games.py                   Load completed games (reg season + playoffs)
  standings.py               Daily team standings snapshots
  goalies.py                 Goalie rolling save % (last 10 starts)
  team_stats.py              Team rolling efficiency stats (PP%, faceoff%, SOG...)
  elo.py                     Elo ratings updated game-by-game
  players.py                 Per-player rolling stats for point prediction
  plays.py                   Shot events + xG feature builder
  playoffs.py                Playoff series context (wins, seeding) from Supabase
  training/
    builder.py               Build regular season training dataset (41 features)
    playoff_builder.py       Build playoff training dataset (41 + 4 series features)

models/
  game.py                    Win model definition + FEATURE_COLS (41 features)
  playoff.py                 Playoff win model + PLAYOFF_FEATURE_COLS (45 features)
  player.py                  Player point prediction model
  xg.py                      Expected goals model

scripts/
  predict/
    run.py                   Interactive game predictor (main entry point)
    builder.py               Assembles live features from Supabase for one game
    inputs.py                CLI prompts (team, date, game type)
    teams.py                 Team abbreviation → ID lookup
  train/
    win.py                   Train win_model.pkl (regular season)
    playoff.py               Train playoff_model.pkl
    scores.py                Train score_model.pkl (regular season)
    playoff_scores.py        Train playoff_score_model.pkl
    player.py                Train player_model.pkl
    xg.py                    Train xg_model.pkl
    compare.py               Compare win model variants side by side
```

---

## Saved models

| File | Predicts | Trained on |
|---|---|---|
| `win_model.pkl` | Home/away win probability | Regular season games 2017-25 |
| `playoff_model.pkl` | Home/away win probability | Playoff games 2018-25 + 4 series features |
| `score_model.pkl` | Home score + away score | Regular season games 2017-25 |
| `playoff_score_model.pkl` | Home score + away score | Playoff games 2018-25 |
| `player_model.pkl` | Will a player score a point? | Player-game rows 2017-25 |
| `xg_model.pkl` | Will a shot result in a goal? | All shot events 2017-25 |

---

## Run a prediction

```bash
PYTHONPATH=. python3 -m scripts.predict.run
```

You will be prompted for:
1. Home team abbreviation (e.g. `CAR`)
2. Away team abbreviation (e.g. `VGK`)
3. Game date (defaults to today)
4. Game type — `1` Regular Season or `2` Playoffs

When **Playoffs** is selected the predictor automatically uses `playoff_model.pkl` for win probability and `playoff_score_model.pkl` for score prediction, both trained on playoff data only. Series wins, game number, and seeding are pulled live from Supabase and used as additional model features.

---

## Train models

Run any of these from the repo root. Pre-trained `.pkl` files are already included — only retrain when you want to incorporate new seasons of data or add new features.

```bash
# Required prep before win/playoff/score retraining:
# 1) train xG
PYTHONPATH=. python3 -m scripts.train.xg

# Win probability
PYTHONPATH=. python3 -m scripts.train.win
PYTHONPATH=. python3 -m scripts.train.playoff

# Score prediction
PYTHONPATH=. python3 -m scripts.train.scores
PYTHONPATH=. python3 -m scripts.train.playoff_scores

# Player point prediction
PYTHONPATH=. python3 -m scripts.train.player

# Compare win model variants before committing
PYTHONPATH=. python3 -m scripts.train.compare
```

Advanced per-game metrics (Corsi / Fenwick / xG / high-danger, including 5v5
splits) are populated into Supabase `game_advanced_stats` by
[my-puckzone-ingest](https://github.com/justin-m-5/my-puckzone-ingest).
Feature builders in this repo read that table directly. Keep
`scripts.train.xg` up to date here because ingest uses its copy of
`xg_model.pkl` to produce `xGF` / `xGA` values during ingest.
