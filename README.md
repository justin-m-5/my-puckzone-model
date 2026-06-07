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
cp .env.example .env   # add SUPABASE_URL/SUPABASE_KEY; add SUPABASE_SERVICE_ROLE_KEY for materialize writes
```

---

## Project structure

```
conftest.py                  Root pytest conftest (mocks db for unit tests)
db.py                        Supabase client + fetch helper

features/
  pipeline.py                ⭐ Unified, point-in-time-safe feature pipeline (v2.0)
                               - DataContext  injectable data container (no live DB needed for tests)
                               - build_feature_row()   single matchup, any as_of_date
                               - build_features_batch()  efficient batch for training
  materialized.py            Materialized feature/snapshot readers + parsers
  games.py                   Load completed games (reg season + playoffs)
  standings.py               Daily team standings snapshots
  goalies.py                 Goalie rolling save % (last 10 starts)
  team_stats.py              Team rolling efficiency stats (PP%, faceoff%, SOG...)
  elo.py                     Elo ratings updated game-by-game
  advanced.py                Corsi / xG / high-danger shares (materialized table + fallback)
  players.py                 Per-player rolling stats for point prediction
  plays.py                   Shot events + xG feature builder
  playoffs.py                Playoff series context (wins, seeding) from Supabase
  training/
    builder.py               Regular-season training dataset (thin wrapper over pipeline)
    playoff_builder.py       Playoff training dataset (thin wrapper + 4 series features)

models/
  game.py                    Win model definition + FEATURE_COLS (51 features)
  goals.py                   Bivariate Poisson goals model (Phase 2.1)
  playoff.py                 Playoff win model + PLAYOFF_FEATURE_COLS (55 features)
  player.py                  Player point prediction model
  xg.py                      Expected goals model

scripts/
  materialize/
    run.py                   Materialize model feature store + form snapshots
  predict/
    run.py                   Interactive game predictor (main entry point)
    builder.py               Assembles live features via unified pipeline for one game
    inputs.py                CLI prompts (team, date, game type)
    teams.py                 Team abbreviation → ID lookup
  train/
    win.py                   Train win_model.pkl (regular season)
    playoff.py               Train playoff_model.pkl
    goals.py                 Train goals_model.pkl (bivariate Poisson)
    scores.py                Train score_model.pkl (regular season)
    playoff_scores.py        Train playoff_score_model.pkl
    player.py                Train player_model.pkl
    xg.py                    Train xg_model.pkl
    compare.py               Compare win model variants side by side
  backtest/
    run.py                   ⭐ Walk-forward backtest harness (v2.0)
    metrics.py               Shared evaluation metrics (log loss, Brier, AUC, calibration)

tests/
  conftest.py                In-memory DataContext fixtures (no Supabase required)
  test_pipeline.py           Parity tests + leakage tests for the unified pipeline
  test_materialized.py       Materialized store parity + fallback tests
```

---

## Unified feature pipeline (v2.0)

`features/pipeline.py` is the single entry point for all feature computation.
All three callers — regular-season training, playoff training, and live serving —
delegate to the same code, eliminating train/serve skew.

```
features/pipeline.py
  └─ build_features_batch(ctx, game_type=2)   ← training (batch, efficient)
  └─ build_feature_row(home, away, date, ctx) ← serving (single game, point-in-time)
       │
       ├─ features/training/builder.py          (wraps build_features_batch)
       ├─ features/training/playoff_builder.py  (wraps build_features_batch + playoff cols)
       └─ scripts/predict/builder.py            (wraps build_feature_row)
```

**Key guarantees**:
- All features use `date < as_of_date` (strictly prior data only — no leakage).
- Goalie rolling sv%/GSAx uses `shift(1).rolling(window)` in training and
  equivalent `tail(window)` on `date < as_of_date`-filtered data in serving.
- Advanced shot-share rolling uses the same shift(1)-equivalent logic in both paths.
- Neutral fills (`NEUTRAL_FILLS` from `models/game.py`) applied identically everywhere.

**Injectable DataContext** — pass an in-memory `DataContext` for unit testing:

```python
from features.pipeline import DataContext, build_feature_row

ctx = DataContext(
    games=my_games_df,
    standings=my_standings_df,
    goalie_df=my_goalie_df,
    gsax_df=my_gsax_df,
    team_stats_df=my_team_stats_df,
    advanced_df=my_advanced_df,
)
row = build_feature_row(home_id, away_id, game_date, ctx)
```

---

## Saved models

| File | Predicts | Trained on |
|---|---|---|
| `win_model.pkl` | Home/away win probability | Regular season games 2017-25 |
| `playoff_model.pkl` | Home/away win probability | Playoff games 2018-25 + 4 series features |
| `score_model.pkl` | Home score + away score | Regular season games 2017-25 |
| `goals_model.pkl` | Full score distribution + derived win prob | Regular season games 2017-25 |
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

---

## Run the walk-forward backtest (v2.0)

Evaluates the win model with a rolling-origin backtest (each season as its own
test fold, trained on all prior seasons) instead of the single-season holdout.
Reports accuracy, log loss, Brier score, and ROC AUC per fold and in aggregate.

```bash
PYTHONPATH=. python3 -m scripts.backtest.run

# Options
PYTHONPATH=. python3 -m scripts.backtest.run --model gradient_boosting
PYTHONPATH=. python3 -m scripts.backtest.run --model goals
PYTHONPATH=. python3 -m scripts.backtest.run --model logistic --no-calibrate
PYTHONPATH=. python3 -m scripts.backtest.run --exclude-seasons 20202021
PYTHONPATH=. python3 -m scripts.backtest.run --min-train-seasons 3
```

---

## Materialized feature store (Phase 2.0)

`model_game_features` and form snapshot tables are managed in
[my-puckzone-db-migration](https://github.com/justin-m-5/my-puckzone-db-migration)
under `migrations/` and applied with its `apply.sh`.
Run that migration once before first materialization.

Materialize leak-safe features and snapshots:

```bash
# build + inspect without writing (no write creds needed)
PYTHONPATH=. python3 -m scripts.materialize.run --dry-run

# optional filters
PYTHONPATH=. python3 -m scripts.materialize.run --game-type 2 --limit 500
PYTHONPATH=. python3 -m scripts.materialize.run --game-type 3
```

Running `scripts.materialize.run` with writes requires
`SUPABASE_SERVICE_ROLE_KEY` in `.env` (service role bypasses RLS on write paths).
Read-only flows continue to work with `SUPABASE_KEY`.

Training builders (`features.training.build_features` / `build_playoff_features`)
now prefer `model_game_features` when rows are present and automatically fall
back to live pipeline computation when the table is empty/unavailable.

---

## Run tests

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```

Tests run without a live Supabase connection (all data is injected via
in-memory `DataContext` fixtures).  The test suite includes:

- **Parity tests** — verify training path == serving path for all deterministic
  features, proving train/serve skew is eliminated.
- **Leakage tests** — verify that features at `as_of_date=D` do not change when
  data dated `>= D` is added to the DataContext.

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
PYTHONPATH=. python3 -m scripts.train.goals
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
Per-game goalie advanced metrics (including GSAx) are populated into
`game_goalie_advanced_stats` and consumed by the feature builders here.
Feature builders in this repo read those tables directly. Keep
`scripts.train.xg` up to date here because ingest uses its copy of
`xg_model.pkl` to produce `xGF` / `xGA` values during ingest.

---

## Goals-based model (Phase 2.1)

Phase 2.1 introduces an additive goals-based probabilistic path (part 1 of 2):
- `models/goals.py` fits two Poisson regressions for home/away goal rates and a shared non-negative covariance term (`lambda3`) for a bivariate Poisson score distribution.
- `scripts/train/goals.py` trains and saves `goals_model.pkl` using the same leak-safe feature set (`FEATURE_COLS`) via `features.training.build_features(use_materialized=True)`.
- `scripts/backtest/run.py --model goals` runs walk-forward folds and compares derived home-win probability from the goals model vs the win model, while also reporting score-distribution metrics (RPS and exact-scoreline hit rate).

Commands:

```bash
PYTHONPATH=. python3 -m scripts.train.goals
PYTHONPATH=. python3 -m scripts.backtest.run --model goals
```

### Playoff series simulation (Phase 2.1)

Phase 2.1 part 2 adds a best-of-7 Monte Carlo path that reuses the part-1
goals model:
- `models/series_sim.py` simulates playoff series with 2-2-1-1-1 home ice.
- `scripts/backtest/series.py` evaluates historical playoff series at the
  series level (calibration/accuracy), using leak-safe features as of series start.

```bash
PYTHONPATH=. python3 -m scripts.backtest.series
PYTHONPATH=. python3 -m scripts.backtest.series --n-sims 50000 --seed 42
```
