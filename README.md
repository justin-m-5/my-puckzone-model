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
  strength.py                Talent-aware team / goalie / lineup strength helpers
  plays.py                   Shot events + xG feature builder
  playoffs.py                Playoff series context (wins, seeding) from Supabase
  training/
    builder.py               Regular-season training dataset (thin wrapper over pipeline)
    playoff_builder.py       Playoff training dataset (thin wrapper + 4 series features)

models/
  game.py                    Win model definition + FEATURE_COLS (87 features)
  goals.py                   Bivariate Poisson goals model (Phase 2.1)
  playoff.py                 Playoff win model + PLAYOFF_FEATURE_COLS (91 features)
  player.py                  Player point prediction model
  xg.py                      Expected goals model

scripts/
  materialize/
    run.py                   Materialize model feature store + form snapshots
  serve/
    run.py                   Phase 2.3 daily orchestrator (predictions + market benchmarking)
    writer.py                App-serving + benchmark table writers (idempotent upserts)
    odds.py                  Odds provider adapter + implied probability normalization
    benchmark.py             Per-game + daily benchmark metrics vs market
  predict/
    run.py                   Interactive game predictor (main entry point)
    builder.py               Assembles live features via unified pipeline for one game
    inputs.py                CLI prompts (team, optional goalie id, date, game type)
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
- Talent-aware team / goalie / lineup strength features are derived from the same
  point-in-time inputs and use neutral defaults when optional skater history is unavailable.
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
    skater_df=my_skater_df,   # optional; lineup proxies fall back to neutral without it
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

## Current daily ops (recommended)

Issue #6 roadmap status: **Phases 2.0–2.3 completed (June 7, 2026)**.

Use this as the canonical daily sequence:

```bash
# 1) Refresh materialized features (safe preview first)
PYTHONPATH=. python3 -m scripts.materialize.run --dry-run
PYTHONPATH=. python3 -m scripts.materialize.run

# 2) Retrain core v2 models
PYTHONPATH=. python3 -m scripts.train.xg
PYTHONPATH=. python3 -m scripts.train.win
PYTHONPATH=. python3 -m scripts.train.goals

# 3) Generate predictions (recommended orchestrator path)
PYTHONPATH=. python3 -m scripts.serve.run --date 2026-06-07 --dry-run
PYTHONPATH=. python3 -m scripts.serve.run --date 2026-06-07

# Fallback interactive predictor (still supported)
PYTHONPATH=. python3 -m scripts.predict.run
```

Quick runbook:
1. Pull latest `main`
2. Refresh env/secrets
3. Materialize `--dry-run`, then real write
4. Retrain (`xg`, `win`, `goals`)
5. Run `scripts.serve.run` (or `scripts.predict.run` fallback)
6. Spot-check output table row counts and latest date

## Run a prediction

```bash
PYTHONPATH=. python3 -m scripts.predict.run
```

You will be prompted for:
1. Home team abbreviation (e.g. `CAR`)
2. Home team starting goalie id (optional, press Enter for auto-selection)
3. Away team abbreviation (e.g. `VGK`)
4. Away team starting goalie id (optional, press Enter for auto-selection)
5. Game date (defaults to today)
6. Game type — `1` Regular Season or `2` Playoffs

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

## Phase 2.3 daily serving + market benchmark pipeline

`scripts.serve.run` orchestrates a deterministic daily cascade:
1. Refresh/ingest inputs (games + optional odds snapshot)
2. Refresh feature layer hook (`--force-recompute` for full refresh workflows)
3. Generate goals-model predictions leak-safe as-of date
4. Upsert app-facing serving rows
5. Compute and upsert market benchmark rows (per-game + daily aggregate)

### App-serving table contract (`model_game_predictions`)

Core fields written per game/date:
- identifiers: `game_id`, `date`, `season`, `home_team_id`, `away_team_id`
- version metadata: `feature_version`, `model_version`, `run_id`, `generated_at`, `prediction_date`
- predictive outputs: `home_win_probability`, `expected_home_goals`, `expected_away_goals`, `most_likely_home_score`, `most_likely_away_score`
- optional uncertainty fields: `regulation_tie_probability`, `lambda_home`, `lambda_away`, `lambda3`
- app safety/status fields: `is_finalized`, `data_source`

Writes are idempotent via upsert key semantics:
`(game_id, prediction_date, feature_version, model_version)`.

### Benchmark tables

- `model_market_benchmarks` (per game): market open/close probs, model edge, CLV-style log edge, optional realized Brier/log loss when outcomes are final.
- `model_market_benchmark_daily` (daily aggregate): counts, mean edges, model/market daily Brier + log loss, calibration slices.

### Commands

```bash
# Daily run (today by default)
PYTHONPATH=. python3 -m scripts.serve.run

# Explicit date + dry-run (no writes)
PYTHONPATH=. python3 -m scripts.serve.run --date 2026-01-15 --dry-run

# Historical replay/backfill
PYTHONPATH=. python3 -m scripts.serve.run --start-date 2025-10-01 --end-date 2025-10-31

# Skip odds ingestion/benchmark
PYTHONPATH=. python3 -m scripts.serve.run --date 2026-01-15 --skip-odds
```

Odds provider config:
- `ODDS_TABLE` (optional, default `market_odds`)
- write paths still require `SUPABASE_SERVICE_ROLE_KEY` for upserts

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

---

## Talent-aware strength features (Phase 2.2 PR 2/2)

Phase 2.2 PR 2/2 completes Phase 2.2 by extending the unified pipeline with
additive talent-aware strength signals:

- **Team strength** — Bayesian-smoothed offense/defense ratings, opponent-strength
  schedule context, home/away split strength with shrinkage, and a recent-form
  blend anchored to the longer-season baseline.
- **Goalie strength** — current + prior-season blended save ability / GSAx proxy,
  recent workload + fatigue context, and a team-defense-adjusted goalie signal.
- **Lineup context** — recent-usage availability proxy for top skaters, top-skater
  impact proxy, and a deployment concentration signal (top-heavy vs balanced).

All of these features are computed strictly with `date < as_of_date`, are shared
by training and serving through `features/pipeline.py`, and are materialized as
additive nullable columns so older readers can still fall back safely.

### Materialization / migration notes

- `model_game_features` now includes the new strength columns in addition to the
  existing feature set.
- `team_form_snapshots` now carries team-strength + lineup snapshot fields.
- `goalie_form_snapshots` now carries goalie strength context fields.
- Schema migrations still belong in the ingest / SQL repo, not this model repo.
  Add the new columns there as nullable columns; readers here already reindex /
  fill missing optional columns safely during rollout.

### Training / backtest commands

Commands are unchanged; the new features flow through the existing builders:

```bash
PYTHONPATH=. python3 -m scripts.train.win
PYTHONPATH=. python3 -m scripts.train.goals
PYTHONPATH=. python3 -m scripts.backtest.run
PYTHONPATH=. python3 -m pytest tests/ -v
```
