import datetime

import numpy as np
import pandas as pd


_TEAM_PRIOR_GAMES = 20
_SPLIT_PRIOR_GAMES = 6
_SCHEDULE_WINDOW = 10
_GOALIE_PRIOR_SHOTS = 500
_GOALIE_PRIOR_STARTS = 15
_LINEUP_ACTIVE_DAYS = 10
_LINEUP_TOP_N = 6
_LINEUP_DEPTH_N = 12


def _as_date(val) -> datetime.date:
    if isinstance(val, datetime.date) and not isinstance(val, datetime.datetime):
        return val
    if isinstance(val, datetime.datetime):
        return val.date()
    if isinstance(val, str):
        return datetime.date.fromisoformat(val)
    if hasattr(val, "date"):
        return val.date()
    return val


def _empty_lineup() -> dict:
    return {
        "lineup_availability": 0.0,
        "top_skater_impact": 0.0,
        "deployment_concentration": 0.0,
    }


def _empty_goalie() -> dict:
    return {
        "goalie_talent_strength": 0.0,
        "goalie_workload": 0.0,
        "goalie_fatigue": 0.0,
        "goalie_team_adj_strength": 0.0,
    }


def _safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bayes_mean(observed, prior_mean, sample_size, prior_weight):
    observed = _safe_float(observed, prior_mean)
    prior_mean = _safe_float(prior_mean, 0.0)
    sample_size = max(_safe_float(sample_size, 0.0), 0.0)
    prior_weight = max(_safe_float(prior_weight, 0.0), 0.0)
    denom = sample_size + prior_weight
    if denom <= 0:
        return prior_mean
    return ((observed * sample_size) + (prior_mean * prior_weight)) / denom


def _previous_season(season: int) -> int | None:
    season_str = str(int(season))
    if len(season_str) != 8:
        return None
    start = int(season_str[:4]) - 1
    end = int(season_str[4:]) - 1
    return int(f"{start}{end}")


def _build_team_games(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "date",
                "team_id",
                "opp_team_id",
                "is_home",
                "goals_for",
                "goals_against",
                "won",
            ]
        )

    home = games_df[
        ["id", "season", "date", "home_team_id", "away_team_id", "home_score", "away_score"]
    ].rename(
        columns={
            "id": "game_id",
            "home_team_id": "team_id",
            "away_team_id": "opp_team_id",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    )
    home["is_home"] = True

    away = games_df[
        ["id", "season", "date", "home_team_id", "away_team_id", "home_score", "away_score"]
    ].rename(
        columns={
            "id": "game_id",
            "away_team_id": "team_id",
            "home_team_id": "opp_team_id",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    )
    away["is_home"] = False

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["date"] = pd.to_datetime(team_games["date"], errors="coerce").dt.date
    team_games["won"] = (team_games["goals_for"] > team_games["goals_against"]).astype(float)
    return team_games.sort_values(["team_id", "date", "game_id"]).reset_index(drop=True)


def _attach_season(df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    if "season" in out.columns:
        return out

    if "game_id" in out.columns and not games_df.empty:
        season_map = games_df[["id", "season"]].rename(columns={"id": "game_id"})
        out = out.merge(season_map, on="game_id", how="left")
    return out


def build_strength_state(
    games_df: pd.DataFrame,
    standings_df: pd.DataFrame,
    goalie_df: pd.DataFrame,
    gsax_df: pd.DataFrame,
    skater_df: pd.DataFrame | None = None,
) -> dict:
    team_games = _build_team_games(games_df)
    goalie_games = _attach_season(goalie_df, games_df)
    gsax_games = _attach_season(gsax_df, games_df)
    skater_games = _attach_season(skater_df if skater_df is not None else pd.DataFrame(), games_df)

    return {
        "team_games_by_team": {
            team_id: group.reset_index(drop=True)
            for team_id, group in team_games.groupby("team_id")
        },
        "goalies_by_team": {
            team_id: group.sort_values(["date", "game_id"]).reset_index(drop=True)
            for team_id, group in goalie_games.groupby("team_id")
        } if not goalie_games.empty else {},
        "goalies_by_player": {
            player_id: group.sort_values(["date", "game_id"]).reset_index(drop=True)
            for player_id, group in goalie_games.groupby("player_id")
        } if not goalie_games.empty else {},
        "gsax_by_player": {
            player_id: group.sort_values(["date", "game_id"]).reset_index(drop=True)
            for player_id, group in gsax_games.groupby("player_id")
        } if not gsax_games.empty else {},
        "skaters_by_team": {
            team_id: group.sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)
            for team_id, group in skater_games.groupby("team_id")
        } if not skater_games.empty else {},
        "standings": standings_df.copy(),
        "league_team_cache": {},
        "standings_cache": {},
        "goalie_league_cache": {},
        "team_cache": {},
        "goalie_cache": {},
        "lineup_cache": {},
    }


def _league_team_baselines(state: dict, season: int, as_of_date: datetime.date) -> dict:
    key = (season, as_of_date)
    cache = state["league_team_cache"]
    if key in cache:
        return cache[key]

    standings = state["standings"]
    if standings.empty or "as_of_date" not in standings.columns:
        cache[key] = {"gf_pg": 3.0, "ga_pg": 3.0}
        return cache[key]

    season_rows = standings[
        (standings["season_id"] == season)
        & (standings["as_of_date"] < as_of_date)
    ].copy()
    if season_rows.empty:
        cache[key] = {"gf_pg": 3.0, "ga_pg": 3.0}
        return cache[key]

    latest_date = season_rows["as_of_date"].max()
    latest_rows = season_rows[season_rows["as_of_date"] == latest_date].copy()
    gp = pd.to_numeric(latest_rows.get("games_played"), errors="coerce").replace(0, np.nan)
    gf_pg = pd.to_numeric(latest_rows.get("goal_for"), errors="coerce").div(gp).replace([np.inf, -np.inf], np.nan)
    ga_pg = pd.to_numeric(latest_rows.get("goal_against"), errors="coerce").div(gp).replace([np.inf, -np.inf], np.nan)
    cache[key] = {
        "gf_pg": float(gf_pg.dropna().mean()) if gf_pg.notna().any() else 3.0,
        "ga_pg": float(ga_pg.dropna().mean()) if ga_pg.notna().any() else 3.0,
    }
    return cache[key]


def _team_point_pctg(state: dict, team_id: int, season: int, as_of_date: datetime.date) -> float | None:
    key = (team_id, season, as_of_date)
    cache = state["standings_cache"]
    if key in cache:
        return cache[key]

    standings = state["standings"]
    if standings.empty:
        cache[key] = None
        return None

    rows = standings[
        (standings["team_id"] == team_id)
        & (standings["season_id"] == season)
        & (standings["as_of_date"] < as_of_date)
    ].sort_values("as_of_date")
    value = None if rows.empty else _safe_float(rows.iloc[-1].get("point_pctg"))
    cache[key] = value
    return value


def team_strength_as_of(
    state: dict,
    *,
    team_id: int,
    season: int,
    as_of_date: datetime.date,
    is_home: bool,
    team_std: dict,
    team_adv: dict,
) -> dict:
    key = (team_id, season, as_of_date, bool(is_home))
    cache = state["team_cache"]
    if key in cache:
        return cache[key]

    league = _league_team_baselines(state, season, as_of_date)
    games_played = max(int(_safe_float(team_std.get("games_played"), 0.0)), 0)
    gf_pg = (_safe_float(team_std.get("goal_for"), 0.0) / max(games_played, 1)) if games_played else league["gf_pg"]
    ga_pg = (_safe_float(team_std.get("goal_against"), 0.0) / max(games_played, 1)) if games_played else league["ga_pg"]
    xgf_pct = _safe_float(team_adv.get("xgf_pct"), 0.5)
    hdcf_pct_5v5 = _safe_float(team_adv.get("hdcf_pct_5v5"), _safe_float(team_adv.get("hdcf_pct"), 0.5))
    win_pct = _safe_float(team_std.get("win_pctg"), 0.5)
    recent_points_pct = min(max(_safe_float(team_std.get("l10_points"), 10.0) / 20.0, 0.0), 1.0)

    smoothed_gf_pg = _bayes_mean(gf_pg, league["gf_pg"], games_played, _TEAM_PRIOR_GAMES)
    smoothed_ga_pg = _bayes_mean(ga_pg, league["ga_pg"], games_played, _TEAM_PRIOR_GAMES)

    team_games = state["team_games_by_team"].get(team_id, pd.DataFrame())
    current_prior = team_games[
        (team_games["season"] == season)
        & (team_games["date"] < as_of_date)
    ] if not team_games.empty else pd.DataFrame()

    split_prior = current_prior[current_prior["is_home"] == bool(is_home)] if not current_prior.empty else pd.DataFrame()
    if split_prior.empty:
        split_win_pct = win_pct
        split_games = 0
    else:
        split_win_pct = float(pd.to_numeric(split_prior["won"], errors="coerce").fillna(0).mean())
        split_games = len(split_prior)

    opp_point_pctgs = []
    if not current_prior.empty:
        for opp_id in current_prior.tail(_SCHEDULE_WINDOW)["opp_team_id"].tolist():
            opp_val = _team_point_pctg(state, int(opp_id), season, as_of_date)
            if opp_val is not None:
                opp_point_pctgs.append(opp_val)

    schedule_raw = float(np.mean(opp_point_pctgs)) if opp_point_pctgs else 0.5
    schedule_strength = _bayes_mean(schedule_raw, 0.5, len(opp_point_pctgs), _SCHEDULE_WINDOW) - 0.5
    split_strength = _bayes_mean(split_win_pct, win_pct, split_games, _SPLIT_PRIOR_GAMES) - win_pct

    offense = (smoothed_gf_pg - league["gf_pg"]) + 0.75 * (xgf_pct - 0.5)
    defense = (league["ga_pg"] - smoothed_ga_pg) + 0.75 * (hdcf_pct_5v5 - 0.5)
    form_blend = (
        0.65 * (_bayes_mean(recent_points_pct, win_pct, 10, 12) - 0.5)
        + 0.2 * offense
        + 0.15 * defense
    )

    cache[key] = {
        "team_off_strength": float(offense),
        "team_def_strength": float(defense),
        "team_schedule_strength": float(schedule_strength),
        "team_split_strength": float(split_strength),
        "team_form_blend": float(form_blend),
    }
    return cache[key]


def _league_goalie_baselines(state: dict, season: int, as_of_date: datetime.date) -> dict:
    key = (season, as_of_date)
    cache = state["goalie_league_cache"]
    if key in cache:
        return cache[key]

    all_goalies = []
    for group in state["goalies_by_player"].values():
        if group.empty:
            continue
        prior = group[(group["season"] == season) & (group["date"] < as_of_date)]
        if not prior.empty:
            all_goalies.append(prior[["saves", "shots_against"]])

    if all_goalies:
        joined = pd.concat(all_goalies, ignore_index=True)
        saves = pd.to_numeric(joined["saves"], errors="coerce").fillna(0).sum()
        shots = pd.to_numeric(joined["shots_against"], errors="coerce").fillna(0).sum()
        league_sv = float(saves / shots) if shots > 0 else 0.905
    else:
        league_sv = 0.905

    gsax_vals = []
    for group in state["gsax_by_player"].values():
        if group.empty:
            continue
        prior = group[(group["season"] == season) & (group["date"] < as_of_date)]
        if not prior.empty:
            vals = pd.to_numeric(prior["gsax"], errors="coerce").dropna()
            if not vals.empty:
                gsax_vals.extend(vals.tolist())

    cache[key] = {
        "league_sv": league_sv,
        "league_gsax": float(np.mean(gsax_vals)) if gsax_vals else 0.0,
    }
    return cache[key]


def _starter_player_id(
    state: dict,
    *,
    team_id: int,
    as_of_date: datetime.date,
    game_id=None,
    is_home=None,
):
    team_rows = state["goalies_by_team"].get(team_id, pd.DataFrame())
    if team_rows.empty:
        return None

    if game_id is not None:
        current = team_rows[team_rows["game_id"] == game_id]
        if is_home is not None and "is_home" in current.columns:
            current = current[current["is_home"] == is_home]
        if not current.empty:
            return int(current.iloc[0]["player_id"])

    prior = team_rows[team_rows["date"] < as_of_date]
    if prior.empty:
        return None
    return int(prior.iloc[-1]["player_id"])


def goalie_strength_as_of(
    state: dict,
    *,
    team_id: int,
    season: int,
    as_of_date: datetime.date,
    team_def_strength: float,
    team_rest_days,
    game_id=None,
    is_home=None,
    starter_player_id=None,
) -> dict:
    key = (team_id, season, as_of_date, game_id, is_home, starter_player_id)
    cache = state["goalie_cache"]
    if key in cache:
        return cache[key]

    player_id = int(starter_player_id) if starter_player_id is not None else _starter_player_id(
        state,
        team_id=team_id,
        as_of_date=as_of_date,
        game_id=game_id,
        is_home=is_home,
    )
    if player_id is None:
        cache[key] = _empty_goalie()
        return cache[key]

    league = _league_goalie_baselines(state, season, as_of_date)
    goalie_rows = state["goalies_by_player"].get(player_id, pd.DataFrame())
    gsax_rows = state["gsax_by_player"].get(player_id, pd.DataFrame())
    prev_season = _previous_season(season)

    current_goalie = goalie_rows[
        (goalie_rows["season"] == season)
        & (goalie_rows["date"] < as_of_date)
    ] if not goalie_rows.empty else pd.DataFrame()
    prev_goalie = goalie_rows[
        goalie_rows["season"] == prev_season
    ] if (prev_season is not None and not goalie_rows.empty) else pd.DataFrame()

    current_saves = pd.to_numeric(current_goalie.get("saves"), errors="coerce").fillna(0).sum()
    current_shots = pd.to_numeric(current_goalie.get("shots_against"), errors="coerce").fillna(0).sum()
    prev_saves = pd.to_numeric(prev_goalie.get("saves"), errors="coerce").fillna(0).sum()
    prev_shots = pd.to_numeric(prev_goalie.get("shots_against"), errors="coerce").fillna(0).sum()

    current_sv = float(current_saves / current_shots) if current_shots > 0 else league["league_sv"]
    prev_sv = float(prev_saves / prev_shots) if prev_shots > 0 else league["league_sv"]
    blended_sv = (
        (current_saves + (0.5 * prev_saves) + (_GOALIE_PRIOR_SHOTS * league["league_sv"]))
        / max(current_shots + (0.5 * prev_shots) + _GOALIE_PRIOR_SHOTS, 1.0)
    )

    current_gsax = gsax_rows[
        (gsax_rows["season"] == season)
        & (gsax_rows["date"] < as_of_date)
    ] if not gsax_rows.empty else pd.DataFrame()
    prev_gsax = gsax_rows[
        gsax_rows["season"] == prev_season
    ] if (prev_season is not None and not gsax_rows.empty) else pd.DataFrame()

    current_gsax_series = current_gsax["gsax"] if "gsax" in current_gsax.columns else pd.Series(dtype=float)
    prev_gsax_series = prev_gsax["gsax"] if "gsax" in prev_gsax.columns else pd.Series(dtype=float)
    current_gsax_vals = pd.to_numeric(current_gsax_series, errors="coerce").dropna()
    prev_gsax_vals = pd.to_numeric(prev_gsax_series, errors="coerce").dropna()
    current_gsax_mean = float(current_gsax_vals.mean()) if not current_gsax_vals.empty else league["league_gsax"]
    prev_gsax_mean = float(prev_gsax_vals.mean()) if not prev_gsax_vals.empty else league["league_gsax"]
    blended_gsax = _bayes_mean(
        _bayes_mean(current_gsax_mean, prev_gsax_mean, len(current_gsax_vals), len(prev_gsax_vals) * 0.5),
        league["league_gsax"],
        len(current_gsax_vals) + (0.5 * len(prev_gsax_vals)),
        _GOALIE_PRIOR_STARTS,
    )

    starts_14 = int(
        len(
            current_goalie[
                current_goalie["date"] >= (as_of_date - datetime.timedelta(days=14))
            ]
        )
    ) if not current_goalie.empty else 0
    starts_7 = int(
        len(
            current_goalie[
                current_goalie["date"] >= (as_of_date - datetime.timedelta(days=7))
            ]
        )
    ) if not current_goalie.empty else 0

    talent = (blended_sv - league["league_sv"]) + (0.015 * blended_gsax)
    workload = starts_14 - 2.0
    fatigue = (max(starts_7 - 1, 0) * 0.5) + (0.75 if _safe_float(team_rest_days, 2.0) <= 1 else 0.0)
    team_adj = talent - (0.35 * team_def_strength)

    cache[key] = {
        "goalie_talent_strength": float(talent),
        "goalie_workload": float(workload),
        "goalie_fatigue": float(fatigue),
        "goalie_team_adj_strength": float(team_adj),
    }
    return cache[key]


def lineup_context_as_of(
    state: dict,
    *,
    team_id: int,
    season: int,
    as_of_date: datetime.date,
) -> dict:
    key = (team_id, season, as_of_date)
    cache = state["lineup_cache"]
    if key in cache:
        return cache[key]

    team_rows = state["skaters_by_team"].get(team_id, pd.DataFrame())
    if team_rows.empty:
        cache[key] = _empty_lineup()
        return cache[key]

    prior = team_rows[
        (team_rows["season"] == season)
        & (team_rows["date"] < as_of_date)
    ].sort_values(["player_id", "date", "game_id"])
    if prior.empty or "rolling_toi_sec" not in prior.columns:
        cache[key] = _empty_lineup()
        return cache[key]

    latest = prior.groupby("player_id").tail(1).copy()
    latest = latest[latest["rolling_toi_sec"].notna()]
    if latest.empty:
        cache[key] = _empty_lineup()
        return cache[key]

    latest = latest.sort_values("rolling_toi_sec", ascending=False).reset_index(drop=True)
    top_pool = latest.head(min(_LINEUP_TOP_N, len(latest)))
    depth_pool = latest.head(min(_LINEUP_DEPTH_N, len(latest)))
    active_cutoff = as_of_date - datetime.timedelta(days=_LINEUP_ACTIVE_DAYS)
    active_ids = set(latest[latest["date"] >= active_cutoff]["player_id"].tolist())
    top_ids = set(top_pool["player_id"].tolist())

    availability = (len(top_ids & active_ids) / max(len(top_pool), 1)) - 0.8

    active_top = top_pool[top_pool["player_id"].isin(active_ids)].copy()
    if active_top.empty:
        impact = 0.0
    else:
        weights = pd.to_numeric(active_top["rolling_toi_sec"], errors="coerce").fillna(0).clip(lower=1.0)
        rolling_points = pd.to_numeric(active_top.get("rolling_points"), errors="coerce").fillna(0.0)
        rolling_sog = pd.to_numeric(active_top.get("rolling_sog"), errors="coerce").fillna(0.0)
        impact_raw = np.average(rolling_points + (0.05 * rolling_sog), weights=weights)
        impact = float(impact_raw - 0.6)

    depth_toi = pd.to_numeric(depth_pool["rolling_toi_sec"], errors="coerce").fillna(0.0)
    top_toi = pd.to_numeric(top_pool["rolling_toi_sec"], errors="coerce").fillna(0.0)
    concentration_raw = float(top_toi.sum() / max(depth_toi.sum(), 1.0))
    concentration = concentration_raw - 0.55

    cache[key] = {
        "lineup_availability": float(availability),
        "top_skater_impact": float(impact),
        "deployment_concentration": float(concentration),
    }
    return cache[key]
