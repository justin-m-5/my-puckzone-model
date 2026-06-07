-- Run this once in the Supabase SQL editor before the first materialization pass.

create table if not exists public.model_game_features (
    game_id bigint primary key,
    season int,
    date date,
    game_type int,
    home_team_id int,
    away_team_id int,
    target int,
    feature_version text,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),

    home_point_pctg double precision,
    home_win_pctg double precision,
    home_reg_win_pctg double precision,
    home_goal_diff double precision,
    home_l10_points double precision,
    home_goalie_sv_pctg double precision,
    home_goalie_gsax double precision,
    home_rest_days double precision,
    home_is_b2b double precision,
    home_pp_pctg double precision,
    home_faceoff_pctg double precision,
    home_sog double precision,
    home_hits double precision,
    home_blocked_shots double precision,

    away_point_pctg double precision,
    away_win_pctg double precision,
    away_reg_win_pctg double precision,
    away_goal_diff double precision,
    away_l10_points double precision,
    away_goalie_sv_pctg double precision,
    away_goalie_gsax double precision,
    away_rest_days double precision,
    away_is_b2b double precision,
    away_pp_pctg double precision,
    away_faceoff_pctg double precision,
    away_sog double precision,
    away_hits double precision,
    away_blocked_shots double precision,

    diff_point_pctg double precision,
    diff_goal_diff double precision,
    diff_l10_points double precision,
    diff_points double precision,
    diff_goalie_sv_pctg double precision,
    diff_goalie_gsax double precision,
    rest_advantage double precision,
    diff_pp_pctg double precision,
    diff_faceoff_pctg double precision,
    diff_sog double precision,

    diff_cf_pct double precision,
    diff_xgf_pct double precision,
    diff_hdcf_pct double precision,
    diff_cf_pct_5v5 double precision,
    diff_xgf_pct_5v5 double precision,
    diff_hdcf_pct_5v5 double precision,

    home_home_win_pctg double precision,
    away_road_win_pctg double precision,
    diff_home_road_pctg double precision,
    h2h_home_win_pctg double precision,
    home_elo double precision,
    away_elo double precision,
    elo_diff double precision,

    series_game_number double precision,
    home_series_wins double precision,
    away_series_wins double precision,
    seed_diff double precision
);

create index if not exists idx_model_game_features_season on public.model_game_features (season);
create index if not exists idx_model_game_features_date on public.model_game_features (date);
create index if not exists idx_model_game_features_game_type on public.model_game_features (game_type);
