-- Run this once in the Supabase SQL editor before writing form snapshots.

create table if not exists public.team_form_snapshots (
    team_id int not null,
    as_of_date date not null,
    season int,

    point_pctg double precision,
    win_pctg double precision,
    reg_win_pctg double precision,
    goal_diff double precision,
    l10_points double precision,

    pp_pctg double precision,
    faceoff_pctg double precision,
    sog double precision,
    hits double precision,
    blocked_shots double precision,

    cf_pct double precision,
    xgf_pct double precision,
    hdcf_pct double precision,
    cf_pct_5v5 double precision,
    xgf_pct_5v5 double precision,
    hdcf_pct_5v5 double precision,

    elo double precision,
    feature_version text,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),

    primary key (team_id, as_of_date)
);

create table if not exists public.goalie_form_snapshots (
    player_id int not null,
    as_of_date date not null,
    team_id int,
    goalie_sv_pctg double precision,
    goalie_gsax double precision,
    feature_version text,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),

    primary key (player_id, as_of_date)
);

create index if not exists idx_team_form_snapshots_as_of_date on public.team_form_snapshots (as_of_date);
create index if not exists idx_goalie_form_snapshots_as_of_date on public.goalie_form_snapshots (as_of_date);
