#!/usr/bin/env python3
"""
Script to fetch NBA game listings via nba_api and predict game outcomes using the trained regression model.

This script pulls upcoming game events between now and the earliest recorded game in data/game_odds.csv.
It updates past game results with actual outcomes, predicts outcomes for new games,
and appends the predictions to the log (data/game_odds.csv).
"""
import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import joblib
from zoneinfo import ZoneInfo

try:
    from nba_api.stats.static import teams as nba_teams
except ImportError:
    print("ERROR: nba_api is not installed; please install nba_api.")
    sys.exit(1)

# Load environment variables from .env if present
env_path = os.path.join(os.path.dirname(__file__), '../.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            os.environ.setdefault(key, val)


def update_correct(df):
    """
    Update the 'correct' column of the DataFrame based on actual game results.

    Fetches completed game results from the NBA Stats API and compares
    against predicted probabilities to set the 'correct' flag.
    """
    try:
        # lazy import of scoreboard endpoint
        from nba_api.stats.endpoints import ScoreboardV2
    except ImportError:
        print("ERROR: nba_api is not installed; cannot update correctness.")
        return df
    if 'correct' not in df.columns:
        df['correct'] = np.nan
    now_utc = datetime.now(timezone.utc)
    for idx, row in df.iterrows():
        if pd.notna(row.get('correct')):
            continue
        ct = row.get('commence_time')
        if not isinstance(ct, str):
            continue
        ts = ct.replace('Z', '+00:00')
        try:
            game_dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        if game_dt < now_utc:
            # convert game datetime (UTC) to US/Eastern to match NBA API's game_date
            eastern = ZoneInfo("US/Eastern")
            game_dt_eastern = game_dt.astimezone(eastern)
            date_str = game_dt_eastern.strftime('%m/%d/%Y')
            try:
                sb = ScoreboardV2(game_date=date_str)
                dfs = sb.get_data_frames()
                if len(dfs) > 1:
                    ls = dfs[1]
                    home_ab = row.get('home_abbr')
                    away_ab = row.get('away_abbr')
                    hm = ls['TEAM_ABBREVIATION'] == home_ab
                    am = ls['TEAM_ABBREVIATION'] == away_ab
                    if hm.any() and am.any():
                        home_pts = int(ls.loc[hm, 'PTS'].values[0])
                        away_pts = int(ls.loc[am, 'PTS'].values[0])
                        winner = 'home' if home_pts > away_pts else 'away'
                        predicted = 'home' if float(row.get('home_prob', 0)) > float(row.get('away_prob', 0)) else 'away'
                        df.at[idx, 'correct'] = (winner == predicted)
            except Exception as e:
                print(f"WARNING: Failed to fetch result for {ct}: {e}")
    return df


def main():
    now = datetime.now(timezone.utc)
    # path to stored odds log
    odds_path = os.path.join('data', 'game_odds.csv')
    # select games from the last 7 days through the next 7 days
    start_dt = now - timedelta(days=7)
    end_dt = now + timedelta(days=7)
    print(f"Selecting games from {start_dt.isoformat()} to {end_dt.isoformat()}")

    # build list of calendar dates to fetch
    dates = []
    d = start_dt.date()
    while d <= end_dt.date():
        dates.append(d)
        d += timedelta(days=1)

    # prepare team mappings
    teams = nba_teams.get_teams()
    id_to_abbr = {t['id']: t['abbreviation'] for t in teams}
    id_to_full = {t['id']: t['full_name'] for t in teams}

    # fetch game schedule via nba_api ScoreboardV2 over the selected dates
    upcoming = []
    for d in dates:
        date_str = d.strftime('%m/%d/%Y')
        try:
            # lazy import to avoid top-level dependency
            from nba_api.stats.endpoints import ScoreboardV2
            sb = ScoreboardV2(game_date=date_str)
            dfs = sb.get_data_frames()
            if not dfs:
                continue
            hdr = dfs[0]
            for _, r in hdr.iterrows():
                gid = str(r.get('GAME_ID'))
                home_id = r.get('HOME_TEAM_ID')
                away_id = r.get('VISITOR_TEAM_ID')
                # parse game date (EST)
                g_date = r.get('GAME_DATE_EST')
                if pd.isna(g_date):
                    continue
                dt_naive = pd.to_datetime(g_date).to_pydatetime()
                est = timezone(timedelta(hours=-5))
                dt_est = dt_naive.replace(tzinfo=est)
                dt_utc = dt_est.astimezone(timezone.utc)
                ct_str = dt_utc.isoformat().replace('+00:00', 'Z')
                upcoming.append({
                    'id': gid,
                    'commence_time': ct_str,
                    'home_team': id_to_full.get(home_id),
                    'away_team': id_to_full.get(away_id),
                    'home_abbr': id_to_abbr.get(home_id),
                    'away_abbr': id_to_abbr.get(away_id)
                })
        except ImportError:
            print('ERROR: nba_api.stats.endpoints.ScoreboardV2 not available; please install nba_api.')
            sys.exit(1)
        except Exception as e:
            print(f"WARNING: failed to fetch scoreboard for {date_str}: {e}")

    if not upcoming:
        print("No upcoming NBA games found in the specified date range.")
        sys.exit(0)

    print(f"Found {len(upcoming)} upcoming games:")
    for i, ev in enumerate(upcoming, 1):
        print(f"{i}. {ev['away_team']} @ {ev['home_team']} at {ev['commence_time']}")

    existing_df = None
    if os.path.exists(odds_path):
        existing_df = pd.read_csv(odds_path)
        if 'event_id' not in existing_df.columns:
            print("Existing odds file missing 'event_id'; recalculating all.")
        else:
            existing_df = update_correct(existing_df)
            seen = set(existing_df['event_id'].astype(str))
            missing = [ev for ev in upcoming if ev['id'] not in seen]
            if not missing:
                os.makedirs('data', exist_ok=True)
                existing_df.to_csv(odds_path, index=False)
                print("Updated past game results; no new games to calculate")
                print(existing_df.to_string(index=False))
                sys.exit(0)
            upcoming = missing
            print(f"Calculating odds for {len(upcoming)} new games...")

    # load trained model
    mdl_path = os.path.join('..', 'models', 'best_player_model.pkl')
    if not os.path.exists(mdl_path):
        print(f"ERROR: Model file not found at {mdl_path}")
        sys.exit(1)
    model_pkg = joblib.load(mdl_path)
    model = model_pkg['model']
    players_list = model_pkg['players']
    stats_cols = model_pkg['stats_cols']
    team_stats_cols = model_pkg.get('team_stats_cols', [])
    n_players = len(players_list)

    # load team game logs for feature computation
    tpath = os.path.join('data', 'team_game_logs.csv')
    if not os.path.exists(tpath):
        print("ERROR: team_game_logs.csv not found; cannot compute features.")
        sys.exit(1)
    tdf = pd.read_csv(tpath)
    tdf['GAME_DATE_DT'] = pd.to_datetime(tdf['GAME_DATE'], errors='coerce')

    def team_feature_diff(home_abbr, away_abbr, ctime):
        # convert to naive UTC for comparison
        if hasattr(ctime, 'tzinfo') and ctime.tzinfo is not None:
            cnaive = ctime.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            cnaive = ctime
        past = tdf[tdf['GAME_DATE_DT'] < cnaive]
        hdf = past[past['TEAM_ABBREVIATION'] == home_abbr]
        adf = past[past['TEAM_ABBREVIATION'] == away_abbr]
        if hdf.empty or adf.empty:
            return np.zeros(len(team_stats_cols))
        hmean = hdf[team_stats_cols].mean()
        amean = adf[team_stats_cols].mean()
        return (hmean - amean).values

    # predict and assemble records
    records = []
    for ev in upcoming:
        eid = ev['id']
        home_full = ev['home_team']
        away_full = ev['away_team']
        home_abbr = ev['home_abbr']
        away_abbr = ev['away_abbr']
        ct = ev['commence_time']
        ctime = datetime.fromisoformat(ct.replace('Z', '+00:00'))
        Xr = np.zeros(n_players)
        Xs = np.zeros(len(stats_cols))
        Xt = team_feature_diff(home_abbr, away_abbr, ctime)
        Xc = np.hstack([Xr, Xs, Xt]).reshape(1, -1)
        pred = model.predict(Xc)[0]
        if pred == 1:
            hp, ap = 0.996, 0.004
        else:
            hp, ap = 0.004, 0.996
        records.append({
            'event_id': eid,
            'commence_time': ct,
            'home_team': home_full,
            'away_team': away_full,
            'home_abbr': home_abbr,
            'away_abbr': away_abbr,
            'home_prob': hp,
            'away_prob': ap
        })

    df_new = pd.DataFrame(records)
    # merge with existing data if present
    if existing_df is not None:
        df_out = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_out = df_new
    # update correctness for all entries
    df_out = update_correct(df_out)
    os.makedirs('data', exist_ok=True)
    df_out.to_csv(odds_path, index=False)
    print(f"Saved odds to {odds_path}")
    print(df_out.to_string(index=False))


if __name__ == '__main__':
    main()