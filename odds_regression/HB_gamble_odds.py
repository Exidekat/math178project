#!/usr/bin/env python3
"""
Script to fetch NBA game listings from The Odds API and predict game outcomes using the trained regression model.
"""
import os
import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import joblib

try:
    from nba_api.stats.static import teams as nba_teams
except ImportError:
    print("DEBUG: nba_api.stats.static import failed, some features will be disabled")
    nba_teams = None
# global ScoreboardV2 import removed; lazy import performed inside update_correct

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


# Debugging helper: update 'correct' column based on NBA API results
def update_correct(df):
    print(f"DEBUG update_correct: starting with df shape {df.shape}")
    # lazy import to avoid global dependency
    try:
        from nba_api.stats.endpoints import ScoreboardV2
    except ImportError:
        print("DEBUG update_correct: nba_api ScoreboardV2 not available, skipping update")
        return df
    if 'correct' not in df.columns:
        df['correct'] = np.nan
        print("DEBUG update_correct: added 'correct' column")
    now_utc = datetime.now(timezone.utc)
    for idx, row in df.iterrows():
        if pd.isna(row.get('correct')):
            ct = row.get('commence_time')
            if not isinstance(ct, str):
                continue
            ct2 = ct.replace('Z', '+00:00') if ct.endswith('Z') else ct
            try:
                cdt = datetime.fromisoformat(ct2)
            except Exception:
                continue
            print(f"DEBUG update_correct: checking game {row.get('event_id')} at {cdt}")
            if cdt < now_utc:
                date_str = cdt.strftime('%m/%d/%Y')
                try:
                    sb = ScoreboardV2(game_date=date_str)
                    dfs = sb.get_data_frames()
                    if len(dfs) > 1:
                        ls = dfs[1]
                        hm = ls['TEAM_ABBREVIATION'] == row.get('home_abbr')
                        am = ls['TEAM_ABBREVIATION'] == row.get('away_abbr')
                        if hm.any() and am.any():
                            home_pts = int(ls.loc[hm, 'PTS'].values[0])
                            away_pts = int(ls.loc[am, 'PTS'].values[0])
                            winner = 'home' if home_pts > away_pts else 'away'
                            predicted = 'home' if float(row.get('home_prob', 0)) > float(
                                row.get('away_prob', 0)) else 'away'
                            result = (winner == predicted)
                            df.at[idx, 'correct'] = result
                            print(
                                f"DEBUG update_correct: game {row.get('event_id')} winner={winner}, predicted={predicted}, correct={result}")
                        else:
                            print(
                                f"WARNING: no matching teams for game {row.get('home_abbr')} vs {row.get('away_abbr')} on {date_str}")
                except Exception as e:
                    print(f"WARNING: Failed to fetch result for game on {ct}: {e}")
    print("DEBUG update_correct: finished processing")
    return df


def main():
    # API key
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key:
        print("ERROR: ODDS_API_KEY not set in environment or .env")
        sys.exit(1)

    # Fetch odds from The Odds API
    print("Fetching upcoming NBA games from Odds API...")
    base_url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'decimal',
        'dateFormat': 'iso'
    }
    query = urllib.parse.urlencode(params)
    url = f"{base_url}?{query}"
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read().decode()
    except Exception as e:
        print(f"ERROR fetching odds: {e}")
        sys.exit(1)
    try:
        events = json.loads(data)
    except json.JSONDecodeError:
        print("ERROR: Could not parse Odds API response as JSON")
        sys.exit(1)

    # Filter for games in next 7 days
    now = datetime.now(timezone.utc) - timedelta(days=1)
    week_later = now + timedelta(days=8)
    upcoming = []
    for ev in events:
        ct = ev.get('commence_time')
        if not ct:
            continue
        # Normalize ISO timestamp
        if ct.endswith('Z'):
            ct2 = ct.replace('Z', '+00:00')
        else:
            ct2 = ct
        try:
            ctime = datetime.fromisoformat(ct2)
        except Exception:
            continue
        if now <= ctime <= week_later:
            upcoming.append(ev)
    if not upcoming:
        print("No upcoming NBA games in the next 7 days.")
        sys.exit(0)

    # Display games
    print(f"Found {len(upcoming)} upcoming games:")
    for i, ev in enumerate(upcoming, 1):
        away = ev.get('away_team')
        home = ev.get('home_team')
        print(f"{i}. {away} @ {home} at {ev.get('commence_time')}")

    odds_path = os.path.join('data', 'game_odds.csv')
    existing_df = pd.read_csv(odds_path)
    if 'event_id' not in existing_df.columns:
        print("Existing odds file missing 'event_id'; recalculating all.")
    else:
        # update past game correctness
        existing_df = update_correct(existing_df)
        seen = set(existing_df['event_id'].astype(str))
        missing = [ev for ev in upcoming if str(ev.get('id')) not in seen]
        if not missing:
            os.makedirs('data', exist_ok=True)
            existing_df.to_csv(odds_path, index=False)
            print("Updated past game results; no new games to calculate")
            print(existing_df.to_string(index=False))
            sys.exit(0)
        upcoming = missing
        print(f"Calculating odds for {len(upcoming)} new games...")

    # Load trained model
    mdl_path = os.path.join('../models', 'best_player_model.pkl')
    if not os.path.exists(mdl_path):
        print(f"ERROR: Model file not found at {mdl_path}")
        sys.exit(1)
    model_pkg = joblib.load(mdl_path)
    model = model_pkg['model']
    players_list = model_pkg['players']
    stats_cols = model_pkg['stats_cols']
    team_stats_cols = model_pkg.get('team_stats_cols', [])
    n_players = len(players_list)

    # Build mapping from full team name to 3-letter abbreviation
    team_map = {t['full_name']: t['abbreviation'] for t in nba_teams.get_teams()}

    # Load team game logs for season-avg stats
    tpath = os.path.join('data', 'team_game_logs.csv')
    if not os.path.exists(tpath):
        print("ERROR: team_game_logs.csv not found; cannot compute features.")
        sys.exit(1)
    tdf = pd.read_csv(tpath)
    # Parse dates
    tdf['GAME_DATE_DT'] = pd.to_datetime(tdf['GAME_DATE'], errors='coerce')

    def team_feature_diff(home_abbr, away_abbr, ctime):
        # average team stats up to game time
        # Ensure comparison between naive datetimes: drop tzinfo
        if hasattr(ctime, 'tzinfo') and ctime.tzinfo is not None:
            # convert to UTC naive
            ctime = ctime.astimezone(timezone.utc).replace(tzinfo=None)
        past = tdf[tdf['GAME_DATE_DT'] < ctime]
        hdf = past[past['TEAM_ABBREVIATION'] == home_abbr]
        adf = past[past['TEAM_ABBREVIATION'] == away_abbr]
        if hdf.empty or adf.empty:
            return np.zeros(len(team_stats_cols))
        hmean = hdf[team_stats_cols].mean()
        amean = adf[team_stats_cols].mean()
        return (hmean - amean).values

    # Prepare records
    records = []
    for ev in upcoming:
        eid = str(ev.get('id'))
        away_full = ev.get('away_team')
        home_full = ev.get('home_team')
        ct = ev.get('commence_time')
        # parse ctime
        ct2 = ct.replace('Z', '+00:00') if ct.endswith('Z') else ct
        ctime = datetime.fromisoformat(ct2)

        # map team abbrs
        home_abbr = team_map.get(home_full)
        away_abbr = team_map.get(away_full)
        if not home_abbr or not away_abbr:
            print(f"WARNING: Could not map {home_full} or {away_full}")
            continue

        # build feature vector: no player-level info, zero arrays
        Xr = np.zeros(n_players)
        Xs = np.zeros(len(stats_cols))
        Xt = team_feature_diff(home_abbr, away_abbr, ctime)
        Xc = np.hstack([Xr, Xs, Xt]).reshape(1, -1)
        pred = model.predict(Xc)[0]
        # assign probabilities
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
    # merge with existing if needed (always merge to preserve historical data)
    if existing_df is not None:
        df_out = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_out = df_new
    # update past game correctness for output
    df_out = update_correct(df_out)
    os.makedirs('data', exist_ok=True)
    df_out.to_csv(odds_path, index=False)
    print(f"Saved odds to {odds_path}")
    print(df_out.to_string(index=False))


if __name__ == '__main__':
    # support update-only mode before full execution
    if '--update-only' in sys.argv:
        odds_path = os.path.join('data', 'game_odds.csv')
        print("DEBUG: running in update-only mode")
        if os.path.exists(odds_path):
            df = pd.read_csv(odds_path)
            print(f"DEBUG: loaded existing file with shape {df.shape}")
            df = update_correct(df)
            print(f"DEBUG: after update_correct shape {df.shape}")
            os.makedirs('data', exist_ok=True)
            df.to_csv(odds_path, index=False)
            print(f"DEBUG: saved updated file to {odds_path}")
        else:
            print("DEBUG: no existing odds file to update")
        sys.exit(0)
    main()
