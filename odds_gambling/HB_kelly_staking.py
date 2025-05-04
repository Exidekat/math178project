#!/usr/bin/env python3
"""
Script to compute recommended Kelly-staking for upcoming NBA games
using predicted win probabilities (data/game_odds.csv) and market odds
from The Odds API.
"""
import os
import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
import pandas as pd

# Load environment variables from .env if present
env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k, v)

def main():
    # TOTAL_FUNDS
    tf_env = os.getenv('TOTAL_FUNDS')
    if tf_env:
        try:
            total_funds = float(tf_env)
        except ValueError:
            print("Invalid TOTAL_FUNDS value; must be numeric.")
            sys.exit(1)
    else:
        val = input("Enter total funds for Kelly staking: ")
        try:
            total_funds = float(val)
        except ValueError:
            print("Invalid input; exiting.")
            sys.exit(1)

    # Load predictions
    odds_path = os.path.join('data', 'game_odds.csv')
    if not os.path.exists(odds_path):
        print(f"ERROR: {odds_path} not found.")
        sys.exit(1)
    preds = pd.read_csv(odds_path, dtype={'event_id': str})
    preds.set_index('event_id', inplace=True)

    # Fetch upcoming games via The Odds API
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key:
        print("ERROR: ODDS_API_KEY not set.")
        sys.exit(1)
    base = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'decimal',
        'dateFormat': 'iso'
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read().decode()
        events = json.loads(data)
    except Exception as e:
        print(f"ERROR fetching odds: {e}")
        sys.exit(1)

    # Filter next 7 days
    now = datetime.now(timezone.utc)
    week_later = now + timedelta(days=7)
    upcoming = []
    for ev in events:
        ct = ev.get('commence_time')
        if not ct:
            continue
        ts = ct.replace('Z', '+00:00') if ct.endswith('Z') else ct
        try:
            ctime = datetime.fromisoformat(ts)
        except Exception:
            continue
        if now <= ctime <= week_later:
            upcoming.append((ev, ctime))
    if not upcoming:
        print("No upcoming NBA games in the next 7 days.")
        sys.exit(0)

    # Compute Kelly stakes
    results = []
    for ev, ctime in upcoming:
        eid = str(ev.get('id'))
        pred = None
        if eid in preds.index:
            row = preds.loc[eid]
            ph = float(row.get('home_prob', 0))
            pa = float(row.get('away_prob', 0))
            # decide side
            if ph >= pa:
                team = ev.get('home_team')
                p = ph
            else:
                team = ev.get('away_team')
                p = pa
        else:
            print(f"No prediction for event {eid}; skipping.")
            continue
        # find market odds for predicted team
        price = None
        for bm in ev.get('bookmakers', []):
            for m in bm.get('markets', []):
                if m.get('key') != 'h2h':
                    continue
                for o in m.get('outcomes', []):
                    if o.get('name') == team:
                        price = o.get('price')
                        break
                if price is not None:
                    break
            if price is not None:
                break
        if price is None:
            print(f"No market odds found for {team} in event {eid}")
            continue
        # Kelly fraction
        b = price - 1
        q = 1 - p
        f = (p * b - q) / b if b > 0 else 0.0
        f = max(f, 0.0)
        stake = round(f * total_funds, 2)
        results.append({
            'event_id': eid,
            'commence_time': ctime.isoformat(),
            'home_team': ev.get('home_team'),
            'away_team': ev.get('away_team'),
            'predicted_team': team,
            'model_prob': p,
            'market_odds': price,
            'kelly_fraction': round(f, 4),
            'stake': stake
        })

    # Display recommended stakes
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    else:
        print("No stakes to recommend.")

if __name__ == '__main__':
    main()