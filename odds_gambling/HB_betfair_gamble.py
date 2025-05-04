#!/usr/bin/env python3
"""
Script to place bets on Betfair Exchange using predicted odds in data/game_odds.csv.
Requires BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY, TOTAL_FUNDS in environment or .env.
Maintains a mapping file data/market_maps.csv to relate events to Betfair market and runner IDs.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import betfairlightweight
from betfairlightweight import APIClient, filters

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
    # Credentials & funds
    user = os.getenv('BETFAIR_USERNAME')
    pw   = os.getenv('BETFAIR_PASSWORD')
    app_key = os.getenv('BETFAIR_APP_KEY')
    tf = os.getenv('TOTAL_FUNDS')
    if not all([user, pw, app_key, tf]):
        print("ERROR: Please set BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY, TOTAL_FUNDS")
        sys.exit(1)
    total_funds = float(tf)

    # Read predicted game odds
    odds_path = os.path.join('data', 'game_odds.csv')
    if not os.path.exists(odds_path):
        print(f"ERROR: {odds_path} not found")
        sys.exit(1)
    df = pd.read_csv(odds_path, parse_dates=['commence_time'])
    # filter future games
    now = datetime.now(timezone.utc)
    future = df[df['commence_time'] > now]
    if future.empty:
        print("No future games to bet on.")
        sys.exit(0)

    # Load or init market map
    map_path = os.path.join('data', 'market_maps.csv')
    if os.path.exists(map_path):
        map_df = pd.read_csv(map_path, dtype=str)
    else:
        map_df = pd.DataFrame(columns=['event_id','market_id','home_runner_id','away_runner_id'])

    # Login to Betfair
    client = APIClient(user, pw, app_key)
    client.login()
    # find basketball eventTypeId
    evts = client.betting.list_event_types().event_types
    bball = next((e for e in evts if e.event_type.name.lower() == 'basketball'), None)
    if not bball:
        print("ERROR: Basketball event type not found on Betfair")
        sys.exit(1)
    et_id = bball.event_type.id
    # find NBA competitionId
    comps = client.betting.list_competitions(event_type_ids=[et_id]).competitions
    nba_comp = next((c for c in comps if 'nba' in c.competition.name.lower()), None)
    if not nba_comp:
        print("ERROR: NBA competition not found on Betfair")
        sys.exit(1)
    comp_id = nba_comp.competition.id

    # Determine mapping for each future event
    # We'll fetch market catalogues around their start times in batch
    # Collect time window
    ts = future['commence_time'].dt.tz_convert(timezone.utc)
    window_from = (ts.min() - timedelta(minutes=15)).isoformat()
    window_to   = (ts.max() + timedelta(minutes=15)).isoformat()
    mf = filters.market_filter(
        event_type_ids=[et_id],
        competition_ids=[comp_id],
        market_start_time={'from': window_from, 'to': window_to},
        market_type_codes=['MATCH_ODDS']
    )
    catalog = client.betting.list_market_catalogue(
        filter=mf,
        market_projection=['RUNNER_METADATA','EVENT'],
        max_results=200
    ).market_catalogue

    # map by matching home/away in market.event.name
    for _, row in future.iterrows():
        eid = str(row['event_id'])
        if eid in map_df['event_id'].values:
            continue
        home = row['home_team']
        away = row['away_team']
        # find matching market
        mkt = next(
            (m for m in catalog
             if home.lower() in m.event.name.lower() and away.lower() in m.event.name.lower()),
            None
        )
        if not mkt:
            print(f"WARNING: No Betfair market found for {away} @ {home}")
            continue
        # find runner IDs
        runners = {r.runner_name: r.selection_id for r in mkt.runners}
        home_id = runners.get(home)
        away_id = runners.get(away)
        if not home_id or not away_id:
            print(f"WARNING: Runner IDs not found in market {mkt.market_id}")
            continue
        map_df = map_df.append({
            'event_id': eid,
            'market_id': mkt.market_id,
            'home_runner_id': home_id,
            'away_runner_id': away_id
        }, ignore_index=True)
    # persist mapping
    os.makedirs('data', exist_ok=True)
    map_df.to_csv(map_path, index=False)

    # Place bets
    bets = []
    for _, row in future.iterrows():
        eid = str(row['event_id'])
        mrow = map_df[map_df['event_id'] == eid]
        if mrow.empty:
            continue
        market_id = mrow['market_id'].iloc[0]
        # decide side
        ph = float(row['home_prob']); pa = float(row['away_prob'])
        if ph >= pa:
            sel = int(mrow['home_runner_id'].iloc[0]); p = ph
        else:
            sel = int(mrow['away_runner_id'].iloc[0]); p = pa
        # get best available back odds
        mb = client.betting.list_market_book(
            market_ids=[market_id],
            price_projection=filters.price_projection(price_data=['EX_BEST_OFFERS'])
        ).market_book[0]
        runner = next(r for r in mb.runners if r.selection_id == sel)
        if not runner.ex.available_to_back:
            print(f"No available back offers for selection {sel}")
            continue
        o = runner.ex.available_to_back[0].price
        # Kelly stake: f = (p*(o-1) - (1-p))/(o-1)
        b = o - 1
        f = max(0.0, (p*b - (1-p)) / b) if b > 0 else 0.0
        stake = round(f * total_funds, 2)
        if stake <= 0:
            print(f"Skipping zero stake for event {eid}")
            continue
        instr = filters.place_instruction(
            selection_id=sel,
            order_type='LIMIT',
            side='BACK',
            limit_order=filters.limit_order(price=o, size=stake, persistence_type='PERSIST')
        )
        resp = client.betting.place_orders(market_id, [instr])
        bets.append({
            'event_id': eid,
            'market_id': market_id,
            'selection_id': sel,
            'odds': o,
            'stake': stake,
            'bet_time': datetime.now(timezone.utc).isoformat(),
            'status': json.dumps(resp._asdict())
        })
        print(f"Placed bet on {sel} @ {o} for stake {stake}")
    # logout
    client.logout()
    # save bets
    if bets:
        dfb = pd.DataFrame(bets)
        bets_path = os.path.join('data', 'betfair_bets.csv')
        if os.path.exists(bets_path):
            dfb_all = pd.concat([pd.read_csv(bets_path, dtype=str), dfb], ignore_index=True)
        else:
            dfb_all = dfb
        dfb_all.to_csv(bets_path, index=False)
        print(f"All bets saved to {bets_path}")

if __name__ == '__main__':
    main()