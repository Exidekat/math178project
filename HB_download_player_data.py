#!/usr/bin/env python3
"""
Script to download NBA game logs for a set of players using nba_api.
Saves concatenated game logs to data/player_game_logs.csv
"""
import os
import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

def main():
    # Fetch list of all players and select 50
    all_players = players.get_players()
    selected_players = sorted(all_players, key=lambda x: x['full_name'])[:50]

    seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    logs = []
    for p in selected_players:
        pid = p['id']
        name = p['full_name']
        for season in seasons:
            time.sleep(0.6)
            try:
                gl = playergamelog.PlayerGameLog(player_id=pid, season=season)
                df = gl.get_data_frames()[0]
                df['PLAYER_NAME'] = name
                logs.append(df)
            except Exception as e:
                print(f"Error fetching {name} for {season}: {e}")

    if not logs:
        print("No game logs fetched. Exiting.")
        return

    all_logs = pd.concat(logs, ignore_index=True)
    os.makedirs('data', exist_ok=True)
    all_logs.to_csv('data/player_game_logs.csv', index=False)
    print("Saved game logs to data/player_game_logs.csv")

if __name__ == '__main__':
    main()