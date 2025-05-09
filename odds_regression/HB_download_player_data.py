#!/usr/bin/env python3
"""
Script to download NBA game logs for a set of players using nba_api.
Saves concatenated game logs to data/player_game_logs.csv
"""
import os
import time
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, teamgamelog

def main():
    # Fetch list of all players and select 50
    all_players = players.get_players()
    selected_players = sorted(all_players, key=lambda x: x['full_name'])[:50]

    seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    logs = []
    # Optionally skip player logs download (e.g., for testing)
    if os.environ.get('SKIP_PLAYER_LOG_DOWNLOAD', '0') == '1':
        print("Skipping player game logs download (SKIP_PLAYER_LOG_DOWNLOAD set).")
    else:
        for p in selected_players:
            pid = p['id']
            name = p['full_name']
            for season in seasons:
                time.sleep(0.25)
                try:
                    print(f"Fetching {name} season {season} data")
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
        # Save player game logs
        pgl_path = os.path.join('data', 'player_game_logs.csv')
        all_logs.to_csv(pgl_path, index=False)
        print(f"Saved player game logs to {pgl_path}")

    # Optionally skip team logs download (e.g., for testing)
    if os.environ.get('SKIP_TEAM_LOG_DOWNLOAD', '0') == '1':
        print("Skipping team game logs download (SKIP_TEAM_LOG_DOWNLOAD set).")
    else:
        print("Downloading team game logs...")
        team_list = teams.get_teams()
        team_logs = []
        for season in seasons:
            for team in team_list:
                time.sleep(0.5)
                try:
                    tg = teamgamelog.TeamGameLog(team_id=team['id'], season=season)
                    df_t = tg.get_data_frames()[0]
                    df_t['TEAM_ABBREVIATION'] = team.get('abbreviation')
                    team_logs.append(df_t)
                except Exception as e:
                    print(f"Error fetching team {team.get('full_name')} for {season}: {e}")
        if team_logs:
            all_team_logs = pd.concat(team_logs, ignore_index=True)
            tgl_path = os.path.join('data', 'team_game_logs.csv')
            all_team_logs.to_csv(tgl_path, index=False)
            print(f"Saved team game logs to {tgl_path}")
        else:
            print("No team game logs fetched.")

if __name__ == '__main__':
    main()