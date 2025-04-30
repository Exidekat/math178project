#!/usr/bin/env python3
"""
Script to fit a multiple linear regression model to predict game outcomes
based on which of the selected players played for home vs away teams.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def main():
    data_path = os.path.join('data', 'player_game_logs.csv')
    df = pd.read_csv(data_path)

    players_list = sorted(df['PLAYER_NAME'].unique())
    game_ids     = df['Game_ID'].unique()
    n_players    = len(players_list)

    # we'll collect rows only for games where at least one sampled player appeared
    X_rows = []
    y_rows = []

    for gid in game_ids:
        df_game = df[df['Game_ID'] == gid].copy()
        if df_game.empty:
            # no sampled players in this game
            continue

        # Derive the 3-letter team codes from the first 3 chars of MATCHUP
        df_game['TEAM_ABBR'] = df_game['MATCHUP'].str[:3]

        # --- parse home vs away directly from the MATCHUP text ---
        # e.g. "ORL @ CHI"  →  away="ORL", home="CHI"
        #      "LAL vs. BOS" →  away="LAL", home="BOS"
        parts      = df_game['MATCHUP'].iloc[0].split()
        away_team  = parts[0]
        home_team  = parts[2]

        # build boolean masks based on those codes
        home_mask = df_game['TEAM_ABBR'] == home_team
        away_mask = df_game['TEAM_ABBR'] == away_team

        # pick a WL from whichever side we have data for
        if home_mask.any():
            wl = df_game.loc[home_mask, 'WL'].iloc[0]
            win_team = home_team if wl == 'W' else away_team
        elif away_mask.any():
            wl = df_game.loc[away_mask, 'WL'].iloc[0]
            win_team = away_team if wl == 'W' else home_team
        else:
            # weird case: we have rows but TEAM_ABBR parsing failed
            print(f"⚠️  Couldn’t parse teams for game {gid}: {df_game['MATCHUP'].unique()}")
            continue

        # record the label: 1 if home squad won, else 0
        y_rows.append(1 if win_team == home_team else 0)

        # build the +1/–1 feature vector
        row = np.zeros(n_players, dtype=int)
        home_players = set(df_game.loc[home_mask, 'PLAYER_NAME'])
        away_players = set(df_game.loc[away_mask, 'PLAYER_NAME'])
        for j, pname in enumerate(players_list):
            if pname in home_players:
                row[j] = 1
            elif pname in away_players:
                row[j] = -1
        X_rows.append(row)

    print(X_rows.columns)

    # stack into arrays
    X = np.vstack(X_rows)
    y = np.array(y_rows)

    # split, fit, evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y_test)

    print(f"Test R^2:       {r2:.3f}")
    print(f"Test accuracy: {accuracy:.3f}")

    # persist the model
    os.makedirs('models', exist_ok=True)
    model_dict = {'model': model, 'players': players_list}
    joblib.dump(model_dict, 'models/mlr_player_model.pkl')
    print("Model saved to models/mlr_player_model.pkl")


if __name__ == '__main__':
    main()
