#!/usr/bin/env python3
"""
Script to fit a multiple linear regression model to predict game outcomes
based on which of the selected players played for home vs away teams.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def main():
    data_path = os.path.join('data', 'player_game_logs.csv')
    df = pd.read_csv(data_path)

    # Exploratory Data Analysis (EDA)
    print(f"Data shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    # Identify numeric columns for potential aggregated features
    stats_cols = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','STL','BLK','TOV','PF','PTS']
    existing_stats = [c for c in stats_cols if c in df.columns]
    print("Numeric (stat) columns:", existing_stats)
    # Show top 10 players by number of games
    game_counts = df.groupby('PLAYER_NAME')['Game_ID'].nunique()
    print("Top 10 players by games played:")
    print(game_counts.sort_values(ascending=False).head(10))

    players_list = sorted(df['PLAYER_NAME'].unique())
    game_ids     = df['Game_ID'].unique()
    n_players    = len(players_list)

    # Load team-level game logs for additional features
    team_path = os.path.join('data', 'team_game_logs.csv')
    if os.path.exists(team_path):
        team_df = pd.read_csv(team_path)
        print(f"Team game logs shape: {team_df.shape}")
        possible_team_stats = [
            'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'PF', 'PTS'
        ]
        team_stats_cols = [c for c in possible_team_stats if c in team_df.columns]
        print("Team numeric stats columns:", team_stats_cols)
    else:
        print("WARNING: team_game_logs.csv not found; skipping team-level features.")
        team_df = None
        team_stats_cols = []

    # we'll collect rows for roster features, aggregated stat features, team features, and labels
    X_rows = []
    X_stat_rows = []
    X_team_rows = []
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
            print(f"️Couldn’t parse teams for game {gid}: {df_game['MATCHUP'].unique()}")
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
        # aggregate stat features: difference of summed stats between home and away
        home_stats = df_game.loc[home_mask, existing_stats].sum()
        away_stats = df_game.loc[away_mask, existing_stats].sum()
        stat_row = (home_stats - away_stats).values
        X_stat_rows.append(stat_row)
        # aggregate team-level features: difference of team stats
        if team_df is not None:
            df_t = team_df[team_df['Game_ID'] == gid]
            df_th = df_t[df_t['TEAM_ABBREVIATION'] == home_team]
            df_ta = df_t[df_t['TEAM_ABBREVIATION'] == away_team]
            if not df_th.empty and not df_ta.empty:
                home_t_stats = df_th[team_stats_cols].iloc[0]
                away_t_stats = df_ta[team_stats_cols].iloc[0]
                team_row = (home_t_stats - away_t_stats).values
            else:
                team_row = np.zeros(len(team_stats_cols), dtype=float)
        else:
            team_row = np.zeros(len(team_stats_cols), dtype=float)
        X_team_rows.append(team_row)

    # stack into arrays for roster, stat, and team features
    X_roster = np.vstack(X_rows)
    X_stats = np.vstack(X_stat_rows)
    y = np.array(y_rows)
    print(f"Roster feature matrix shape: {X_roster.shape}")
    print(f"Stat feature matrix shape:   {X_stats.shape}")
    if team_stats_cols:
        X_team = np.vstack(X_team_rows)
        print(f"Team feature matrix shape:   {X_team.shape}")
    else:
        X_team = np.zeros((X_roster.shape[0], 0))
        print("No team features used; skipping team feature matrix.")

    # Split data for different feature sets
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_roster, y, test_size=0.2, random_state=42)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_stats, y, test_size=0.2, random_state=42)
    # combine all features
    Xc = np.hstack([X_roster, X_stats, X_team])
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, y, test_size=0.2, random_state=42)

    # 1) Baseline: Linear Regression on roster features
    linreg = LinearRegression()
    linreg.fit(Xr_train, yr_train)
    y_pred_lr = (linreg.predict(Xr_test) > 0.5).astype(int)
    acc_lr = np.mean(y_pred_lr == yr_test)
    print(f"Linear Regression (roster) accuracy:        {acc_lr:.3f}")

    # 2) Logistic Regression on stat features
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(Xs_train, ys_train)
    acc_log = logreg.score(Xs_test, ys_test)
    print(f"Logistic Regression (stats) accuracy:       {acc_log:.3f}")

    # 3) Logistic Regression on combined features
    logreg_c = LogisticRegression(max_iter=1000, random_state=42)
    logreg_c.fit(Xc_train, yc_train)
    acc_logc = logreg_c.score(Xc_test, yc_test)
    print(f"Logistic Regression (combined) accuracy:    {acc_logc:.3f}")

    # Choose best model
    acc_dict = {'linreg_roster': acc_lr, 'logreg_stats': acc_log, 'logreg_combined': acc_logc}
    best_key = max(acc_dict, key=acc_dict.get)
    best_model = {'linreg_roster': linreg, 'logreg_stats': logreg, 'logreg_combined': logreg_c}[best_key]
    print(f"Best model: {best_key} with accuracy {acc_dict[best_key]:.3f}")
    # Feature importance ranking
    if hasattr(best_model, 'coef_'):
        coefs = best_model.coef_[0]
    else:
        coefs = best_model.feature_importances_
    # feature names: roster, stats, then team stats
    feature_names = [f"player_{p}" for p in players_list] + [f"stat_{s}" for s in existing_stats] + [f"team_{t}" for t in team_stats_cols]
    importances = sorted(zip(feature_names, np.abs(coefs)), key=lambda x: x[1], reverse=True)
    print("Features ranking by importance:")
    for fname, imp in importances:
        print(f"{fname}: {imp:.4f}")

    # Persist the best model
    os.makedirs('models', exist_ok=True)
    model_dict = {
        'model': best_model,
        'players': players_list,
        'stats_cols': existing_stats,
        'feature_set': best_key
    }
    joblib.dump(model_dict, 'models/best_player_model.pkl')
    print("Best model saved to models/best_player_model.pkl")


if __name__ == '__main__':
    main()
