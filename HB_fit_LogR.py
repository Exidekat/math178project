#!/usr/bin/env python3
"""
Script to fit a logistic regression model to predict home-team wins
using per-game boxscore aggregates and home-court advantage.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from nba_api.stats.endpoints import boxscoretraditionalv2

def main():
    # --- 0) Read logs, ensuring Game_ID stays a zero-padded 10-character string ---
    data_path = os.path.join('data', 'player_game_logs.csv')
    df_logs = pd.read_csv(data_path, dtype={'Game_ID': str})
    df_logs['Game_ID'] = df_logs['Game_ID'].str.zfill(10)

    game_ids = df_logs['Game_ID'].unique()
    feature_dicts = []
    labels        = []

    for gid in game_ids:
        df_game = df_logs[df_logs['Game_ID'] == gid]
        if df_game.empty:
            # no sampled players in this game
            continue

        # parse away/home from MATCHUP (e.g. "DAL vs. SAS" or "DAL @ DEN")
        matchup = df_game['MATCHUP'].iloc[0]
        parts = matchup.split()
        if len(parts) == 3 and parts[1] == 'vs.':
            home_team = parts[0]
            away_team = parts[2]
        elif len(parts) == 3 and parts[1] == '@':
            away_team = parts[0]
            home_team = parts[2]
        else:
            print(f"⚠️  Unexpected matchup format: {matchup}")
            continue

        # fetch the official boxscore
        try:
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid)
            df_box = box.get_data_frames()[0]
        except Exception as e:
            print(f"⚠️  Could not fetch boxscore for game {gid}: {e}")
            continue

        # normalize column names
        df_box.columns = [c.upper() for c in df_box.columns]
        if 'TEAM_ABBREVIATION' not in df_box.columns or 'PTS' not in df_box.columns:
            print(f"⚠️  Unexpected boxscore format for {gid}")
            continue

        # aggregate points, rebounds, assists by team
        stats_by_team = (
            df_box
            .groupby('TEAM_ABBREVIATION')
            .agg({'PTS':'sum','REB':'sum','AST':'sum'})
        )

        # ensure both teams are present
        if not {home_team, away_team} <= set(stats_by_team.index):
            print(f"⚠️  Missing stats for {home_team}/{away_team} in game {gid}")
            continue

        home_stats = stats_by_team.loc[home_team]
        away_stats = stats_by_team.loc[away_team]

        # build feature dict: home–away differences + home-court dummy
        fd = {
            'DIFF_PTS':   home_stats['PTS'] - away_stats['PTS'],
            'DIFF_REB':   home_stats['REB'] - away_stats['REB'],
            'DIFF_AST':   home_stats['AST'] - away_stats['AST'],
            'HOME_COURT': 1
        }
        feature_dicts.append(fd)

        # label: did home team win? based on total points
        labels.append(1 if home_stats['PTS'] > away_stats['PTS'] else 0)

    # assemble feature matrix and label array
    X = pd.DataFrame(feature_dicts)
    y = np.array(labels)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # fit logistic regression
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    # evaluate
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")

    # save model + feature list
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': clf, 'features': X.columns.tolist()},
                'models/logreg_team_stats.pkl')
    print("Model and feature list saved to models/logreg_team_stats.pkl")

if __name__ == '__main__':
    main()
