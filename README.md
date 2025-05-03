# Math 178 Final Project

**Authors**
- Haydon Behl
- Arjun Bemarker
- Neal Chandra
- Brian Chiang
- Karthik Renuprasad

**Sections**
1. **NBA Play-By-Play Predictions** via Hidden Markov Models
2. **NBA Game Predictions** via Regression
3. **NBA Gambling Bot** via Polymarket

# 1. NBA Play-By-Play Predictions

- Authored by Arjun Bemarker, Neal Chandra, Brian Chiang, Karthik Renuprasad

WORK IN PROGRESS

# 2. NBA Game Predictions

- Authored by Haydon Behl

This project builds and evaluates predictive models for NBA game outcomes, using player-level and team-level game statistics.

## Iteration History

1. **Baseline Roster Encoding**
   - Features: +1/–1 indicator per sampled player on home/away teams
   - Model: `LinearRegression`
   - Test accuracy: ~51.6%

2. **Player Stat Aggregation & Model Comparison**
   - Added EDA for data shape, numeric columns, and top players by games played
   - Engineered aggregated player stat features (home minus away sums)
   - Compared: 
     - `LinearRegression` on roster only (51.6%)
     - `LogisticRegression` on player stats only (54.9%)
     - `LogisticRegression` on combined roster+stats (57.3%)

3. **Team-Level Feature Integration**
   - Extended `HB_download_player_data.py` to fetch and save team game logs (`team_game_logs.csv`)
   - In `HB_fit_regression.py`, load team stats (W, L, W_PCT, FGM, PTS, etc.) and compute home–away differences
   - Final modeling compares all feature sets and selects the best-performing model

## Usage

1. Fetch data (players and teams):
   ```bash
   python HB_download_player_data.py
   ```
2. Fit and evaluate models:
   ```bash
   python HB_fit_regression.py
   ```

## Final Results Summary

After integrating roster, player stat, and team stat features, we evaluated three models:

- LinearRegression (roster only) accuracy: **51.6%**
- LogisticRegression (player stats only) accuracy: **54.9%**
- LogisticRegression (combined roster + stats + team) accuracy: **99.6%**

The best model is the combined `LogisticRegression`, with near-perfect test accuracy thanks to the inclusion of team point-differential features.

Top 5 features by absolute coefficient:
  1. team_PTS (team points difference)
  2. team_FGM (team field goals made difference)
  3. team_FTM (team free throws made difference)
  4. team_FG3M (team 3-pt field goals made difference)
  5. stat_FTM (player free throws made difference)

The final model and metadata are saved as `models/best_player_model.pkl`, containing:
  - `model`: the trained logistic regression object
  - `players`: list of sampled player names
  - `stats_cols`: list of aggregated player stat columns
  - `team_stats_cols`: list of aggregated team stat columns
  - `feature_set`: identifier for the combined feature model

# 3. NBA Gambling Bot

GAMBLING IN PROGRESS