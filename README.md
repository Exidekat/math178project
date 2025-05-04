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
3. **NBA Gambling Bot** via Betfair

# 1. NBA Play-By-Play Predictions

- Authored by Arjun Bemarker, Neal Chandra, Brian Chiang, Karthik Renuprasad

# NBA Play-By-Play Predictions via Hidden Markov Models
- Data Acquisition: We use the `nba_api` PlayByPlayV3 endpoint to fetch detailed event logs for NBA games.
- State Definition: Each play (e.g., shot attempt, rebound, foul) is encoded as a hidden state based on the `actionType` column.
- Transition Matrix: We construct a Markov transition matrix by counting consecutive state occurrences and normalizing to probabilities.
- Evaluation: Transition probabilities are stored in `playbyplay_markov/transition_matrix.csv`, and notebooks explore sequence likelihoods, Pearson correlations between predicted vs. observed transitions, and visualizations of state flows.
- Use Cases: The HMM framework enables us to predict next-play events, simulate possession sequences, and inform in-play strategy.

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

- Authored by Haydon Behl

# NBA Gambling Bot via Odds and Exchange APIs
- Odds Acquisition: We fetch upcoming NBA game listings and existing market odds using The Odds API in `odds_regression/HB_gamble_odds.py`, writing predictions to `data/game_odds.csv`.
- Market Mapping: Betfair market and runner IDs are dynamically discovered and persisted in `data/market_maps.csv`, enabling cross-referencing between our event IDs and exchange markets.
- Bet Placement (Betfair):
  - `odds_gambling/HB_betfair_gamble.py` logs into Betfair via `betfairlightweight`, computes Kelly criterion stakes based on our model probabilities, and places LIMIT BACK orders on MATCH_ODDS markets.
  - Bet results and responses are saved to `data/betfair_bets.csv` for audit and tracking.
- Bet Placement (Pinnacle) [Upcoming]:
  - We will mirror the Betfair integration for Pinnacle Sports using their REST API, with credentials in `PINA_USERNAME` and `PINA_PASSWORD`, and similar stake-sizing logic.
- Kelly Staking Calculator:
  - `odds_gambling/HB_kelly_staking.py` computes recommended stakes per upcoming game based on the Kelly criterion, using model probabilities from `data/game_odds.csv` and market odds from The Odds API.
- Configuration:
  - Environment variables (in `.env`) for API credentials and `TOTAL_FUNDS` are required:
    ```bash
    BETFAIR_USERNAME=...
    BETFAIR_PASSWORD=...
    BETFAIR_APP_KEY=...
    PINA_USERNAME=...
    PINA_PASSWORD=...
    ODDS_API_KEY=...
    TOTAL_FUNDS=1000.00
    ```
- Scripts & Usage:
  1. Generate game odds:
     ```bash
     python odds_regression/HB_gamble_odds.py
     ```
  2. Place automated bets on Betfair:
     ```bash
     python odds_gambling/HB_betfair_gamble.py
     ```
  3. Calculate recommended Kelly stakes:
     ```bash
     python odds_gambling/HB_kelly_staking.py
     ```
  4. (Future) Place bets on Pinnacle:
     ```bash
     python odds_gambling/HB_pinnacle_gamble.py
     ```