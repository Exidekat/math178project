from nba_api.stats.endpoints import playbyplayv3
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy.stats import pearsonr
import functools

log = playbyplayv3.PlayByPlayV3(game_id="0022400849")
frames = log.get_data_frames()
gamelog = frames[0]
gamelog.to_csv('playbyplay.csv')

gamelog['state'] = gamelog['actionType'].fillna('')
gamelog['next_state'] = gamelog['state'].shift(-1)

# Drop the last row if next_state is NaN
transitions = gamelog.dropna(subset=['next_state'])

# Now count transitions
transition_counts = transitions.groupby(['state', 'next_state']).size().unstack(fill_value=0)

# Normalize to probabilities
transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)

# Display the transition matrix
transition_matrix.to_csv('transition_matrix.csv')
print(transition_matrix)
