''' baseline: predict the average training landmark position for every face 
- measures how far off a random guess would be for all 14 landmarks, averaging all the misses on each face 
- guesses are pretty good, since all these faces are centered and about same size 
'''

import numpy as np
import pandas as pd
import config

train_size = 5000

labels = pd.read_csv("data/processed/labels.csv").set_index("idx")

def load_ids(name):
    return np.loadtxt(f"data/processed/splits/{name}.txt", dtype=int)

# average of each of the 28 coordinates over the training faces (missing landmarks skipped)
train_ids = load_ids(f"train_{train_size}")
mean_landmarks = labels.loc[train_ids, config.LANDMARK_COLS].mean().to_numpy()

for name in [f"train_{train_size}", "val", "test_frontal", "test_not_frontal"]:
    y = labels.loc[load_ids(name), config.LANDMARK_COLS].to_numpy()
    squared_error = (y - mean_landmarks) ** 2 
    mse = np.nanmean(squared_error)           
    px = mse ** 0.5 * 64
    print(f"{name}: {px:.2f} px")