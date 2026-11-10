''' split data 
    train/val on frontal faces only, test on frontal and non-frontal faces
    determined by median of the dataset, what faces are considered "straight on"
'''

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config

split_path = "data/processed/splits"

# all adjustable 
n_test = 3000
n_val = 1000
train_sizes = [2000, 5000, 10000] 

# build file path - writes whole numbers 
def save_ids(name, ids):
    np.savetxt(f"{split_path}/{name}.txt", ids, fmt="%d")

# FRONTAL IMAGES ONLY FOR TRAINING 
max_yaw, max_pitch, max_roll = 10, 10, 10
labels = pd.read_csv("data/processed/labels.csv")
yaw = (labels["Yaw"] - labels["Yaw"].median()).abs()
pitch = (labels["Pitch"] - labels["Pitch"].median()).abs()
roll = labels["Roll"].abs()

# all have to be True 
frontal = (yaw <= max_yaw) & (pitch <= max_pitch) & (roll <= max_roll)

# number per image 
#ids = np.arange(config.n_images)
frontal_ids = labels.loc[frontal, "idx"].to_numpy()
not_frontal_ids = labels.loc[~frontal, "idx"].to_numpy()

# frontal faces: test, then val, then the training pool
train_val_ids, test_frontal_ids = train_test_split(frontal_ids, test_size=n_test, random_state=config.seed)
train_pool, val_ids = train_test_split(train_val_ids, test_size=n_val, random_state=config.seed)

os.makedirs(split_path, exist_ok=True)
save_ids("test_frontal", np.sort(test_frontal_ids))
save_ids("test_not_frontal", np.sort(not_frontal_ids))
save_ids("val", np.sort(val_ids))
save_ids("train_pool", train_pool)

# for each train size, extract from pool 
for size in train_sizes:
    save_ids(f"train_{size}", np.sort(train_pool[:size]))

# print(f"test: {len(test_ids)}\nval: {len(val_ids)}\ntrain pool: {len(train_pool)}")
print(f"frontal: {len(frontal_ids)}  not frontal: {len(not_frontal_ids)}")
print(f"test_frontal: {len(test_frontal_ids)}\ntest_not_frontal: {len(not_frontal_ids)}\n"
      f"val: {len(val_ids)}\ntrain pool: {len(train_pool)}")