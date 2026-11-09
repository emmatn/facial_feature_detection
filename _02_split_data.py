# split data 

import os
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

# number per image 
ids = np.arange(config.n_images)

# split 27000 - 3000
train_val_ids, test_ids = train_test_split(ids, test_size=n_test, random_state=config.seed)

# split 26000 - 1000
train_pool, val_ids = train_test_split(train_val_ids, test_size=n_val, random_state=config.seed)

os.makedirs(split_path, exist_ok=True)
save_ids("test", np.sort(test_ids))
save_ids("val", np.sort(val_ids))
save_ids("train_pool", train_pool)

# for each train size, extract from pool 
for size in train_sizes:
    save_ids(f"train_{size}", np.sort(train_pool[:size]))

print(f"test: {len(test_ids)}\nval: {len(val_ids)}\ntrain pool: {len(train_pool)}")