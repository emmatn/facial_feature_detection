# 02: Splitting the Data

import os
import numpy as np
import pandas as pd
import config

split_path = "data/processed/splits"
labels_path = "data/processed/labels.csv"

n_test = 3000
n_val = 1000
train_sizes = [2000, 5000, 10000]  # smaller training samples to compare
turned_degrees = 30              # yaw above this counts as a turned head

def save_ids(name, ids):
    np.savetxt(f"{split_path}/{name}.txt", ids, fmt="%d")

def load_ids(name):
    return np.loadtxt(f"{split_path}/{name}.txt", dtype=int)

def make_splits():
    rng = np.random.default_rng(config.seed)
    shuffled = rng.permutation(config.n_images)

    val_start = n_test
    train_start = n_test + n_val
    test_ids = shuffled[:val_start]
    val_ids = shuffled[val_start:train_start]
    train_pool = shuffled[train_start:]

    os.makedirs(split_path, exist_ok=True)
    save_ids("test", np.sort(test_ids))
    save_ids("val", np.sort(val_ids))
    # pool stays shuffled so its first N entries are a random sample
    save_ids("train_pool", train_pool)
    for size in train_sizes:
        save_ids(f"train_{size}", np.sort(train_pool[:size]))

    print(f"test: {len(test_ids)}  val: {len(val_ids)}  train pool: {len(train_pool)}")
    print(f"train samples: {train_sizes}")

def check_splits():
    """Share of hard faces in each split; they should be similar."""
    labels = pd.read_csv(labels_path)
    if len(labels) < config.n_images:
        print("labels.csv is from a test run; skipping split check")
        return
    labels = labels.set_index("idx")

    split_names = ["test", "val"] + [f"train_{size}" for size in train_sizes]
    for name in split_names:
        split = labels.loc[load_ids(name)]
        glasses = split.Eyeglasses.mean()
        hat = split.Wearing_Hat.mean()
        turned = (split.Yaw.abs() > turned_degrees).mean()
        print(f"{name}: {len(split)} faces, glasses {glasses:.1%}, "
              f"hat {hat:.1%}, turned {turned:.1%}")

if __name__ == "__main__":
    make_splits()
    check_splits()