''' shows predicted vs actual landmarks on a few frontal and non-frontal test faces
    green dots = actual, red x = predicted, yellow line = how far off each point is
'''
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import config
import _11_model

n_per_group = 5 # faces per row

images = np.load("data/processed/images.npy")
labels = pd.read_csv("data/processed/labels.csv").set_index("idx")

# rebuild the model and load the trained weights from _12_train.py
model = _11_model.CNN(n_outputs=28)
model.load_state_dict(torch.load(f"{config.results_path}/model.pt"))
model.eval()

# pick random faces from each test set (same faces every run because of the seed)
#rng = np.random.default_rng(config.seed)
groups = {}

groups["test_frontal"] = [7, 181, 510, 666, 1655]
groups["test_not_frontal"] = [4, 2689, 368, 5002, 6234] # random
#groups["test_not_frontal"] = [0, 49, 236, 299, 1688] # slightly turned 
#groups["test_not_frontal"] = [15709, 11885, 12407, 16747, 28813] # rlly turned 


fig, axes = plt.subplots(2, n_per_group, figsize=(4 * n_per_group, 8.5))
for row, (name, ids) in enumerate(groups.items()):
    # same preprocessing as training: small image, color x height x width, 0-1
    x = torch.from_numpy(images[ids]).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        pred = model(x).view(-1, 14, 2).numpy()
    true = labels.loc[ids, config.LANDMARK_COLS].to_numpy().reshape(-1, 14, 2)

    for col, idx in enumerate(ids):
        ax = axes[row, col]
        # draw on the full-size photo so the points are easy to see
        img = Image.open(f"{config.img_path}/{idx}.jpg")
        t = true[col] * img.width
        p = pred[col] * img.width

        ax.imshow(img)
        for (tx, ty), (px, py) in zip(t, p):
            if not np.isnan(tx): # skip landmarks missing from the labels
                ax.plot([tx, px], [ty, py], color="yellow", linewidth=1)
        ax.scatter(t[:, 0], t[:, 1], c="lime", s=25, label="actual")
        ax.scatter(p[:, 0], p[:, 1], c="blue", s=25, marker="x", label="predicted")

        # average distance between predicted and actual, in pixels of the image the model sees
        miss = np.nanmean(np.linalg.norm(true[col] - pred[col], axis=1)) * config.image_size
        group = "frontal" if name == "test_frontal" else "non-frontal"
        ax.set_title(f"{group} | image {idx}\navg miss {miss:.2f} px")
        ax.axis("off")

axes[0, 0].legend(loc="lower left")
fig.tight_layout()
fig.savefig(f"{config.results_path}/predictions3.png", dpi=120)
plt.show()