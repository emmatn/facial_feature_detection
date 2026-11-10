''' training the model 
    prints training and validation error in pixels each epoch
    reports error on the frontal and non-frontal test sets
'''

import numpy as np
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import config
import _11_model

# set seed
torch.manual_seed(config.seed)

train_size = 5000 
n_epochs = 20
batch_size = 32 # faces per update

# Loading the data 
images = torch.from_numpy(np.load("data/processed/images.npy"))
labels = pd.read_csv("data/processed/labels.csv").set_index("idx") # fetch rows my index

def load_split(name, images, labels):
    ids = np.loadtxt(f"data/processed/splits/{name}.txt", dtype=int)
    #batch stays in place, reorders to color x height x width, normalizes to 0-1
    x = images[ids].permute(0, 3, 1, 2).float() / 255
    #get 28 landmarks for each face 
    y = torch.tensor(labels.loc[ids, config.LANDMARK_COLS].to_numpy(), dtype=torch.float32)
    # mark existing landmarks 
    visible = ~torch.isnan(y)
    #replace NA with 0s
    y = torch.nan_to_num(y, nan=0.0)
    return TensorDataset(x, y, visible)

train_loader = DataLoader(load_split(f"train_{train_size}", images, labels), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(load_split("val", images, labels), batch_size=batch_size)

# mse loss function that ignores missing landmarks
def masked_mse(pred, target, visible):
    return ((pred - target) ** 2)[visible].mean()

model = _11_model.CNN(n_outputs=28)

# can this be manipulated more 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_losses, val_losses = [], []
for epoch in range(n_epochs):
    model.train()
    train_total = 0.0
    for x, y, visible in train_loader:
        pred = model(x)
        loss = masked_mse(pred, y, visible)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_total += loss.item() * len(x) # add batch loss weighted by batch size to get total loss for epoch

    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for x, y, visible in val_loader:
            val_total += masked_mse(model(x), y, visible).item() * len(x)

    train_losses.append(train_total / len(train_loader.dataset))
    val_losses.append(val_total / len(val_loader.dataset))
    # easier read is print pixels instead ; take sqrt() and multiple by 64 to get pixel error
    train_px = train_losses[-1] ** 0.5 * 64
    val_px = val_losses[-1] ** 0.5 * 64
    print(f"epoch {epoch + 1}: train {train_px:.2f}  val {val_px:.2f}")

model.eval()
test_results = {}
for name in ["test_frontal", "test_not_frontal"]:
    loader = DataLoader(load_split(name, images, labels), batch_size=batch_size)
    total = 0.0
    with torch.no_grad():
        for x, y, visible in loader:
            total += masked_mse(model(x), y, visible).item() * len(x)
    test_px = (total / len(loader.dataset)) ** 0.5 * 64
    test_results[name] = test_px
    print(f"{name}: {test_px:.2f} px")

torch.save(model.state_dict(), f"{config.results_path}/model.pt") # trained weights, for the visualize file

# ------------- VISUALIZE ----------------
train_px_curve = [loss ** 0.5 * 64 for loss in train_losses]
val_px_curve = [loss ** 0.5 * 64 for loss in val_losses]

epochs = range(1, n_epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# left: train vs validation error per epoch
ax1.plot(epochs, train_px_curve, "o-", label="train", color="tab:blue")
ax1.plot(epochs, val_px_curve, "o-", label="validation", color="tab:pink")
ax1.set_xlabel("epoch")
ax1.set_ylabel("error (px)")
ax1.set_title("train vs validation")
ax1.legend()

# right: frontal vs non-frontal test error
ax2.bar(["frontal", "non-frontal"], list(test_results.values()), color=["tab:blue", "tab:pink"])
ax2.set_ylabel("error (px)")
ax2.set_title("test: frontal vs non-frontal")
fig.tight_layout()

#os.makedirs(config.results_path, exist_ok=True) # creates results/ if it doesn't exist
fig.savefig(f"{config.results_path}/04_results_{train_size}.png")

plt.show()

