# 01: Loading the Data
import os
import numpy as np
import pandas as pd
from PIL import Image # imaging library
import config

# True: only the first 50 + skip resizing images (quick check)
# False: all images (full run)
test_run = False
n_images = 50 if test_run else config.n_images

mask_size = 512
output_path = "data/processed"

# landmarks from masks
# l_ and r_ are the person's left and right, so l_ parts appear on the right side of the image
LANDMARK_NAMES = ["l_brow_out", "l_brow_in", "r_brow_in", "r_brow_out",
                  "l_eye_out", "l_eye_in", "r_eye_in", "r_eye_out",
                  "nose_bottom", "l_mouth", "r_mouth", "u_lip_top",
                  "l_lip_bottom", "chin"]

LANDMARK_COLS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")]

def read_mask(idx, part):
    path = f"{config.mask_path}/{idx // 2000}/{idx:05d}_{part}.png"
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L")) > 0

def load_part(idx, *parts):
    """Merged True/False mask of the given parts, or None if none are present."""
    merged = None
    for part in parts:
        mask = read_mask(idx, part)
        if mask is None:
            continue
        merged = mask if merged is None else merged | mask
    if merged is None or not merged.any():
        return None
    return merged

def leftmost_point(mask):
    x = np.flatnonzero(mask.any(axis=0))[0]
    y = np.flatnonzero(mask[:, x]).mean()
    return (x, y)

def rightmost_point(mask):
    x = np.flatnonzero(mask.any(axis=0))[-1]
    y = np.flatnonzero(mask[:, x]).mean()
    return (x, y)

def rows_near_column(mask, x, width):
    strip = mask[:, max(x - width, 0): x + width + 1]
    return np.flatnonzero(strip.any(axis=1))

def top_point_at(mask, x, width=2):
    rows = rows_near_column(mask, x, width)
    if len(rows) == 0:
        return None
    return (x, rows[0])

def bottom_point_at(mask, x, width=2):
    rows = rows_near_column(mask, x, width)
    if len(rows) == 0:
        return None
    return (x, rows[-1])

def nose_bottom_point(mask):
    bottom_row = np.flatnonzero(mask.any(axis=1))[-1]
    bottom_strip = mask[max(bottom_row - 3, 0): bottom_row + 1]
    x = np.flatnonzero(bottom_strip.any(axis=0)).mean()
    return (x, bottom_row)

def landmarks_for(idx):
    points = {name: None for name in LANDMARK_NAMES}

    l_brow = load_part(idx, "l_brow")
    if l_brow is not None:
        points["l_brow_in"] = leftmost_point(l_brow)
        points["l_brow_out"] = rightmost_point(l_brow)

    r_brow = load_part(idx, "r_brow")
    if r_brow is not None:
        points["r_brow_out"] = leftmost_point(r_brow)
        points["r_brow_in"] = rightmost_point(r_brow)

    l_eye = load_part(idx, "l_eye")
    if l_eye is not None:
        points["l_eye_in"] = leftmost_point(l_eye)
        points["l_eye_out"] = rightmost_point(l_eye)

    r_eye = load_part(idx, "r_eye")
    if r_eye is not None:
        points["r_eye_out"] = leftmost_point(r_eye)
        points["r_eye_in"] = rightmost_point(r_eye)

    nose = load_part(idx, "nose")
    if nose is not None:
        points["nose_bottom"] = nose_bottom_point(nose)

    center_x = None
    mouth = load_part(idx, "mouth", "u_lip", "l_lip")
    if mouth is not None:
        points["r_mouth"] = leftmost_point(mouth)
        points["l_mouth"] = rightmost_point(mouth)
        center_x = int(round((points["l_mouth"][0] + points["r_mouth"][0]) / 2))

    upper_lip = load_part(idx, "u_lip")
    if center_x is not None and upper_lip is not None:
        points["u_lip_top"] = top_point_at(upper_lip, center_x)

    lower_lip = load_part(idx, "l_lip")
    if center_x is not None and lower_lip is not None:
        points["l_lip_bottom"] = bottom_point_at(lower_lip, center_x)

    skin = load_part(idx, "skin")
    if skin is not None:
        if center_x is not None:
            chin_x = center_x
        else:
            chin_x = int(round(np.flatnonzero(skin.any(axis=0)).mean()))
        chin = bottom_point_at(skin, chin_x, width=3)
        # skin reaching the bottom edge means the chin is cropped out of the photo
        if chin is not None and chin[1] < mask_size - 1:
            points["chin"] = chin

    row = []
    for name in LANDMARK_NAMES:
        point = points[name]
        row += [np.nan, np.nan] if point is None else list(point)
    return np.array(row, dtype=float) / mask_size

def read_anno(path):
    """Read the attribute or pose file: a count line, a header line, then one row per image."""
    with open(path) as f:
        f.readline()
        columns = f.readline().split()
    table = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None, names=["file"] + columns)
    table["idx"] = table["file"].str.replace(".jpg", "", regex=False).astype(int)
    return table.drop(columns="file")

def build_labels():
    rows = [landmarks_for(idx) for idx in range(n_images)]
    labels = pd.DataFrame(rows, columns=LANDMARK_COLS)
    labels["idx"] = range(n_images)

    attrs = read_anno(config.attr_path)
    attrs = attrs[["idx", "Eyeglasses", "Wearing_Hat"]].copy()
    attrs[["Eyeglasses", "Wearing_Hat"]] = (attrs[["Eyeglasses", "Wearing_Hat"]] == 1).astype(int)

    pose = read_anno(config.pose_path)
    pose = pose.rename(columns={"Raw": "Roll"})

    labels = labels.merge(attrs, on="idx").merge(pose, on="idx")
    labels.to_csv(f"{output_path}/labels.csv", index=False)
    print(labels.shape)
    print(labels[LANDMARK_COLS[::2]].isna().sum())

def build_images():
    size = config.image_size
    images = np.zeros((config.n_images, size, size, 3), dtype=np.uint8)
    for idx in range(config.n_images):
        img = Image.open(f"{config.img_path}/{idx}.jpg").convert("RGB")
        images[idx] = np.array(img.resize((size, size), Image.BILINEAR))
    np.save(f"{output_path}/images.npy", images)
    print(images.shape)

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    build_labels()
    if not test_run:
        build_images()