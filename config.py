# paths 
data_path = "data/CelebAMask-HQ"
img_path = f"{data_path}/CelebA-HQ-img"
mask_path = f"{data_path}/CelebAMask-HQ-mask-anno"
attr_path = f"{data_path}/CelebAMask-HQ-attribute-anno.txt"
pose_path = f"{data_path}/CelebAMask-HQ-pose-anno.txt"
results_path = "results"

#image sizes 
n_images = 30000
image_size = 64 # 1024 too large; use 128 if results are too poor
seed = 2026

# landmarks from masks
# l_ and r_ are the person's left and right, so l_ parts appear on the right side of the image
LANDMARK_NAMES = ["l_brow_out", "l_brow_in", "r_brow_in", "r_brow_out",
                  "l_eye_out", "l_eye_in", "r_eye_in", "r_eye_out",
                  "nose_bottom", "l_mouth", "r_mouth", "u_lip_top",
                  "l_lip_bottom", "chin"]

LANDMARK_COLS = [f"{name}_{axis}" for name in LANDMARK_NAMES for axis in ("x", "y")]
