# paths 
data_path = "data/CelebAMask-HQ"
img_path = f"{data_path}/CelebA-HQ-img"
mask_path = f"{data_path}/CelebAMask-HQ-mask-anno"
results_path = "results"

#image sizes 
n_images = 30000
image_size = 64 # 1024 too large; use 128 if results are too poor
seed = 2026

# for segmentation 
parts = ["skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
         "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat",
         "ear_r", "neck_l", "neck", "cloth"] 