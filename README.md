# Facial Feature Detection Nueral Network Model

Doing segmentation classification - label every pixel for each part of the face to identify (finding locations of facial features)

Goal by Nov 10: underfit the model (scoring poorly on train + test) --> improve from there
Visual - Plot training loss and validation loss per epoch

Data: CelebA HQ - 30k images 3 labels: pixel masks of facial parts, 40 yes/no attributes, and head angles

    1. CelebA-HQ-img - 30k jpeg face images - 1024×1024 pixels with 3 color channels 
        - already cropped / centered
        - * NEED TO SHRINK * 
    2. CelebAMask-HQ-mask-anno
        - 15 subfolders - 2k images
        - 512 x 512 RGB
        - * NEED TO CONVERT TO GRAYSCALE * 
        - 18 parts: skin, nose, eye_g (eyeglasses), l_eye, r_eye, l_brow, r_brow, l_ear, r_ear, mouth, u_lip (upper lip), l_lip (lower lip), hair, hat, ear_r (earring), neck_l (necklace), neck, cloth
        

