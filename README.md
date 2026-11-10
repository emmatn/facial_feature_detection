# Facial Landmark/Feature Detection Nueral Network Model

Question: Can a dynamic (early-exit) neural network detect facial landmarks as accurately as a standard network while using less computation by spending more effort only on difficult faces?

    - Changing how much of the network each input uses - early exits and SkipNet
    - network is fixed after training but at prediction time an easy face leaves early while a hard face goes deeper

Model: 
- Landmark Coordinate Regression
- For each face, it predicts 28 numbers (x,y) coordinates of the 14 facial landmarks.
- training on frontal images, testing on frontal vs nonfrontal, seeing if it does worst on "harder"/more angled images 

Data: CelebA HQ - 30k images 3 labels: pixel masks of facial parts, 40 yes/no attributes, and head angles

    1. CelebA-HQ-img - 30k jpeg face images - 1024×1024 pixels with 3 color channels 
        - already cropped / centered
        - * NEED TO SHRINK * 
    2. CelebAMask-HQ-mask-anno
        - 15 subfolders - 2k images
        - 512 x 512 RGB
        - 18 parts: skin, nose, eye_g (eyeglasses), l_eye, r_eye, l_brow, r_brow, l_ear, r_ear, mouth, u_lip (upper lip), l_lip (lower lip), hair, hat, ear_r (earring), neck_l (necklace), neck, cloth
        - Note: Humans drew on these masks so the labels stay accurate on hard faces

Steps: 
- Derive landmark points from mask annotations (14 landmarks)
- upload data, resizing, grayscale 
- split data - train, validate, test
- write Dataset class
- Define simple model 

Notes: 
- Masks are raw material drawn by humans (used once when loading, then for occlusion)
- Landmarks are compact, finished labels (fast, easy to measure error)
- 14 landmarks from masks: 
        Eyebrow ends and eye corners: leftmost and rightmost pixels of each brow and eye mask.
        Mouth corners: leftmost and rightmost pixels of the mouth and both lips combined into one shape.
        Nose bottom: the lowest pixels of the nose mask.
        Top of upper lip: look only at the column halfway between the mouth corners and take the highest u_lip pixel there.
        Bottom of lower lip: the lowest l_lip pixel in that same center column.
        Chin: the lowest skin pixel in the face's center column.

