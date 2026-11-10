as of 11/9 

Baseline numbers for loss comparison: 
        train_5000: 0.94 px
        val: 0.95 px
        test_frontal: 0.95 px
        test_not_frontal: 1.34 px

Results: 
1. 01_results_5000 - model with 2 layers, adaptive pooling to 8x8
    - training + validation error stay relatively the same to each other 

2. 02_results_5000 - model with 2 layers, no adaptive but using dropout of .3
    - training is consistently higher, but still looks like its dropping at epoch 20. Validation error fluctuates a good amount
    - both losses lower than the updated 02 
3. 02_results_5000 - same as 2, but lower learning rate 
    - both losses are consistently decreasing until epoch 13 (validation jumps up and down)
    - would need to increase number of epochs since lr lowered 

4. 03_results_5000 - changed conv1 layer output 32, instead of 16
    - adaptive pooling to 8x8 
    - seeing if it helps the model learn features better 
    - both losses consistently decreasing, very close to each other 
  * - next steps is increasing number of epochs as loss is still dropping for both *
    

Currently still underfitting: 
Functions to look at: 
- size of training data
- lr in Adam optimizer 
- number of epochs 
- need to fix what pictures count as frontal images (based on the angle cutoff)

Main point: my best model beats the "always guess the average face" baseline, but only by about 10%. Training and validation errors are nearly equal, so it's underfitting.

Findings

Data: 14 landmarks derived from the CelebAMask-HQ masks for 30,000 faces. Training and validation use frontal faces only; testing covers both frontal and non-frontal faces.
Baseline: CelebA-HQ faces are pre-aligned, so guessing the average face already scores 0.95 px on frontal and 1.34 px on non-frontal faces at 64×64.
Best model (2 conv layers, 5,000 training faces): about 0.86 px frontal and 1.18 px non-frontal.
What I tried: a single conv layer only matched the baseline. Removing pooling overfit. A second layer, more filters, dropout and a lower learning rate each helped a little.
Turned heads are harder: non-frontal error is about 35–40% higher. On strongly turned faces, predictions drift back toward the average frontal layout.
Next steps

Standardize the landmark targets so the model can't settle on the average face.
Make the network deeper: a third conv layer, or the early-exit ResNet.
Train on 128×128 images and the 10,000-face split.
Report normalized mean error, the error divided by the outer-eye distance. The baseline is about 4.9% on frontal and 7.2% on non-frontal faces.
Once a standard network clearly beats the baseline, compare the early-exit model against it on accuracy and computation.
