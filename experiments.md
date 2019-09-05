# **Experiment Results**

Final loss: Final value of loss graph smoothed with 0.9.  
Validation metrics are from the best epoch.

# Baseline

GPU1 machine   
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-06-16-31-L1-contrast_1_half_2_half/`

## Configuration
- Dual-frame U-Net
- Adam, 1e-2
- Batch size: 12
- `loss_l1`
- `Train_0704_1_0.5_2_0.5.mat`
- 25 epochs

## Results
- final `loss_l1`: 0.1013
- Best Epoch: 20
- `HFEN: 41.63 54.62 55.87 51.20 53.12`
- `RMSE: 45.74 58.21 58.88 56.29 56.29`
- `SSIM: 94.68 93.00 91.07 92.26 92.52 * 1e-2`
  
---
---

# HFEN Loss 1

Titan Machine
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-05-23-35-hfen/`
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-15-16-17-hfen-long/`
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-18-11-50-hfen-long-cont/`

## Configuration
- Dual-frame U-Net
- Adam, 5e-3
- Batch size: 12
- `1 * loss_l1 + 0.01 * loss_hfen`
- `Train_0704_1_0.5_2_0.5.mat`
- 20 epochs

## Results
- Final `loss_l1`: 0.1115
- Final `loss_hfen`: 16.11
- Best epoch: 12 > 17, 65, 90
- `HFEN: 39.59 50.88 54.46 48.83 50.86`
- `RMSE: 44.41 55.66 58.29 54.19 54.79`
- `SSIM: 94.62 93.10 90.95 92.11 92.85 * 1e-2`


# HFEN Loss 2

Titan Machine  
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-06-01-19-half-hfen/`

## Configuration
- Dual-frame U-Net
- Adam, 5e-3
- Batch size: 12
- `1 * loss_l1 + 0.005 * loss_hfen`
- `Train_0704_1_0.5_2_0.5.mat`
- 25 epochs

## Results
- Final `loss_l1`: 0.08926
- Final `loss_hfen`: 19.99
- Best epoch: 19 > 23
- `HFEN: 42.39 53.62 56.95 53.32 55.60`
- `RMSE: 46.21 57.51 59.23 57.29 58.06`
- `SSIM: 94.61 93.38 91.30 92.07 92.45 * 1e-2`

---
---

# Sub-pixel Convolution

Titan Machine  
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet-pixelshuffle/07-06-18-14-L1_contrast_1_half_2_half/`

## Configuration
- Dual-frame U-Net with upscaling by sub-pixel convolution
- Adam, 5e-3
- Batch size: 8
- `loss_l1`
- `Train_0704_1_0.5_2_0.5.mat`
- 25 epochs

## Results
- Final `loss_l1`:
- Best epoch: 18 > 23
- `HFEN: 44.60 57.64 60.75 56.72 57.46`
- `RMSE: 48.56 60.82 62.66 60.44 59.71`
- `SSIM: 94.50 92.97 90.82 91.81 92.34 * 1e-2`

---
---

# Linearity Loss

Titan Machine  
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-07-19-31-linearity-loss/`

## Configuration
- Dual-frame U-Net
- Adam, 5e-3
- Batch size: 6
- `1 * loss_l1 + 0.1 * loss_linearity`
- `Train_0702_Patient1to7_Aug5.mat`
- 25 epochs
- Alpha: 1.5 ~ 2.5 uniform

## Results
- Final `loss_l1`: 0.1851
- Final `loss_linearity`: 0.02178
- Best epoch: N/A
- Terrible

---
---

# Loss transfer 15
Titan Machine
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-20-22-01-change-to-L2-after-15/`

## Configuration
- Dual-frame U-Net
- Adam, 5e-3
- Batch size: 12
- `1 * loss_l1 + 0.01 * loss_hfen` -> `1 * loss_l2 + 0.01 * loss_hfen`
- 15 epochs -> 35 epochs

## Results
- Best epoch: 40 > 45
- `HFEN: 43.55 53.24 56.54 52.46 54.05`
- `PSNR: 41.85 39.46 38.10 39.43 40.29`
- `RMSE: 6.051 7.038 7.193 6.938 7.032 * 1e-3`
- `SSIM: 94.46 93.31 91.52 92.11 92.67 * 1e-2`

# Loss transfer 25

Titan Machine
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-20-21-57-change-to-L2-after-25/`

## Configuration
- Dual-frame U-Net
- Adam, 5e-3
- Batch size: 12
- `1 * loss_l1 + 0.01 * loss_hfen` -> `1 * loss_l2 + 0.01 * loss_hfen`
- 25 epochs -> 25 epochs

## Results
- Best epoch: 45
- `HFEN: 43.35 54.31 56.54 51.85 53.73`
- `PSNR: 42.28 39.19 38.49 39.07 40.40`
- `RMSE: 6.028 7.083 7.232 6.926 6.980 * 1e-3`
- `SSIM: 94.57 93.37 91.59 92.33 92.77 * 1e-2`

---
---

# Only artificial data 1e-2

Titan Machine
`~/jaywonchung/QSMFirstOrDeath/Checkpoints/dual-frame-unet/07-13-01-48-only-artificial-data/`

## Configuration
- Dual-frame U-Net
- Adam, 1e-2
- Batch size: 12
- `loss_l1`
- 45 epochs

## Results
- Normal validation scores suck (since they are on real data). Need to inspect with eye.


# Only artificial data 5e-3

Titan Machine


---
---