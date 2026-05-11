# Null-space Guided Consistency Models for Few-Step Image Restoration

# Introduction

This repository contains the code release for Null-space Guided Consistency Models for Few-Step Image Restoration [NGCM].

Consistency models enable high-quality restoration with only a few steps, but they often suffer from instability caused by conflicts between data-consistency enforcement and stochastic exploration, as well as sensitivity to initialization errors. NGCM stabilizes and enhances few-step zero-shot image restoration through two synergistic innovations:

*

Null-space Noise Projection (NNP): Projects injected noise into the null-space of the degradation operator, ensuring stochastic exploration preserves data consistency by construction.

*

Denoising State Refinement (DSR): Refines the initial estimate using a single additional NFE (Neural Function Evaluation) to establish a more accurate and stable restoration trajectory.

## Supported degradations

1. Super-Resolution (Bicubic, x2 and x4)

2. Gaussian Deblurring

3. Inpainting (e.g., random binary masks with 80% missing pixels)


# Qualitative Results

<img width="1168" height="843" alt="Snipaste_2026-05-07_11-50-28" src="https://github.com/user-attachments/assets/9e2449f0-95c3-45f7-9b98-0df98608c1f8" />
<img width="1146" height="733" alt="Snipaste_2026-05-07_11-50-13" src="https://github.com/user-attachments/assets/2bf8f0a6-7933-410c-baf4-4f71ced35009" />
<img width="1272" height="821" alt="Snipaste_2026-05-07_11-49-42" src="https://github.com/user-attachments/assets/84749db8-d1d6-457b-9110-4d7f13c6aa7e" />

# Hyperparameters Setting
<img width="1197" height="410" alt="image" src="https://github.com/user-attachments/assets/30b6754b-56c7-488b-86ab-17261a0ab9e9" />

# Quick Start

#---- Noise level 0.025 ----
Super-Resolution Bicubic x4
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg sr_bicubic --deg_scale 4 \
  --sigma_y 0.025 -i NGCM_lsun_bedroom_sr_bicubic_sigma_y_0.025 --iN 400 --gamma 0.7 \
  --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt --deltas "0,0.3,0.1,0"  --use_dsr 1 --use_nnp 1  --T_sampling 4 

Gaussian Deblurring
python main.py --config lsun_bedroom_256.yml --path_y lsun_bedroom --deg deblur_gauss \
  --sigma_y 0.025 -i NGCM_lsun_bedroom_deblur_gauss_sigma_y_0.025 --iN 90 --gamma 0.02 \
  --zeta 3 --model_ckpt lsun_bedroom/cd_bedroom256_lpips.pt --deltas "0,0.1,0,0" --eta 0 --use_dsr 1 --use_nnp 1  --T_sampling 4 




# Setup

## Installation

### Clone this repository

To clone this repository and the code, run:

Bash

```
git clone https://github.com/Xydon-9/NGCM.git
```

## Pre-Trained Models

We used the identical pre-trained Consistency Models provided by OpenAI as used in previous SOTA methods in the [CM repo](https://github.com/openai/consistency_models?tab=readme-ov-file#pre-trained-models).

To set the models used in the paper

### LSUN Bedroom

The LSUN Bedroom 256x256 model checkpoint can be found [here](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt).
Download it and place it in `NGCM/exp/logs/lsun_bedroom/`.

### LSUN Cat

The LSUN Cat 256x256 model checkpoint can be found [here](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_lpips.pt).
Download it and place it in `NGCM/exp/logs/lsun_cat/`.

### ImageNet

The ImageNet 64x64 model checkpoint can be found [here](https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_lpips.pt).
Download it and place it in `NGCM/exp/logs/imagenet/`.

## Full Datasets

&#x20;

The datasets used in the paper are LSUN bedroom, LSUN cat and ImageNet.

Both LSUN Bedroom and LSUN Cat validation sets used in the paper can be found [here](https://drive.google.com/drive/folders/1umSbW_91LTJuK11Il_pmleC4OPei7LAE?usp=sharing)(Same LSUN setting as CoSIGN and CM4IR)

ImageNet can be found in: [[Google drive](https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4?usp=sharing)](DDNM)





This implementation is inspired by https://github.com/openai/consistency_models and https://github.com/bahjat-kawar/ddrm and https://github.com/tirer-lab/CM4IR.
