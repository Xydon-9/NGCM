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

# Setup

## Installation

### Clone this repository

To clone this repository and the code, run:

Bash

```
https://github.com/Xydon-9/NGCM.git
```

## Pre-Trained Models

We used the identical pre-trained Consistency Models provided by OpenAI as used in previous SOTA methods in the [CM repo](https://github.com/openai/consistency_models?tab=readme-ov-file#pre-trained-models).

To set the models used in the paper:

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

Both LSUN Bedroom and LSUN Cat validation sets used in the paper can be found [here](https://drive.google.com/drive/folders/1umSbW_91LTJuK11Il_pmleC4OPei7LAE?usp=sharing)(CM4IR and Cosign)

ImageNet can be found in: [[Google drive](https://drive.google.com/drive/folders/1cSCTaBtnL7OIKXT4SVME88Vtk4uDd_u4?usp=sharing)](DDNM)
