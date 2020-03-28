# CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement

# [In Construction]

Ho Kei Cheng, Jihoon Chung, Yu-Wing Tai, Chi-Keung Tang

[[Paper]]()

[[Supplementary Information (Comparisons with DenseCRF included!)]]()

[[Supplementary image results]]()

## Introduction

CascadePSP is a deep learning model for high-resolution segmentation refinement.
This repository contains our PyTorch implementation with both training and testing functionalities. We also provide the annotated UHD dataset **BIG** and the pretrained model.

Here are some refinement results on high-resolution images.
![teaser](docs/images/teaser.jpg)

## Network Overview

### Global Step & Local Step

| Global Step | Local Step |
|:-:|:-:|
| ![Global Step](docs/images/global.jpg) | ![Local Step](docs/images/local.jpg) |


### Refinement Module

![Refinement Module](docs/images/rm.png)

## Table of Contents

Running:

- [Installation](docs/installation.md)
- [Training](docs/training.md)
- [Testing on Semantic Segmentation](docs/testing_segmentation.md)
- [Testing on Scene Parsing](docs/testing_scene_parsing.md)

Downloads:

- [Pretrained Models](docs/models.md)
- [BIG Dataset and Relabeled PASCAL VOC 2012](docs/dataset.md)

## Credit

PSPNet implementation: https://github.com/Lextal/pspnet-pytorch

SyncBN implementation: https://github.com/vacancy/Synchronized-BatchNorm-PyTorch

If you find our work useful in your research, please cite the following:

```
@inproceedings{CascadePSP2020,
  title={CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement},
  author={Cheng, Ho Kei and Chung, Jihoon and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2020}
}
```
