# Iterative-Cycle-consistent-Semi-supervised-Learning-for-fibroglandular-tissue-segmentation

## Paper:
Please see: Breast Fibroglandular Tissue Segmentation for Automated BPE Quantification with Iterative Cycle-consistent Semi-supervised Learning

## Introduction:
This is the PyTorch implementation for fibroglandular tissue segmentation.

## Requirements:
* python 3.10
* pytorch 1.12.1
* numpy 1.23.3
* tensorboard 2.10.1
* simpleitk 2.1.1.1
* scipy 1.9.1

## Setup

### Dataset
* The public dataset can be downloaded via https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903.
* To train the segmentation models, you need to organize the data in the following format:

```
./data
├─train.txt
├─test.txt
├─w_label
      ├─Breast_MRI_002
            ├─Breast.nii.gz
            ├─P0.nii.gz
            ├─P1_reg.nii.gz
            └─Tissue_gt.nii.gz

      ├─Breast_MRI_006
      ├─Breast_MRI_018
      ...

└─wo_label
      ├─Breast_MRI_001
            ├─Breast.nii.gz
            ├─P0.nii.gz
            └─P1_reg.nii.gz

      ├─Breast_MRI_003
      ├─Breast_MRI_004
      ...
```
* The format of the train.txt / test.txt is as follow：
```
./data/train.txt
├─w-Breast_MRI_002
├─w-Breast_MRI_006
├─w-Breast_MRI_018
...
├─wo-Breast_MRI_001
├─wo-Braest_MRI_003
...
```

### Whole Breast Segmentation Model
* The whole breast segmentation process is required to locate the breast ROI first.
* Partial images and whole breast annotations are available at: https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation

### Tumor Segmentation Model
* The tumor segmentation process is required to remove tumor enhancement for accurate BPE (Background Parenchymal Enhancement) quantification ratio.
* A well-designed tumor segmentation assistant is available at: https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation

## Citation
If you find the code useful, please consider citing the following papers:
* Zhang et al., Breast Fibroglandular Tissue Segmentation for Automated BPE Quantification with Iterative Cycle-consistent Semi-supervised Learning, IEEE Transactions on Medical Imaging (2023)
* Zhang et al., A robust and efficient AI assistant for breast tumor segmentation from DCE-MRI via a spatial-temporal framework, Patterns (2023), https://doi.org/10.1016/j.patter.2023.100826
* Zhang et al., Recent advancements in artificial intelligence for breast cancer: Image augmentation, segmentation, diagnosis, and prognosis approaches, Seminars in Cancer Biology (2023), https://doi.org/10.1016/j.semcancer.2023.09.001








