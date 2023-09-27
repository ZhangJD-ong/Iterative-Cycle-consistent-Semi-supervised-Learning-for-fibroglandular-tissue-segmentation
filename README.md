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
├─Guangdong
      ├─Guangdong_1
          ├─P0.nii.gz
          ├─P1.nii.gz
          ├─P2.nii.gz
          ├─P3.nii.gz
          ├─P4.nii.gz     
          └─P5.nii.gz
      ├─Guangdong_2
      ├─Guangdong_3
      ...
├─Guangdong_breast
      ├─Guangdong_1.nii.gz
      ├─Guangdong_2.nii.gz
      ├─Guangdong_2.nii.gz
      ...
├─Guangdong_gt
      ├─Guangdong_1.nii.gz
      ├─Guangdong_2.nii.gz
      ├─Guangdong_2.nii.gz
      ...         
└─Yunzhong
└─Yunzhong_breast
└─Yunzhong_gt
└─Ruijin
└─Ruijin_breast
└─Ruijin_gt
...
```
* The format of the train.txt / test.txt is as follow：
```
./data/train.txt
├─'Guangdong_1'
├─'Guangdong_2'
├─'Guangdong_3'
...
├─'Yunzhong_100'
├─'Yunzhong_101'
...
├─'Ruijin_1010'
...
```

### Tumor Segmentation Model
* The tumor segmentation process is required to remove tumor enhancement for accurate BPE (Background Parenchymal Enhancement) quantification ratio.
* A well-designed tumor segmentation assistant is available at: https://github.com/ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation

## Citation
If you find the code useful, please consider citing the following papers:
* Zhang et al., Breast Fibroglandular Tissue Segmentation for Automated BPE Quantification with Iterative Cycle-consistent Semi-supervised Learning, IEEE Transactions on Medical Imaging (2023)
* Zhang et al., A robust and efficient AI assistant for breast tumor segmentation from DCE-MRI via a spatial-temporal framework, Patterns (2023), https://doi.org/10.1016/j.patter.2023.100826
* Zhang et al., Recent advancements in artificial intelligence for breast cancer: Image augmentation, segmentation, diagnosis, and prognosis approaches, Seminars in Cancer Biology (2023), https://doi.org/10.1016/j.semcancer.2023.09.001








