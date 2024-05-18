# DTS

## Overview
This is the code of our Neural Networks submission.

## Environment Setup

To replicate our environment and ensure seamless execution of the code, please rebuild the environment using the provided `environment.yaml` file. You can do this with the following Conda command:

```bash
conda env create -f environment.yaml
```

## Usage

### CIFAR-100
#### Pretrain
```bash
python pretrain.py --dataset CIFAR100 --n_labels 5000 --n_unlabels 20000 --n_valid 10000 --n_class 50 --ratio 0.6 --warm_up 200
```
#### Run
CIFAR-100 with class mismatch ratio 0.3
```bash
python train.py --dataset CIFAR100 --n_labels 5000 --n_unlabels 20000 --n_valid 10000 --n_class 50 --ratio 0.3 --lm 0.5 --threshold 0.85 --Ctk_weight 0.25 --Ctu_weight 0.1 --socr_s2_weight 0.3
```

CIFAR-100 with class mismatch ratio 0.6
```bash
python train.py --dataset CIFAR100 --n_labels 5000 --n_unlabels 20000 --n_valid 10000 --n_class 50 --ratio 0.6 --lm 0.5 --threshold 0.85 --Ctk_weight 0.25 --Ctu_weight 0.1 --socr_s2_weight 0.3
```
