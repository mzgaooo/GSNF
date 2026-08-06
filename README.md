# GSNF

This is a PyTorch implementation of the paper: [One-Step Graph-Structured Neural Flows for Irregular Multivariate Time Series Classification ](https://arxiv.org/abs/2605.10179)published in ICML2026.

![image-20260806214558250](..\GSNF\model\model.png)

## Requirements

The model is implemented using Python 3.10 with the following dependencies:

```text
pytorch==1.12.1
cudatoolkit==11.3
numpy
scikit-learn
scipy
```

## Environment

```bash
conda env create -f environment.yml
conda activate gsnf
```

## Training

```bash
python train.py \
  --data-root /path/to/prepared_splits \
  --device cuda \
  --seed 0
```

