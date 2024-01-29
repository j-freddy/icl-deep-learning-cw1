# Deep Learning CW1

## Usage

TODO

```sh
# Train
python -m train --epochs 20 --model_save_path model.pt
# Test
python -m test --model_load_path model_final.pt
# Tensorboard
tensorboard --logdir=runs
```

## Notes

### ImageNet Baseline

```sh
# Runtime on Paperspace P4000
# 8 CPU | 30 GB RAM | 8 GB GPU
13m 55s
# Runtime on DoC SLURM
# MIG GPU
11m 38s
```

### TODO

1. hparam tuning of augmentation pipeline (save graphs only)
    - look into custom naming of runs
2. then hparam tuning of other params (save graphs only) (justification: to
   squeeze out last bit of performance)
