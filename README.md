# Deep Learning CW1

## Usage

TODO

```sh
python -m train --epochs 1 --model_save_path to_delete.pt
python -m test --model_load_path model.pt
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

### Changes to Skeleton Code

- overall refactor for maintainability (e.g. less local variables, modularise
  code) and allow flexibility (e.g. `split_dataset` function with split factor
  parameter)
- no shuffling in test DataLoader
