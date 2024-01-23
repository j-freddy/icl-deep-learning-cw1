# Deep Learning CW1

## Usage

TODO

```sh
python -m main --help
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

# Validation (last epoch)
Got 727 / 1998 correct of val set (36.39)

# Test
Got 691 / 2000 correct of val set (34.55)
```

### Changes to Skeleton Code

- overall refactor for maintainability (e.g. less local variables, modularise
  code) and allow flexibility (e.g. `split_dataset` function with split factor
  parameter)
- no shuffling in test DataLoader
