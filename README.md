# Deep Learning CW1

Part 2 of the coursework has been split into multiple `.py` files instead of
writing code directly in the notebook. See [Navigation](#navigation) for
details.

## Navigation

- **train.py** - The main entry point for training. Usage for reproducing all
  results is provided as part of the answer to Q2.2. Alternatively, see
  [Usage](#usage).
- **test.py** - The main entry point for testing. Usage for reproducing all
  results is provided as part of the answer to Q2.2. Alternatively, see
  [Usage](#usage).
- **resnet.py** - Contains the ResNet model. This is mostly provided skeleton
  code, but I modified it to match the original paper. See Q2.2 for details.
- **augmented_dataset.py** - Contains the augmentation pipeline which is used on
  the training set.
- **normalised_dataset.py** - Contains the transformation pipeline provided in
  the skeleton code. I used a Torch Dataset, as this allows me to apply the
  pipeline to the validation and test set only.
- **utils.py** - Contains utility functions. Most of this is copied from the
  skeleton code, such as `plot_model_features`. Some have been modified, such as `split_dataset` which takes in a split parameter.

## Usage

Training the final model (50 epochs) takes several hours on a non-GPU machine. I
suggest using a GPU instead. See `scripts/slurm-train-final.sh`. I've included
the script below.

```sh
python -m train --model_save_path model_final.pt --lr 1e-3 --weight_decay 1e-4 --batch_size 32 --epochs 50
```

### Train

A lot of these parameters are for the purpose of hyperparameter tuning.

To print the train accuracy, I suggest using the `test.py` entry point instead,
since that prints the true accuracy. The train accuracy printed in `train.py` is
w.r.t. the augmented dataset, which may not reflect the true accuracy (it is
usually significantly lower).

```sh
usage: train.py [-h] [--epochs EPOCHS] [--model_save_path MODEL_SAVE_PATH] [--lr LR] [--weight_decay WEIGHT_DECAY] [--batch_size BATCH_SIZE] [--aug_crop_scale AUG_CROP_SCALE] 
                [--aug_color_bcs AUG_COLOR_BCS] [--aug_color_hue AUG_COLOR_HUE] [--aug_jitter_p AUG_JITTER_P] [--view_samples | --no-view_samples] [--analysis | --no-analysis]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of training epochs. (default: 10)
  --model_save_path MODEL_SAVE_PATH
                        Save model to path. (default: model.pt)
  --lr LR               Learning rate. (default: 0.001)
  --weight_decay WEIGHT_DECAY
                        Weight decay. (default: 0.0001)
  --batch_size BATCH_SIZE
                        Batch size. (default: 128)
  --aug_crop_scale AUG_CROP_SCALE
  --aug_color_bcs AUG_COLOR_BCS
  --aug_color_hue AUG_COLOR_HUE
  --aug_jitter_p AUG_JITTER_P
  --view_samples, --no-view_samples
                        View sample images. (default: False)
  --analysis, --no-analysis
                        View confusion matrix and sample of incorrect predictions. (default: False)
```

### Test

```sh
usage: test.py [-h] [--model_load_path MODEL_LOAD_PATH] [--analysis | --no-analysis]

options:
  -h, --help            show this help message and exit
  --model_load_path MODEL_LOAD_PATH
                        Load existing model. (default: model.pt)
  --analysis, --no-analysis
                        View confusion matrix and sample of incorrect predictions. (default: False)
```
