import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Conv2d, MaxPool2d
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from const import IMAGENET_MEAN, IMAGENET_STD, SEED
from utils import create_dataloaders, split_dataset, view_samples

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--view_samples",
        action=argparse.BooleanOptionalAction,
        help="View sample images. Default: False",
        default=False,
    )

    return parser.parse_args()


if __name__=="__main__":
    flags = parse_args()
    
    # Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN.tolist(),
                IMAGENET_STD.tolist(),
            ),
        ]
    )

    train_path = "NaturalImageNetTrain"
    test_path = "NaturalImageNetTest"

    # Create and split datasets
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    train_set, val_set = split_dataset(train_dataset, val_split=0.1)
    
    print(f"Train set: {len(train_set)}")
    print(f"Val set: {len(val_set)}")
    print(f"Test set: {len(test_dataset)}")
    
    # Create DataLoaders
    loader_train, loader_val, loader_test = create_dataloaders(
        train_set,
        val_set,
        test_dataset,
        batch_size=128,
    )
    
    # View sample images
    if flags.view_samples:
        view_samples(loader_val)
