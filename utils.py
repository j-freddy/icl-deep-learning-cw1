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

from const import IMAGENET_MEAN, IMAGENET_STD

def split_dataset(train_set, val_split=0.1):
    n = len(train_set)
    n_val = int(n * val_split)

    return torch.utils.data.random_split(
        train_set,
        [n - n_val, n_val],
    )

def create_dataloaders(train_set, val_set, test_set, batch_size=128):
    loader_train = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    loader_val = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    loader_test = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return loader_train, loader_val, loader_test

def denorm(x):
    """
    Function to reverse the normalization so that we can visualise the outputs
    """

    unnormalize = transforms.Normalize(
        (-IMAGENET_MEAN / IMAGENET_STD).tolist(),
        (1.0 / IMAGENET_STD).tolist(),
    )
    
    x = unnormalize(x)
    x = x.view(x.size(0), 3, 256, 256)
    return x

def show(img):
    """
    Function to visualise tensors
    """

    if torch.cuda.is_available():
        img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).clip(0, 1))

def view_samples(loader):
    sample_inputs, _ = next(iter(loader))
    fixed_input = sample_inputs[:27, :, :, :]

    img = make_grid(
        denorm(fixed_input),
        nrow=9,
        padding=2,
        normalize=False,
        value_range=None,
        scale_each=False,
        pad_value=0,
    )
    
    plt.figure(figsize=(12, 6))
    plt.axis("off")
    show(img)
    plt.show()
