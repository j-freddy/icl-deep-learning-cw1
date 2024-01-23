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

from const import IMAGENET_MEAN, IMAGENET_STD, INPUT_DTYPE, SEED
from resnet import MyResNet
from utils import check_accuracy, create_dataloaders, setup_device, split_dataset, view_samples

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--view_samples",
        action=argparse.BooleanOptionalAction,
        help="View sample images. Default: False",
        default=False,
    )

    parser.add_argument(
        "--analysis",
        action=argparse.BooleanOptionalAction,
        help="View confusion matrix and sample of incorrect predictions. Default: False",
        default=False,
    )

    return parser.parse_args()

def train_part(
    model,
    optimizer,
    device,
    loader_train,
    loader_val,
    epochs=1,
):
    """
    Train a model on NaturalImageNet using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - device: A PyTorch device object giving the device on which the model runs
        on, e.g. CUDA, GPU, CPU
    - loader_train: A DataLoader object for training data batches
    - loader_val: A DataLoader object for validation data batches
    - epochs: (Optional) A Python integer giving the number of epochs to train
        for
    
    Returns: Nothing, but prints model accuracies during training.
    """

    model = model.to(device=device)

    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            # Put model to training mode
            model.train()

            x = x.to(device=device, dtype=INPUT_DTYPE)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the
            # optimizer will update
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % 10 == 0:
                print("Epoch: %d, Iteration %d, loss = %.4f" % (e, t, loss.item()))

        check_accuracy(loader_val, model, device)

def main(flags):
    # Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = setup_device()
    
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

    # Train the network
    model = MyResNet()
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, weight_decay=1e-7) 

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))

    train_part(
        model,
        optimizer,
        device,
        loader_train,
        loader_val,
        epochs=10,
    )

    # Report train set accuracy
    check_accuracy(loader_train, model, device, analysis=flags.analysis)

    # Report test set accuracy
    check_accuracy(loader_val, model, device, analysis=flags.analysis)

    # Save the model
    torch.save(model.state_dict(), "model.pt")

if __name__=="__main__":
    flags = parse_args()
    main(flags)
