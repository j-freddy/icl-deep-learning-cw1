import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from augmented_dataset import AugmentedDataset

from const import INPUT_DTYPE, SEED
from normalised_dataset import NormalisedDataset
from resnet import MyResNet
from utils import check_accuracy, create_dataloaders, setup_device, split_dataset, view_samples

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs. (default: 10)",
        default=10,
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        help="Save model to path. (default: model.pt)",
        default="model.pt",
    )

    parser.add_argument(
        "--view_samples",
        action=argparse.BooleanOptionalAction,
        help="View sample images.",
        default=False,
    )

    parser.add_argument(
        "--analysis",
        action=argparse.BooleanOptionalAction,
        help="View confusion matrix and sample of incorrect predictions.",
        default=False,
    )

    return parser.parse_args()

def train_part(
    model,
    optimizer,
    device,
    loader_train,
    loader_val,
    writer,
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
    - writer: A SummaryWriter object for Tensorboard
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
                curr_step = e * len(loader_train) + t
                
                # Log loss
                print("Epoch: %d, Iteration %d, loss = %.4f" % (e, t, loss.item()))
                writer.add_scalar("Loss/train over step", loss, curr_step)

        val_acc = check_accuracy(
            loader_val,
            model,
            device,
            label="validation",
            writer=writer,
        )
        
        writer.add_scalar("Accuracy/val over epoch", val_acc, e)

def main(flags):
    # Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = setup_device()

    train_path = "NaturalImageNetTrain"
    test_path = "NaturalImageNetTest"
    
    # Tensorboard
    writer = SummaryWriter()

    # Create and split datasets
    train_dataset = datasets.ImageFolder(train_path)
    test_dataset = datasets.ImageFolder(test_path)

    train_set, val_set = split_dataset(train_dataset, val_split=0.1)
    
    # Augment the training set
    train_set = AugmentedDataset(train_set)
    # Normalise the validation and test sets
    val_set = NormalisedDataset(val_set)
    test_dataset = NormalisedDataset(test_dataset)

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
        # View augmented samples
        view_samples(loader_train)
        # View non-augmented samples
        view_samples(loader_val)
        # TODO Remove
        assert False

    # Train the network
    model = MyResNet()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))

    train_part(
        model,
        optimizer,
        device,
        loader_train,
        loader_val,
        writer,
        epochs=flags.epochs,
    )

    # Report accuracy
    print("Checking accuracy...")
    
    # Note: This reports the training accuracy of the randomly augmented set. To
    # report the accuracy of the original training set, use test.py.
    check_accuracy(loader_train, model, device, "train", analysis=flags.analysis)
    check_accuracy(loader_val, model, device, "validation", analysis=flags.analysis)
    check_accuracy(loader_test, model, device, "test", analysis=flags.analysis)

    # Save the model
    torch.save(model.state_dict(), flags.model_save_path)

if __name__=="__main__":
    flags = parse_args()
    main(flags)
