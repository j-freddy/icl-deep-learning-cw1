import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from const import IMAGENET_MEAN, IMAGENET_STD, INPUT_DTYPE, SEED
from resnet import MyResNet
from utils import check_accuracy, create_dataloaders, setup_device, split_dataset, view_samples

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_load_path",
        type=str,
        help="Load existing model. (default: model.pt)",
        default=None,
    )

    parser.add_argument(
        "--analysis",
        action=argparse.BooleanOptionalAction,
        help="View confusion matrix and sample of incorrect predictions.",
        default=False,
    )

    return parser.parse_args()

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

    # Load existing model
    model = MyResNet()
    model.load_state_dict(
        torch.load(flags.model_load_path, map_location=device)
    )

    # Report accuracy
    print("Checking accuracy...")
    print("Note: Ignore training and validation accuracy if model was not trained on this particular training set.")

    # Check train accuracy to detect overfitting
    
    # The training set should be the same as train.py as it is split in the same
    # way and seeded the same
    check_accuracy(loader_train, model, device, "train", analysis=flags.analysis)
    check_accuracy(loader_val, model, device, "validation", analysis=flags.analysis)
    check_accuracy(loader_test, model, device, "test", analysis=flags.analysis)

if __name__=="__main__":
    flags = parse_args()
    main(flags)
