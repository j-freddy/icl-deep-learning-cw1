import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from const import SEED
from normalised_dataset import NormalisedDataset
from resnet import MyResNet
from utils import check_accuracy, create_dataloaders, setup_device, split_dataset

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

def network_performance(model_load_path):
    # Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = setup_device()
    
    test_path = "NaturalImageNetTest"
    
    # Create and normalise dataset
    test_dataset = datasets.ImageFolder(test_path)
    test_dataset = NormalisedDataset(test_dataset)
    
    loader_test = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
    )
    
    # Load existing model
    model = MyResNet()
    model.load_state_dict(
        torch.load(model_load_path, map_location=device)
    )
    
    # Report accuracy
    check_accuracy(loader_test, model, device, "test", analysis=True)

def test_out_of_distribution(model_load_path, loader):
    # Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = setup_device()
    
    # Load existing model
    model = MyResNet()
    model.load_state_dict(
        torch.load(model_load_path, map_location=device)
    )
    
    # Report accuracy
    check_accuracy(loader, model, device, "ood", analysis=True)

def main(flags):
    # Seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = setup_device()

    train_path = "NaturalImageNetTrain"
    test_path = "NaturalImageNetTest"

    # Create and split datasets
    train_dataset = datasets.ImageFolder(train_path)
    test_dataset = datasets.ImageFolder(test_path)
    
    train_set, val_set = split_dataset(train_dataset, val_split=0.1)
    
    # Normalise all sets
    train_set = NormalisedDataset(train_set)
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
