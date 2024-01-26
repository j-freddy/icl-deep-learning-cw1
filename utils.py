import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from const import IMAGENET_MEAN, IMAGENET_STD, INPUT_DTYPE, USE_GPU

def setup_device():
    if USE_GPU and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

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

def confusion(preds, y):
    labels = ["African Elephant", "Kingfisher", "Deer","Brown Bear", "Chameleon", "Dragonfly",
        "Giant Panda", "Gorilla", "Hawk", "King Penguin", "Koala", "Ladybug", "Lion",
        "Meerkat", "Orangutan", "Peacock", "Red Fox", "Snail", "Tiger", "White Rhino"]
    # Plotting the confusion matrix
    cm = confusion_matrix(y.cpu().numpy(), preds.cpu().numpy(), normalize="true")
    fig, ax= plt.subplots(1, 1, figsize=(15,10))
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel("Predicted labels");ax.set_ylabel("True labels"); 
    ax.set_title("Confusion Matrix");
    ax.xaxis.set_ticklabels(labels, rotation = 70); ax.yaxis.set_ticklabels(labels, rotation=0);
    plt.show()

def incorrect_preds(preds, y, test_img):
    labels = ["African Elephant", "Kingfisher", "Deer","Brown Bear", "Chameleon", "Dragonfly",
        "Giant Panda", "Gorilla", "Hawk", "King Penguin", "Koala", "Ladybug", "Lion",
        "Meerkat", "Orangutan", "Peacock", "Red Fox", "Snail", "Tiger", "White Rhino"]
    # lets see a sample of the images which were classified incorrectly!
    correct = (preds == y).float()
    test_labels_check = correct.cpu().numpy()
    incorrect_indexes = np.where(test_labels_check == 0)

    test_img = test_img.cpu()
    samples = make_grid(denorm(test_img[incorrect_indexes][:9]), nrow=3,
                        padding=2, normalize=False, value_range=None, 
                        scale_each=False, pad_value=0)
    plt.figure(figsize = (20,10))
    plt.title("Incorrectly Classified Instances")
    show(samples)
    labels = np.asarray(labels)
    print("Predicted label",labels[preds[incorrect_indexes].cpu().numpy()[:9]])
    print("True label", labels[y[incorrect_indexes].cpu().numpy()[:9]])
    print("Corresponding images are shown below")

def check_accuracy(loader, model, device, label, analysis=False):
    # function for test accuracy on validation and test set
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=INPUT_DTYPE)  # move to device
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            if t == 0 and analysis:
              stack_labels = y
              stack_predicts = preds
            elif analysis:
              stack_labels = torch.cat([stack_labels, y], 0)
              stack_predicts = torch.cat([stack_predicts, preds], 0)
        acc = float(num_correct) / num_samples
        
        print(f"Got %d / %d correct of {label} set (%.2f)" % (num_correct, num_samples, 100 * acc))
        
        if analysis:
          print("check acc", type(stack_predicts), type(stack_labels))
          confusion(stack_predicts, stack_labels)
          incorrect_preds(preds, y, x)
        return float(acc)
