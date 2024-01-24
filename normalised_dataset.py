from torchvision import transforms
from torch.utils.data import TensorDataset as Dataset

from const import IMAGENET_MEAN, IMAGENET_STD

class NormalisedDataset(Dataset):
  def __init__(self, data):
    self.data = data
    self.transform = transforms.Compose(
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

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    input, label = self.data[idx]
    return self.transform(input), label
