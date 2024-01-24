from torchvision import transforms
from torch.utils.data import TensorDataset as Dataset

from const import IMAGENET_MEAN, IMAGENET_STD

class AugmentedDataset(Dataset):
  def __init__(self, data):
    self.data = data
    self.transform = transforms.Compose(
        [
            # Original pipeline converts to Tensor, so convert back to PIL
            # before performing image transforms
          
            # Note: Original pipeline already resizes to 256x256 and center
            # crops

            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
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
