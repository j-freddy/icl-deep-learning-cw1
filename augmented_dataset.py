from torchvision import transforms
from torch.utils.data import TensorDataset as Dataset

from const import IMAGENET_MEAN, IMAGENET_STD

class AugmentedDataset(Dataset):
  def __init__(
    self,
    data,
    crop_scale=0.2,
    color_bcs=0.5,
    color_hue=0.1,
    jitter_p=0.75,
  ):
    self.data = data
    self.transform = transforms.Compose(
        [
            transforms.Resize(256),

            # Apply augmentations
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(256, scale=(crop_scale, 1.0)),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=color_bcs,
                        contrast=color_bcs,
                        saturation=color_bcs,
                        hue=color_hue,
                    )
                ],
                p=jitter_p,
            ),

            # Normalise
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
