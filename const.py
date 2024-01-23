import torch

IMAGENET_MEAN = torch.Tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.Tensor([0.229, 0.224, 0.225])
INPUT_DTYPE = torch.float32
SEED = 90
USE_GPU = True
