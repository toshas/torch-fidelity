import sys
from contextlib import redirect_stdout

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from torch_fidelity.helpers import vassert


class TransformPILtoRGBTensor:
    def __call__(self, img):
        vassert(type(img) is Image.Image, 'Input is not a PIL.Image')
        width, height = img.size
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(height, width, 3)
        img = img.permute(2, 0, 1)
        return img


class ImagesPathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        img = self.transforms(img)
        return img


class Cifar10_RGB(CIFAR10):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img
