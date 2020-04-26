import sys

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_and_extract_archive


class TransformPILtoRGBTensor:
    def __call__(self, img):
        assert type(img) is Image.Image, 'Input is not a PIL.Image'
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
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img

    def download(self):
        if self._check_integrity():
            print('CIFAR10 dataset already downloaded and verified', file=sys.stderr)
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
