#!/usr/bin/env python3
import os
import sys
import tempfile

import torchvision
from tqdm import tqdm

dataset_name, path = sys.argv[1:]

assert not os.path.exists(path), 'Path should not exist'
os.makedirs(path)

if dataset_name == 'cifar10':
    dataset = torchvision.datasets.CIFAR10(tempfile.gettempdir(), train=True, transform=None, download=True)
else:
    raise NotImplementedError

for i, sample in tqdm(enumerate(dataset)):
    img = sample[0]
    sample_name = os.path.join(path, f'{i:05d}.png')
    img.save(sample_name)
