#!/usr/bin/env python3
import math
import os
import sys
import tempfile

import torchvision
from tqdm import tqdm

dataset_name, path = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

os.makedirs(path, exist_ok=True)

if dataset_name == 'cifar10-train':
    dataset = torchvision.datasets.CIFAR10(tempfile.gettempdir(), train=True, download=True)
elif dataset_name == 'cifar10-valid':
    dataset = torchvision.datasets.CIFAR10(tempfile.gettempdir(), train=False, download=True)
else:
    raise NotImplementedError

nsamples = len(dataset)
decimal_pts = int(math.log10(nsamples)) + 1

for i, sample in tqdm(enumerate(dataset)):
    if limit is not None and i == limit:
        break
    img = sample[0]
    sample_name = os.path.join(path, f'{i:0{decimal_pts}d}.png')
    img.save(sample_name)
