#!/usr/bin/env python3
import argparse
import math
import os
import tempfile

import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset_name", type=str, help=f"Dataset name")
    parser.add_argument("dataset_dst_path", type=str, help="Where to dump dataset")
    parser.add_argument("-l", "--limit", default=None, type=int, help="Random subset of dataset of this size")
    parser.add_argument("-n", "--noise", action="store_true", help="Add image noise")
    parser.add_argument("-r", "--resolution", type=int, default=None, help="Force this resolution")
    args = parser.parse_args()

    rng = np.random.RandomState(2020)
    os.makedirs(args.dataset_dst_path, exist_ok=True)

    if args.dataset_name == "cifar10-train":
        dataset = torchvision.datasets.CIFAR10(tempfile.gettempdir(), train=True, download=True)
    elif args.dataset_name == "cifar10-valid":
        dataset = torchvision.datasets.CIFAR10(tempfile.gettempdir(), train=False, download=True)
    else:
        raise NotImplementedError

    nsamples = len(dataset)
    decimal_pts = int(math.log10(nsamples)) + 1

    indices = range(nsamples)
    if args.limit is not None:
        indices = rng.choice(len(dataset), min(args.limit, nsamples), replace=False)

    for i in tqdm(indices):
        sample = dataset[i]
        img = sample[0]
        if args.resolution:
            img = img.resize((args.resolution, args.resolution))
        if args.noise:
            img = np.array(img).astype(np.float32)
            img += rng.randn(*img.shape) * 64
            img = Image.fromarray(img.astype(np.uint8))
        sample_name = os.path.join(args.dataset_dst_path, f"{i:0{decimal_pts}d}.png")
        img.save(sample_name)


if __name__ == "__main__":
    main()
