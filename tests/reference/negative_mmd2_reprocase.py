#!/usr/bin/env python
#
# Problem: KID(cifar10-train, cifar10-valid) < 0
# Following script reproduces this issue using the official compute_scores.py file from
# https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py commit id 4738ea6

import os
import subprocess
import tempfile

import numpy as np
import torchvision

print('Step 1: save cifar10 train and validation splits as npy files accepted by MMD scoring program...')

cifar_train = torchvision.datasets.CIFAR10(tempfile.gettempdir(), train=True, download=True)
cifar_valid = torchvision.datasets.CIFAR10(tempfile.gettempdir(), train=False, download=True)

imgs_train = [np.expand_dims(np.array(img[0]), 0) for img in cifar_train]
imgs_train = np.concatenate(imgs_train)

imgs_valid = [np.expand_dims(np.array(img[0]), 0) for img in cifar_valid]
imgs_valid = np.concatenate(imgs_valid)

np.save('cifar10_train', imgs_train)
np.save('cifar10_valid', imgs_valid)

print('Done')

print('Step 2: extract features as npy files accepted by MMD scoring program...')

for name in ('cifar10_train', 'cifar10_valid'):
    if not os.path.isfile(f'{name}_codes.npy'):
        args = ['python', 'compute_scores.py', f'{name}.npy', '--save-codes', f'{name}_codes.npy']
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert res.returncode == 0, \
            f'Codes extraction failed for {name}\nSTDOUT={res.stdout.decode()}\nSTDERR={res.stderr.decode()}'

print('Done')

print('Step 3: run MMD computation for cifar10 train-valid pair...')

args = ['python', 'compute_scores.py', 'cifar10_valid.npy', 'cifar10_train_codes.npy', '--no-inception', '--do-mmd']
res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
assert res.returncode == 0, f'MMD computation failed\nSTDOUT={res.stdout.decode()}\nSTDERR={res.stderr.decode()}'

print('Done')

print(res.stdout.decode())