#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pytorch-fidelity',
    version='0.0.1',
    description='Deterministic performance metrics for generative models in PyTorch',
    install_requires=requirements,
    packages=find_packages(),
    author='Anton Obukhov',
    url='https://www.github.com/toshas/torch-fidelity',
)
