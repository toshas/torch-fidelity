#!/usr/bin/env python

from setuptools import setup, find_packages

from torch_fidelity.version import __version__

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(
    name='torch_fidelity',
    version=__version__,
    description='High-fidelity performance metrics for generative models in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    python_requires='>=3.6',
    packages=find_packages(),
    author='Anton Obukhov',
    url='https://www.github.com/toshas/torch-fidelity',
    keywords=[
        'inception', 'score', 'frechet', 'distance', 'kernel', 'reproducibility', 'fidelity',
        'deep', 'generative', 'adversarial', 'networks', 'gan',
    ],
    entry_points={
        'console_scripts': [
            'fidelity=torch_fidelity.fidelity:main',
        ],
    },
)
