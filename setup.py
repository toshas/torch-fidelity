#!/usr/bin/env python
import os

from setuptools import setup, find_packages

with open(os.path.join('torch_fidelity', 'version.py')) as f:
    version_pycode = f.read()
exec(version_pycode)

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
    license='Apache License 2.0',
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
