#!/usr/bin/env python
import os

from setuptools import setup, find_packages

with open(os.path.join("torch_fidelity", "version.py")) as f:
    version_pycode = f.read()
exec(version_pycode)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

long_description = """
Evaluation of generative models such as GANs is an important part of the deep learning research. 
In the domain of 2D image generation, three approaches became widely spread: Inception Score 
(aka IS), Fréchet Inception Distance (aka FID), and Kernel Inception Distance (aka KID).

These metrics, despite having a clear mathematical and algorithmic description, were initially 
implemented in TensorFlow, and inherited a few properties of the framework itself and the code 
they relied upon. These design decisions were effectively baked into the evaluation protocol and 
became an inherent part of the metrics specification. As a result, researchers wishing to 
compare against state of the art in generative modeling are forced to perform evaluation using 
codebases of the original metric authors. Reimplementations of metrics in PyTorch and other 
frameworks exist, but they do not provide a proper level of fidelity, thus making them 
unsuitable for reporting results and comparing to other methods.   

This software aims to provide epsilon-exact implementations of the said metrics in PyTorch, and thus 
remove inconveniences associated with generative models evaluation and development. 
Find more details and the most up-to-date information on the project webpage:
https://www.github.com/toshas/torch-fidelity
"""

setup(
    name="torch_fidelity",
    version=__version__,
    description="High-fidelity performance metrics for generative models in PyTorch",
    long_description=long_description,
    long_description_content_type="text/plain",
    install_requires=requirements,
    python_requires=">=3.6",
    packages=find_packages(),
    author="Anton Obukhov",
    license="Apache License 2.0",
    url="https://www.github.com/toshas/torch-fidelity",
    keywords=[
        "reproducibility",
        "fidelity",
        "deep",
        "generative",
        "adversarial",
        "networks",
        "gan",
        "inception",
        "score",
        "frechet",
        "distance",
        "kernel",
        "perceptual",
        "path",
        "length",
        "isc",
        "fid",
        "kid",
        "lpips",
        "ppl",
        "prc",
        "precision",
        "recall",
    ],
    entry_points={
        "console_scripts": [
            "fidelity=torch_fidelity.fidelity:main",
        ],
    },
)
