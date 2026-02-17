#!/usr/bin/env python
import os

from setuptools import setup, find_packages

with open(os.path.join("torch_fidelity", "version.py")) as f:
    version_pycode = f.read()
exec(version_pycode)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

long_description = """
torch-fidelity is the reference implementation of generative image model evaluation metrics in
PyTorch, providing epsilon-exact computation of Inception Score (ISC), Fréchet Inception Distance
(FID), Kernel Inception Distance (KID), and Precision and Recall (PRC). It supports multiple
feature extractors including InceptionV3, CLIP, DINOv2, and VGG16.

Originally created to bring numerically faithful metric implementations to the PyTorch ecosystem —
matching TensorFlow reference code to machine precision — torch-fidelity is now widely adopted as
a foundational dependency (e.g., by torchmetrics) and a standard tool for benchmarking GANs,
diffusion models, flow-matching, and other generative approaches.

Key features:
- Epsilon-exact: values match reference implementations to floating-point precision
- Efficient: feature sharing and multi-level caching minimize redundant computation
- Extensible: register custom feature extractors to evaluate any modality — images, video, audio,
  3D volumes, or anything else with a suitable learned representation

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
        "diffusion",
        "flow-matching",
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
        "clip",
        "dinov2",
        "vgg16",
    ],
    entry_points={
        "console_scripts": [
            "fidelity=torch_fidelity.fidelity:main",
        ],
    },
)
