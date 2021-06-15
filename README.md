![High-fidelity performance metrics for generative models in PyTorch](doc/img/header.png)

[![TestStatus](https://circleci.com/gh/toshas/torch-fidelity.svg?style=shield)](https://circleci.com/gh/toshas/torch-fidelity)
[![PyPiVersion](https://badge.fury.io/py/torch-fidelity.svg)](https://pypi.org/project/torch-fidelity/)
![PythonVersion](https://img.shields.io/badge/python-%3E%3D3.6-yellowgreen)
[![PyPiDownloads](https://pepy.tech/badge/torch-fidelity)](https://pepy.tech/project/torch-fidelity)
![License](https://img.shields.io/pypi/l/torch-fidelity)
[![Twitter Follow](https://img.shields.io/twitter/follow/AntonObukhov1?style=social&label=Subscribe!)](https://twitter.com/antonobukhov1)

This repository provides **precise**, **efficient**, and **extensible** implementations of the popular metrics for 
generative model evaluation, including:
- Inception Score ([ISC](https://arxiv.org/pdf/1606.03498.pdf))
- Fréchet Inception Distance ([FID](https://arxiv.org/pdf/1706.08500.pdf))
- Kernel Inception Distance ([KID](https://arxiv.org/pdf/1801.01401.pdf))
- Perceptual Path Length ([PPL](https://arxiv.org/pdf/1812.04948.pdf))

**Precision**: Unlike many other reimplementations, the values produced by torch-fidelity match reference 
implementations up to machine precision. This allows using torch-fidelity for reporting metrics in papers instead of 
scattered and slow reference implementations. [Read more about precision of this code.](doc/sphinx/source/precision.md) 

**Efficiency**: Feature sharing between different metrics saves recomputation time, and an additional caching 
level avoids recomputing features and statistics whenever possible. High efficiency allows using torch-fidelity in the 
training loop, for example at the end of every epoch.

**Extensibility**: Going beyond 2D image generation is easy due to high modularity and abstraction of the metrics from
input data, models, and feature extractors. For example, one can swap out InceptionV3 feature extractor for a one
accepting 3D scan volumes, such as used in MRI.

**TLDR; fast and reliable GAN evaluation in PyTorch**

## Installation

```shell script
pip install torch-fidelity
```

## Usage Examples with Command Line

Below are three examples of using torch-fidelity to evaluate metrics from the command line. See more examples in the 
documentation.

### Simple 

Inception Score of CIFAR-10 training split:
```shell script
> fidelity --gpu 0 --isc --input1 cifar10-train

inception_score_mean: 11.23678
inception_score_std: 0.09514061
```

### Medium 

Inception Score of a directory of images stored in `~/images/`:
```shell script
> fidelity --gpu 0 --isc --input1 ~/images/
```

### Pro

Efficient computation of ISC and PPL for `input1`, and FID and KID between a generative model stored in 
`~/generator.onnx` and CIFAR-10 training split:
```shell script
> fidelity \
  --gpu 0 \
  --isc \
  --fid \
  --kid \
  --ppl \
  --input1 ~/generator.onnx \ 
  --input1-model-z-type normal \
  --input1-model-z-size 128 \
  --input1-model-num-samples 50000 \ 
  --input2 cifar10-train 
```

## Quick Start with Python API

When it comes to tracking the performance of generative models as they train, evaluating metrics after every epoch 
becomes prohibitively expensive due to long computation times. 
`torch_fidelity` tackles this problem by making full use 
of caching to avoid recomputing common features and per-metric statistics whenever possible. 
Computing all metrics for 50000 32x32 generated images and `cifar10-train` takes only 2 min 26 seconds on NVIDIA P100 
GPU, compared to >10 min if using original codebases. 
Thus, computing metrics 20 times over the whole training cycle makes overall training time just one hour longer.

In the following example, assume unconditional image generation setting with CIFAR-10, and the generative model 
`generator`, which takes a 128-dimensional standard normal noise vector.

First, import the module:

```python
import torch_fidelity
```

Add the following lines at the end of epoch evaluation:
```python
wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(generator, 128, 'normal', 0)

metrics_dict = torch_fidelity.calculate_metrics(
    input1=wrapped_generator, 
    input2='cifar10-train', 
    cuda=True, 
    isc=True, 
    fid=True, 
    kid=True, 
    verbose=False,
)
```

The resulting dictionary with computed metrics can logged directly to tensorboard, wandb, or console: 

```python
print(metrics_dict)
```

Output:

```python
{
    'inception_score_mean': 11.23678, 
    'inception_score_std': 0.09514061, 
    'frechet_inception_distance': 18.12198,
    'kernel_inception_distance_mean': 0.01369556, 
    'kernel_inception_distance_std': 0.001310059
}
```

### Example of Integration with the Training Loop

Refer to [sngan_cifar10.py](examples/sngan_cifar10.py) for a complete training example.

Evolution of fixed generator latents in the example:

![Evolution of fixed generator latents](doc/img/sngan-cifar10.gif)

A generator checkpoint resulting from training the example can be downloaded 
[here](https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/example-sngan-cifar10-generator.pth). 

## Citation

Citation is recommended to reinforce the evaluation protocol in works relying on torch-fidelity. 
To ensure reproducibility when citing this repository, use the following BibTeX:

```
@misc{obukhov2020torchfidelity,
  author={Anton Obukhov and Maximilian Seitzer and Po-Wei Wu and Semen Zhydenko and Jonathan Kyl and Elvis Yu-Jing Lin},
  year=2020,
  title={High-fidelity performance metrics for generative models in PyTorch},
  url={https://github.com/toshas/torch-fidelity},
  publisher={Zenodo},
  version={v0.2.0},
  doi={10.5281/zenodo.3786540},
  note={Version: 0.2.0, DOI: 10.5281/zenodo.3786540}
}
```
