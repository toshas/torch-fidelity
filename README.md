![High-fidelity performance metrics for generative models in PyTorch](doc/img/header.png)

[![TestStatus](https://circleci.com/gh/toshas/torch-fidelity.svg?style=shield)](https://circleci.com/gh/toshas/torch-fidelity)
[![PyPiVersion](https://badge.fury.io/py/torch-fidelity.svg)](https://pypi.org/project/torch-fidelity/)
![PythonVersion](https://img.shields.io/badge/python-%3E%3D3.6-yellowgreen)
[![PyPiDownloads](https://pepy.tech/badge/torch-fidelity)](https://pypi.org/project/torch-fidelity/)
![License](https://img.shields.io/pypi/l/torch-fidelity)

Evaluation of generative models such as GANs is an important part of the deep learning research. 
In the domain of 2D image generation, three approaches became widely spread: 
**Inception Score** (aka IS) [[1]](https://arxiv.org/pdf/1606.03498.pdf), **Fréchet Inception Distance** (aka FID) 
[[2]](https://arxiv.org/pdf/1706.08500.pdf), and **Kernel Inception Distance** (aka KID) 
[[3]](https://arxiv.org/pdf/1801.01401.pdf). 

These metrics, despite having a clear mathematical and algorithmic description, were initially implemented in 
TensorFlow, and inherited a few properties of the framework itself (see Interpolation) and the code they relied upon 
(see Model). 
These design decisions were effectively baked into the evaluation protocol and became an inherent part of the 
metrics specification.
As a result, researchers wishing to compare against state of the art in generative modeling are forced to perform 
evaluation using codebases of the original metric authors. 
Reimplementations of metrics in PyTorch and other frameworks exist, but they do not provide a proper level of fidelity, 
thus making them unsuitable for reporting results and comparing them to other methods.   

This software aims to provide epsilon-exact implementations of the said metrics in PyTorch, and thus remove
inconveniences associated with generative models evaluation and development. All steps of the evaluation pipeline are
correctly tested, with relative errors and sources of remaining non-determinism summarized in sections below.  

**TLDR; fast and reliable GAN evaluation in PyTorch**

## Installation

```shell script
pip install torch-fidelity
```

## Quick Start Usage with Command Line

Inception Score of CIFAR-10 training split:
```shell script
> fidelity --gpu 0 --isc cifar10-train

inception_score_mean: 11.23678
inception_score_std: 0.09514061
```

Inception Score of a directory of images:
```shell script
> fidelity --gpu 0 --isc <path>
```

Fréchet Inception Distance between a directory of images and CIFAR-10 training split:
```shell script
> fidelity --gpu 0 --fid <path> cifar10-train
```

Kernel Inception Distance between two directories with images:
```shell script
> fidelity --gpu 0 --kid <path1> <path2>
```

Efficient computation of all three metrics with feature caching (Inception Score computed only for the first positional 
argument):
```shell script
> fidelity --gpu 0 --isc --fid --kid <path> cifar10-train
```

## Quick Start Usage with Python API

When it comes to tracking the performance of generative models as they train, evaluating metrics after every epoch becomes
prohibitively expensive due to long computation times. `torch_fidelity` tackles this problem by making full use of
caching to avoid recomputing common features and per-metric statistics whenever possible. 
Computing all metrics for 50000 32x32 generated images and `cifar10-train` takes only 2 min 26 seconds on NVIDIA P100 
GPU, compared to >10 min if using original codebases. Thus, computing metrics 20 times over the whole training cycle
makes overall training time just one hour longer.  

```python
from torch_fidelity import calculate_metrics

# Each input can be either a string (path to images, registered input), or a Dataset instance
metrics_dict = calculate_metrics(input1, input2, cuda=True, isc=True, fid=True, kid=True, verbose=False)

print(metrics_dict)
# Output:
# {
#     'inception_score_mean': 11.23678, 
#     'inception_score_std': 0.09514061, 
#     'frechet_inception_distance': 18.12198,
#     'kernel_inception_distance_mean': 0.01369556, 
#     'kernel_inception_distance_std': 0.001310059
# }
```

## Advanced Usage and Extensibility

The `fidelity` command takes two positional arguments and a few optional keys (run `fidelity --help` to see reference). 
Most of the keys directly correspond to keyword arguments accepted by the `calculate_metrics` function (see 
[API](torch_fidelity/metrics.py)).  
Each positional argument can be either a path to a directory containing samples, or a _registered input_.
Registered inputs are commonly used datasets, which can be resolved by name, and are subject to caching. 
Currently, there are just two registered inputs - `cifar10-train` and `cifar10-val`, however extending them with
other torchvision datasets is straightforward (see [API](torch_fidelity/registry.py), Extensibility). 

### Storage, Cache, Datasets

The cached items can occupy quite a lot of space. The default cache root is created under `$ENV_TORCH_HOME/fidelity_cache`,
which is usually under the `$HOME`. Users with limited home partition should use `--cache_root` key to change cache
location, or specify `--no-cache` to disable it (not recommended).

Likewise, torchvision datasets may not be suitable for storage under the home directory (default location is
`$ENV_TORCH_HOME/fidelity_datasets`). It can be changed with the key `--datasets-root`. 
If torchvision datasets do not need to be downloaded, it is possible to disable download check using the key 
`--datasets-downloaded`.

To save time on recomputations of features and statistics of inputs that do not change often, caching is enabled
on all registered inputs. In addition to that, one can force caching on a path input by assigning a new cache slot name
to such input via `cache_input1_name` or `cache_input2_name` keys, for the first and second positional arguments
respectively.

### Extensibility

#### Add your own feature extractor instead of Inception in three easy steps 

1. Subclass a new Feature Extractor (e.g. `NewFeatureExtractor`) from `FeatureExtractorBase` class, implement all 
methods, 
2. Register it using `register_feature_extractor` method: `register_feature_extractor('new-fe', NewFeatureExtractor)`
3. Pass `feature_extractor='new-fe'` as a keyword argument to `calculate_metrics`.

#### Add a new dataset to the pool of registered inputs in three easy steps

1. Prepare a new `Dataset` subclass (as well as the `Transforms` pipeline) returning items which can be directly fed
into the feature extractor (refer to `Cifar10_RGB` for example),
2. Register it under some new name (`new-ds`): 
`register_dataset('new-ds', lambda root, download: NewDataset(root, download))`,
3. Pass the registered input name (`new-ds`) as a positional argument to `calculate_metrics`.

#### Working with Files

If a positional argument is not a Dataset instance or a registered input, it is treated as a directory with images (jpg, png). 
To collect files recursively under the provided path, add `--samples-find-deep` command line key, or set 
the `samples_find_deep` keyword argument to True. 
To change file extensions picked up when traversing the path, specify `--samples-find-ext` command line key or 
the `samples_find_ext` keyword argument.
After the files list is collected, it is sorted alpha-numerically, and then shuffled. If shuffling is not desired, it
can be disabled by setting the `--samples-alphanumeric` key or using the `samples_shuffle` keyword argument.

#### Other Options

Both the `fidelity` command and the API provide several options to tweak the default behavior of the evaluation 
procedure. 
Below is a summary of use cases when changing the defaults is required:

- *Reducing GPU RAM usage*: by default, evaluating `cifar10-train` images with all metrics takes about 2.5 GB of RAM. 
If it is desired to reduce memory footprint, use `--batch-size` key to reduce batch size. 
It is not possible to go below a certain threshold required to store the feature extractor model (Inception).

- *Changing device placement*: by default, `fidelity` app decides whether to use GPU judging by the `CUDA_VISIBLE_DEVICES` 
environment variable of the process. It is possible to override this default behavior by specifying GPU ids with
`--gpu` command line key, or forcing computations to CPU with `--cpu` flag.

- *Verifying reproducibility*: For the sake of reproducibility, all pseudo-random numbers generators are pre-seeded with 
the default value of `--rng_seed` command line key. 
This affects choosing subsets and splits in metrics and the order of shuffled files from positional arguments.  
One may want to change this flag to see the effect of seed on the metrics outputs. However, seeds should be kept fixed
and reported as a part of the evaluation protocol.  
 
- *Changing verbosity*: verbose mode is enabled by default, so both command line and programmatic interfaces report
progress to `stderr`. It can be disabled using the `--silent` command line flag, or `verbose` keyword argument. 
However, it is useful to keep it enabled when extending the code with custom input sources or feature extractors, 
as it allows to prevent unintended usage of the cache.

## Level of Fidelity

The path from inputs to metrics contains the following steps:
1. Image interpolation (resizing) and normalization (bringing RGB values to the range acceptable by the pre-trained 
model),
2. Using the pre-trained model to extract features from properly sized and normalized images,
3. Compute metrics values from the extracted features.

Let's consider how close each of these steps can be implemented in a different framework:

### Interpolation

Bilinear interpolation is a known source of problems with determinism and reproducibility, as there are quite a few ways
to align corners of an image, each giving a different result [[5]](https://machinethink.net/blog/coreml-upsampling/). 
The effect of the issue is amplified when going from small to large image sizes, as is the case with evaluating CIFAR10: 
32x32 is upsampled to 299x299. `torch_fidelity` provides a custom implementation of bilinear resize function, implemented
exactly as in TensorFlow. The tests show that on CPU this function produces bit-exact results consistent with 
TF implementation. However, on GPU the output is epsilon-exact, presumably due to fused-multiply-add floating-point 
optimizations employed in gridpoint coordinates computation on the GPU hardware side.

### Pre-trained Model

All three metrics were introduced to evaluate 2D images with Inception V3 as a feature extractor. The architecture
of Inception V3 used in TensorFlow is slightly different from both the paper and what is implemented in 
torchvision. 
The author of [[4]](https://github.com/mseitzer/pytorch-fid) did the heavy lifting of porting the original architecture
to PyTorch, as well as exporting all learnable parameters into a PyTorch checkpoint. `torch_fidelity` attempts to 
download the checkpoint from github. However, it is also possible to perform model conversion locally and load the 
checkpoint from a file (see `util_convert_inception_weights.py` and `--feature-extractor-weights-path` command line key). 

As indicated by the tests (`test_convolution.py`, `test_inception.py`), the output of the pre-trained TensorFlow model
varies from time to time given the same inputs, which can be fixed using NVIDIA's `tensorflow-determinism` package. 
PyTorch uses CuDNN as a backend, and its outputs do not vary even when determinism is disabled. Slight differences
between outputs of TF and PT convolutions exist even with maximum determinism, which does not allow us to claim
bit-exactness. Further troubleshooting reasons for inexact outputs of convolution is complicated by varying behavior
between CPU and GPU, and also between different CPUs.

### Computing metrics from features
This step is usually performed on CPU using a math library like scipy. It is neither the bottleneck of computations,
nor the source of precision loss or non-determinism; `torch_fidelity` uses the original features-to-metric code for this 
step in most cases.

### Relative Errors Summary

The tests are performed using fixed pseudo-random subsets of 5000 CIFAR10 images (see [tests/precision](tests/precision)).  
Below is a list of pixel-, feature-, or metric-wise relative errors of different steps of the computational 
pipeline, between reference implementations and `torch_fidelity`:
- Interpolation: 1e-5
- Convolution: 1e-7
- InceptionV3 RGB 8bpp to pool3 features: 1e-4
- Inception Score: 1e-3
- Fréchet Inception Distance: 1e-6
- Kernel Inception Distance: 1e-6

The badge 'CircleCI: passing' means that the declared tolerances were met during the last test run, involving all the 
latest versions of the dependencies. 

## Feedback

Feedback in the form of comments, tickets, and stars is welcome!

## References and Acknowledgements

Original implementations:
* Inception Score: 
  * https://github.com/openai/improved-gan/blob/master/inception_score/model.py
  * https://github.com/bioinf-jku/TTUR/fid.py 
  * https://github.com/bioinf-jku/TTUR/blob/master/FIDvsINC/fidutils.py
* Fréchet Inception Distance: https://github.com/bioinf-jku/TTUR/fid.py
* Kernel Inception Distance: https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py

[1] Tim Salimans et al., Improved Techniques for Training GANs, https://arxiv.org/pdf/1606.03498.pdf

[2] Martin Heusel et al., GANs Trained by a Two Time-Scale Update Rule
Converge to a Local Nash Equilibrium, https://arxiv.org/pdf/1706.08500.pdf

[3] Mikołaj Binkowski et al., Demystifying MMD GANs, https://arxiv.org/pdf/1801.01401.pdf

[4] https://github.com/mseitzer/pytorch-fid

[5] https://machinethink.net/blog/coreml-upsampling/
