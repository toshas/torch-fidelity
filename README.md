![High-fidelity performance metrics for generative models in PyTorch](doc/img/header.png)

[![TestStatus](https://circleci.com/gh/toshas/torch-fidelity.svg?style=shield)](https://circleci.com/gh/toshas/torch-fidelity)
[![PyPiVersion](https://badge.fury.io/py/torch-fidelity.svg)](https://pypi.org/project/torch-fidelity/)
![PythonVersion](https://img.shields.io/badge/python-%3E%3D3.6-yellowgreen)
[![PyPiDownloads](https://pepy.tech/badge/torch-fidelity)](https://pepy.tech/project/torch-fidelity)
![License](https://img.shields.io/pypi/l/torch-fidelity)
[![Twitter Follow](https://img.shields.io/twitter/follow/AntonObukhov1?style=social&label=Subscribe!)](https://twitter.com/antonobukhov1)

This repository provides **epsilon-exact**, **efficient**, and **extensible** implementations of the popular metrics for 
generative model evaluation, including:
- Inception Score (ISC) [[paper]](https://arxiv.org/pdf/1606.03498.pdf)
- Fréchet Inception Distance (FID) [[paper]](https://arxiv.org/pdf/1706.08500.pdf)
- Kernel Inception Distance (KID) [[paper]](https://arxiv.org/pdf/1801.01401.pdf)
- Perceptual Path Length (PPL) [[paper]](https://arxiv.org/pdf/1812.04948.pdf)

**Epsilon-exactness**: Unlike many other reimplementations, the values produced by torch-fidelity match reference 
implementations up to machine precision. This allows using torch-fidelity for reporting metrics in papers instead of 
scattered and slow reference implementations. [Read more about epsilon-exactness of this code.](doc/fidelity.md) 

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

## What's new

See [CHANGELOG.md](CHANGELOG.md) for a full list of changes since the last release.

## Usage Examples with Command Line

Inception Score of CIFAR-10 training split. This command will automatically download the dataset:
```shell script
> fidelity --gpu 0 --isc --input1 cifar10-train

inception_score_mean: 11.23678
inception_score_std: 0.09514061
```

Inception Score of a directory of images stored in `~/images/`:
```shell script
> fidelity --gpu 0 --isc --input1 ~/images/
```

Inception Score of a generative model stored in `~/generator.pth`, whose input is a 128-dimensional standard normal 
random sample. This is equivalent to sampling 50000 images from the model, saving them in a temporary directory, and 
running the previous command:
```shell script
> fidelity --gpu 0 --isc --input1 ~/generator.pth --input1-model-z-size 128 --input1-model-num-samples 50000 
```

Fréchet Inception Distance between a directory of images `~/images/` and CIFAR-10 training split:
```shell script
> fidelity --gpu 0 --fid --input1 ~/images/ --input2 cifar10-train
```

Efficient computation of ISC, FID, and KID with feature caching between the first and the second inputs, where 
the value of `--input1` can be either a registered input (e.g., `cifar10-train`), or a directory with samples, or a 
generative model stored in either `.pth` or `.onnx` formats.
```shell script
> fidelity --gpu 0 --isc --fid --kid --input1 <input> --input2 cifar10-train
```

Efficient computation of ISC and PPL for `input1`, and FID and KID between a generative model stored in 
`~/generator.onnx` and CIFAR-10 training split:
```shell script
> fidelity 
  --gpu 0 
  --isc 
  --fid 
  --kid 
  --ppl 
  --input1 ~/generator.onnx 
  --input1-model-z-type normal
  --input1-model-z-size 128 
  --input1-model-num-samples 50000 
  --input2 cifar10-train 
```

## Quick Start with Python API

When it comes to tracking the performance of generative models as they train, evaluating metrics after every epoch becomes
prohibitively expensive due to long computation times. `torch_fidelity` tackles this problem by making full use of
caching to avoid recomputing common features and per-metric statistics whenever possible. 
Computing all metrics for 50000 32x32 generated images and `cifar10-train` takes only 2 min 26 seconds on NVIDIA P100 
GPU, compared to >10 min if using original codebases. Thus, computing metrics 20 times over the whole training cycle
makes overall training time just one hour longer.

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

Refer to [sngan_cifar10.py](examples/sngan_cifar10.py) for a complete training example. 

## Advanced Usage and Extensibility

The `fidelity` tool (run with `--help` to see the reference) is a command line interface to the `calculate_metrics` 
function (see [API](torch_fidelity/metrics.py)).

For the most part, `calculate_metrics` keyword arguments correspond to the command line keys. Exceptions are:
- Instances of `torch.util.data.Dataset` can be passed as inputs only programmatically;
- Instances of `torch.nn.GenerativeModelBase` can be passed as inputs only programmatically;
- CLI usage of registered inputs, feature extractors, _etc._ is limited to those pre-registered in 
  [torch_fidelity/registry.py](torch_fidelity/registry.py);
- CLI argument `--silent` sets kwarg `verbose=False`, whose default value is `True`;
- CLI argument `--no-datasets-download` sets kwarg `datasets_download=False`, whose default value is `True`;
- CLI argument `--no-samples-shuffle` sets kwarg `samples_shuffle=False`, whose default value is `True`;
- CLI argument `--no-cache` sets kwarg `cache=False`, whose default value is `True`;

### Inputs

ISC and PPL are computed for `input1` only, whereas FID and KID are computed between `input1` and `input2`.

Each input can be one of the following:
- Registered input, such as a `cifar10-train` string. Registered inputs are commonly used datasets, which can be 
  resolved by name, and are subject to caching;
- Path to a directory with samples;
- Path to a generative model in the ONNX or PTH (JIT) format;
- Instance of `torch.util.data.Dataset`;
- Instance of `torch_fidelity.GenerativeModelBase`.

### Registry

A number of inputs, feature extractors, sample similarities, noise source types, and interpolation methods have been 
pre-registered, and can be resolved in both CLI and API modes by their names.

- Inputs (can be used as values to `input1` and `input2` arguments):
  - `cifar10-train` - CIFAR-10 training split with 50000 images
  - `cifar10-val` - CIFAR-10 validation split with 10000 images
  - `stl10-train` - STL-10 training split with 500 images
  - `stl10-test` - STL-10 testing split with 800 images
  - `stl10-unlabeled` - STL-10 unlabeled split with 100000 images
- Feature extractors (can be used as values to the `feature_extractor` argument):
  - `inception-v3-compat` - a standard InceptionV3 feature extractor from the original reference implementations of the 
    Inception Score. This feature extractor is carefully ported to reproduce the original extractor's bilinear 
    interpolation and neural architecture quirks.
- Sample similarities (can be used as values to the `ppl_sample_similarity` argument): 
  - `lpips-vgg16` - a standard LPIPS sample similarity measure, based on a pre-trained VGG-16 and deep feature 
    aggregation.
- Noise source types (can be used as values to `input1_model_z_type` and `input2_model_z_type` arguments):
  - `normal` - standard normal distribution
  - `unit` - uniform distribution on a unit sphere
  - `uniform_0_1` - standard uniform distribution
- Interpolation methods (can be used as values to the `ppl_z_interp_mode` argument):
  - `lerp` - linear interpolation
  - `slerp_any` - spherical interpolation of `normal` samples
  - `slerp_unit` - spherical interpolation of `unit` samples

### Extensibility

On top of the above, it is possible to implement and register a new input, feature extractor, sample similarity, noise 
source type, or interpolation method before using using them in `calculate_metrics`:


#### Register a new input 

1. Prepare a new `torch.util.data.Dataset` subclass (as well as the `Transforms` pipeline) returning items which can 
   be directly fed into the feature extractor (refer to `Cifar10_RGB` for example),
2. Register it under some new name (`new-ds`): 
   `register_dataset('new-ds', lambda root, download: NewDataset(root, download))`,
3. Pass `new-ds` as a value of either `input1` or `input2` keyword arguments to `calculate_metrics`.

#### Register a new feature extractor 

1. Subclass a new feature extractor (e.g. `NewFeatureExtractor`) from `FeatureExtractorBase` class, implement all 
   methods, 
2. Register it under some new name (`new-fe`): `register_feature_extractor('new-fe', NewFeatureExtractor)`
3. Pass `feature_extractor='new-fe'` as a keyword argument to `calculate_metrics`.

#### Register a new sample similarity measure

1. Subclass a new sample similarity (e.g. `NewSampleSimilarity`) from `SampleSimilarityBase` class, implement all 
   methods, 
2. Register it under some new name (`new-ss`): `register_sample_similarity('new-ss', NewSampleSimilarity)`
3. Pass `ppl_sample_similarity='new-ss'` as a keyword argument to `calculate_metrics`.

#### Register a new noise source type 

1. Prepare a new function for drawing a sample from a multivariate distribution of a given shape, e.g., 
   `def random_new(rng, shape): pass`,
2. Register it under some new name (`new-ns`): `register_noise_source('new-ns', random_new)`,
3. Pass `new-ns` as a value of either `input1_model_z_type` or `input2_model_z_type` keyword arguments to 
   `calculate_metrics`.

#### Register a new interpolation method

1. Prepare a new sample interpolation function, e.g., `def new_interp(a, b, t): pass`,
2. Register it under some new name (`new-interp`): `register_interpolation('new-interp', new_interp)`,
3. Pass `new-interp` as a value of `ppl_z_interp_mode` keyword arguments to `calculate_metrics`.

### Storage, Cache, Datasets

The cached items can occupy quite a lot of space. The default cache root is created under 
`$ENV_TORCH_HOME/fidelity_cache`, which is usually under the `$HOME`. Users with limited home partition should use 
`--cache_root` key to change cache location, or specify `--no-cache` to disable it (not recommended).

Likewise, torchvision datasets may not be suitable for storage under the home directory (default location is
`$ENV_TORCH_HOME/fidelity_datasets`). It can be changed with the key `--datasets-root`. 
If torchvision datasets do not need to be downloaded, it is possible to disable download check using the key 
`--no-datasets-download`.

To save time on recomputations of features and statistics of inputs that do not change often, caching is enabled
on all registered inputs. In addition to that, one can force caching on a path input by assigning a new cache slot name
to such input via `input1_cache_name` or `input2_cache_name` keys, for the first and second positional arguments
respectively.

#### Working with Directories of Samples as Inputs

To collect files recursively under the path provided as the input, add `--samples-find-deep` command line key, or set 
the `samples_find_deep` keyword argument to True. 
To change file extensions picked up when traversing the path, specify `--samples-find-ext` command line key or 
the `samples_find_ext` keyword argument.
After the files list is collected, it is sorted alpha-numerically, and then shuffled. If shuffling is not desired, it
can be disabled by setting the `--no-samples-shuffle` key or using the `samples_shuffle` keyword argument.

#### Other Options

Both the `fidelity` command and the API provide several options to tweak the default behavior of the evaluation 
procedure. 
Below is a summary of use cases when changing the defaults is required:

- *Reducing GPU RAM usage*: by default, evaluating `cifar10-train` images with all metrics takes about 2.5 GB of RAM. 
  If it is desired to reduce memory footprint, use `--batch-size` key to reduce batch size. 
  It is not possible to go below a certain threshold required to store the feature extractor model (Inception).

- *Changing device placement*: by default, `fidelity` app decides whether to use GPU judging by the 
  `CUDA_VISIBLE_DEVICES` environment variable of the process. 
  It is possible to override this default behavior by specifying GPU ids with `--gpu` command line key, or forcing 
  computations to CPU with `--cpu` flag.

- *Verifying reproducibility*: For the sake of reproducibility, all pseudo-random numbers generators are pre-seeded 
  with the default value of `--rng_seed` command line key. 
  This affects choosing subsets and splits in metrics and the order of shuffled files from positional arguments.  
  One may want to change this flag to see the effect of seed on the metrics outputs. However, seeds should be kept 
  fixed and reported as a part of the evaluation protocol.  
 
- *Changing verbosity*: verbose mode is enabled by default, so both command line and programmatic interfaces report
  progress to `stderr`. 
  It can be disabled using the `--silent` command line flag, or `verbose` keyword argument. 
  However, it is useful to keep it enabled when extending the code with custom input sources or feature extractors, 
  as it allows to prevent unintended usage of the cache.
  
## Citation

Citation is required to reinforce the evaluation protocol in works relying on torch-fidelity. 
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
