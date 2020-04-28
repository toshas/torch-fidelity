# High-fidelity performance metrics for generative models in PyTorch

Evaluation of generative models such as GANs is an important part of deep learning research. 
In the domain of 2D image generation, three approaches became widely spread: 
*Inception Score* (aka IS) [[1]](https://arxiv.org/pdf/1606.03498.pdf), *Fréchet Inception Distance* (aka FID) 
[[2]](https://arxiv.org/pdf/1706.08500.pdf), and *Kernel Inception Distance* (aka KID, MMD) 
[[3]](https://arxiv.org/pdf/1801.01401.pdf). 

These metrics, despite having a clear mathematical and algorithmic description, were initially implemented in 
TensorFlow, and inherited a few hidden properties of the legacy code they relied upon. 
These properties make the metrics outputs depend not only on the input data, but also on the framework design, 
which effectively bakes in the framework requirement into the evaluation protocol. 
As a result, researchers wishing to compare against state of the art in generative modelling are forced to perform 
evaluation using codebases of the original metric authors. 
Reimplementations of metrics in PyTorch and other frameworks exist, but they do not have a proper level of fidelity, 
thus making them unsuitable for reporting results and comparing to other methods.   

This code aims to provide epsilon-exact implementations of the said generative metrics in PyTorch, and thus remove 
inconveniences associated with generative models evaluation. All steps of the evaluation pipeline are properly tested, 
with levels of exactness and sources of remaining non-determinism summarized in sections below.  

# Installation

```shell script
pip install torch-fidelity
```

or

```shell script
pip install git+https://github.com/toshas/torch-fidelity.git
```

# Quick Start Usage

Inception Score of CIFAR-10 training split:
```shell script
> fidelity --isc cifar10-train

inception_score_mean: 11.23678
inception_score_std: 0.09514061
```

Inception Score of a directory of images:
```shell script
> fidelity --isc <path>
```

Fréchet Inception Distance between a directory of images and CIFAR-10 training split:
```shell script
> fidelity --fid <path> cifar10-train
```

Kernel Inception Distance between two directories with images:
```shell script
> fidelity --kid <path1> <path2>
```

Efficient computation of all three metrics with feature caching (Inception Score computed only for the first positional 
argument):
```shell script
> fidelity --isc --fid --kid <path> cifar10-train
```

# Advanced Usage and Extensibility

`fidelity` script takes two positional arguments and a number of keys, having good default values. 
Each positional argument can be either a path to a directory containing samples, or a _registered input_. 
Currently there are just two registered inputs - `cifar10-train` and `cifar10-val`, however extending them with
other torchvision datasets is straightforward. 

To save time on recomputations of features and statistics of inputs which do not change often, caching is enabled 
on all registered inputs. In addition to that, one can force caching on a path input by assigning a new cache name 
to such input via `cache_input1_name` or `cache_input2_name` keys.

## Storage, Cache, Datasets

The cached items occupy quite a lot of space. The default cache root is created under `$ENV_TORCH_HOME/fidelity_cache`,
which is usually under the `$HOME`. Users with limited home storage should use `--cache_root` key to change cache 
location, or specify `--no-cache` to disable it.

Likewise, torchvision datasets may not be suitable for storage under the home folder. Their default location is under
`$ENV_TORCH_HOME/fidelity_datasets`, it can be changed with the key `--datasets-root`. If torchvision datasets do not need
to be downloaded, it is possible to disable download check using the key `--datasets-downloaded`.

## Extensibility

Add your own feature extractor instead of Inception in three easy steps: 
1. Subclass a new Feature Extractor (e.g. `NewFeatExt`) from `FeatureExtractorBase` class, implement all methods, 
2. Register it using `register_feature_extractor` method: `register_feature_extractor('new-fe', NewFeatExt)`
3. Specify `--feature-extractor new-fe` key.

Add a new dataset to the pool of registered inputs in three easy steps:
1. Prepare a `Dataset` class (as well as the `Transforms` pipeline) returning items which can be directly fed
into the feature extractor (refer to `Cifar10_RGB` for example),
2. Register it: `register_dataset('new-ds', lambda root, download: NewDataset(root, download))`,
3. Use new-ds as a positional argument.

## Command Line Reference

```shell script
usage: fidelity.py [-h] [-b BATCH_SIZE] [-g GPU | -c] [-j] [-i] [-f] [-k]                                                                                                            
                   [--feature-extractor {inception-v3-compat}]                                                                                                                       
                   [--feature-layer-isc FEATURE_LAYER_ISC]                                                                                                                           
                   [--feature-layer-fid FEATURE_LAYER_FID]                                                                                                                           
                   [--feature-layer-kid FEATURE_LAYER_KID]                                                                                                                           
                   [--feature-extractor-weights-path FEATURE_EXTRACTOR_WEIGHTS_PATH]                                                                                                 
                   [--isc-splits ISC_SPLITS] [--kid-subsets KID_SUBSETS]                                                                                                             
                   [--kid-subset-size KID_SUBSET_SIZE]                                                                                                                               
                   [--kid-degree KID_DEGREE] [--kid-gamma KID_GAMMA]                                                                                                                 
                   [--kid-coef0 KID_COEF0] [--samples-alphanumeric]                                                                                                                  
                   [--samples-find-deep] [--samples-find-ext SAMPLES_FIND_EXT]                                                                                                       
                   [--samples-ext-lossy SAMPLES_EXT_LOSSY]                                                                                                                           
                   [--datasets-root DATASETS_ROOT] [--datasets-downloaded]                                                                                                           
                   [--cache-root CACHE_ROOT] [--no-cache]                                                                                                                            
                   [--rng-seed RNG_SEED] [--silent]                                                                                                                                  
                   input1 [input2]                                                                                                                                                   
```

# Level of Fidelity

```
CODE              FID VALUES                            RUN TIME         GPU RAM USED
TTUR              18.121856, 18.120691, 18.121186       3 min 15 sec     15 GB
pytorch-fid       18.185271, 18.185271, 18.185271       2 min 01 sec     2 GB
```

# References and Credits

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).

This project would not be possible without Maximilian Seitzer's [pytorch-fid](https://github.com/mseitzer/pytorch-fid).  

[1] Tim Salimans et al, Improved Techniques for Training GANs, https://arxiv.org/pdf/1606.03498.pdf

[2] Martin Heusel et al, GANs Trained by a Two Time-Scale Update Rule
Converge to a Local Nash Equilibrium, https://arxiv.org/pdf/1706.08500.pdf

[3] Mikołaj Binkowski et al, Demystifying MMD GANs, https://arxiv.org/pdf/1801.01401.pdf
