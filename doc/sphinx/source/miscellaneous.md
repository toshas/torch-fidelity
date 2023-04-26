# Miscellaneous

## Inputs

ISC and PPL are computed for `input1` only, whereas FID and KID are computed between `input1` and `input2`.

Each input can be one of the following:
- Registered input, such as a `cifar10-train` string. Registered inputs are commonly used datasets, which can be 
  resolved by name, and are subject to caching;
- Path to a directory with samples;
- Path to a generative model in the ONNX or PTH (JIT) format;
- Instance of `torch.util.data.Dataset`;
- Instance of `torch_fidelity.GenerativeModelBase`.

## Storage, Cache, Datasets

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

## Working with Directories of Samples as Inputs

To collect files recursively under the path provided as the input, add `--samples-find-deep` command line key, or set 
the `samples_find_deep` keyword argument to True. 
To change file extensions picked up when traversing the path, specify `--samples-find-ext` command line key or 
the `samples_find_ext` keyword argument.
After the files list is collected, it is sorted alpha-numerically, and then shuffled. If shuffling is not desired, it
can be disabled by setting the `--no-samples-shuffle` key or using the `samples_shuffle` keyword argument.
Since in-the-wild images can be of arbitrary shapes, it is necessary to bring them all to the same canonical size and
square shape, compatible with the evaluation protocol. Specify `--samples_resize_and_crop` to resize all images to match
the given size on the shorter side and then perform center cropping.

## Other Options

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
