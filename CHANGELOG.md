# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2021-05-17
### Added
- API
  - `calculate_metrics`
    - `ppl`: Calculate PPL (Perceptual Path Length)
    - `model`: Path to generator model in ONNX format, or an instance of torch.nn.Module
    - `model_z_type`: Type of noise for generator model input
    - `model_z_size`: Dimensionality of generator noise
    - `model_conditioning_num_classes`: Number of classes for conditional generation, or 0 for unconditional
    - `ppl_num_samples`: Number of samples to generate using the model in PPL
    - `ppl_epsilon`: Interpolation step size in PPL
    - `ppl_z_interp_mode`: Noise interpolation mode in PPL
- Command line
  - `--ppl`: Calculate PPL (Perceptual Path Length)
  - `--model`: Path to generator model in ONNX format
  - `--model-z-type`: Type of noise for generator model input
  - `--model-z-size`: Dimensionality of generator noise
  - `--model-conditioning-num-classes`: Number of classes for conditional generation, or 0 for unconditional
  - `--ppl-num-samples`: Number of samples to generate using the model in PPL
  - `--ppl-epsilon`: Interpolation step size in PPL
  - `--ppl-z-interp-mode`: Noise interpolation mode in PPL
- ONNX model loading via command line (`--model`) functionality to support PPL
- STL-10 dataset registered inputs: `stl10-train`, `stl10-test`, `stl10-unlabeled`
- Test for LPIPS fidelity as compared to StyleGAN PyTorch implementation
- Test for feature extraction layer
- Unrecognized command line arguments warning

### Changed
- API
  - First input positional argument of `calculate_metrics` is now expected as a value to kwarg `input1`
  - Second input (optional) positional argument of `calculate_metrics` is now expected as a value to kwarg argument 
  `input2`
- Command line
  - First input positional argument is now expected as a value to the key `--input1`
  - Second input (optional) positional argument is now expected as a value to the key `--input2`
  - `--datasets-downloaded` renamed to `--no-datasets-download`
  - `--samples-alphanumeric` renamed to `--no-samples-shuffle`
- Change `torch.save` to an atomic saving operation in all functions of the caching layer, which makes it 
  safe to use torch-fidelity in multiprocessing environment, such as a compute cluster with a shared file system.

### Fixed
- [#15](https://github.com/toshas/torch-fidelity/issues/15): Fix '64', '192', and '768' feature layers usage in all metrics
- [#8](https://github.com/toshas/torch-fidelity/issues/8): Fix a missing exception for when KID subset size is larger than the number of samples in one of the inputs
- Fix a missing check that the elements of inputs are actually instances of `torch.Tensor`

## [0.2.0] - 2020-05-05
### Added
- Initial release with Inception Score (ISC), Frechet Inception Distance (FID),
  and Kernel Inception Distance (KID) metrics
- Precision unit tests for all three metrics
- Command line tool and Python API
