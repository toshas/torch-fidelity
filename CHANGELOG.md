# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Unreleased
### Added in 0.4.0
- New metrics: Precision, Recall, F-score
- New registered inputs: `cifar100-train`, `cifar100-val`
- New registered feature extractors: `clip-vit-b-32`, `vgg16`, `dinov2-vit-s-14`, `dinov2-vit-b-14`, `dinov2-vit-l-14`, `dinov2-vit-g-14`
- API
  - `calculate_metrics`
    - `prc`: Calculate PRC (Precision and Recall)
    - `prc_neighborhood`: Number of nearest neighbours to consider in PRC
    - `prc_batch_size`: Batch size in PRC
    - `feature_layer_prc`: Name of the feature layer to use with PRC metric
    - `samples_resize_and_crop`: Transform all images found in the directory to a given size and square shape
    - `feature_extractor`: Accepts a new feature extractors `clip-vit-b-32`, `vgg16`, `dinov2-vit-s-14`, `dinov2-vit-b-14`, `dinov2-vit-l-14`, `dinov2-vit-g-14`
    - `feature_extractor_internal_dtype`: Allows to change the internal dtype used in the feature extractor's weights and activations; might be useful to counter numerical issues arising in fp32 implementations, e.g. those seen with the growth of the batch size
    - `feature_extractor_compile`: Compile feature extractor (experimental: may have negative effect on the metrics numerical precision)
    - `kid_kernel`: Allows choosing between the default `poly` (polynomial, default) and `rbf` (RBF) kernels 
    - `kid_kernel_rbf_sigma`: Specifies RBF kernel sigma in KID  
- Command line
    - `prc`: Calculate PRC (Precision and Recall)
    - `prc-neighborhood`: Number of nearest neighbours to consider in PRC
    - `prc-batch-size`: Batch size in PRC
    - `feature-layer-prc`: Name of the feature layer to use with PRC metric
    - `--samples-resize-and-crop`: Transform all images found in the directory to a given size and square shape
    - `--feature-extractor`: Accepts a new feature extractors `clip-vit-b-32`, `vgg16`, `dinov2-vit-s-14`, `dinov2-vit-b-14`, `dinov2-vit-l-14`, `dinov2-vit-g-14`
    - `--feature-extractor-internal-dtype`: Allows to change the internal dtype used in the feature extractor's weights and activations; might be useful to counter numerical issues arising in fp32 implementations, e.g. those seen with the growth of the batch size
    - `--feature-extractor-compile`: Compile feature extractor (experimental: may have negative effect on the metrics numerical precision)
    - `--kid-kernel`: Allows choosing between the default `poly` (polynomial, default) and `rbf` (RBF) kernels 
    - `--kid-kernel-rbf-sigma`: Specifies RBF kernel sigma in KID  

### Changed in 0.4.0
- Default features for all metrics are now read from the selected feature extractor
- Default feature extractor is now inferred based on the selected metrics
- All tests run in docker now
- API
  - `calculate_metrics`
    - `kid_degree`: Deprecated, new name is `kid_kernel_poly_degree`
    - `kid_gamma`: Deprecated, new name is `kid_kernel_poly_gamma`
    - `kid_coef0`: Deprecated, new name is `kid_kernel_poly_coef0`
- Command line
    - `--kid-degree`: Deprecated, new name is `--kid-kernel-poly-degree`
    - `--kid-gamma`: Deprecated, new name is `--kid-kernel-poly-gamma`
    - `--kid-coef0`: Deprecated, new name is `--kid-kernel-poly-coef0`

### Fixed in 0.4.0
- [#19](https://github.com/toshas/torch-fidelity/issues/19): Adds Precision and Recall metrics
- [#42](https://github.com/toshas/torch-fidelity/issues/42): Fixes missing files in the wheel
- [#46](https://github.com/toshas/torch-fidelity/issues/46): Adds new FID computation code and removed scipy dependency
- [#47](https://github.com/toshas/torch-fidelity/issues/47): Adds new CLIP-based feature extractor

## [0.3.0] - 2021-06-08
### Added in 0.3.0
- API
  - `calculate_metrics`
    - `ppl`: Calculate PPL (Perceptual Path Length)
    - `ppl_epsilon`: Interpolation step size in PPL
    - `ppl_reduction`: Reduction type to apply to the per-sample output values
    - `ppl_sample_similarity`: Name of the sample similarity to use in PPL metric computation
    - `ppl_sample_similarity_resize`: Force samples to this size when computing similarity, unless set to None
    - `ppl_sample_similarity_dtype`: Check samples are of compatible dtype when computing similarity, unless set to None.
    - `ppl_discard_percentile_lower`: Removes the lower percentile of samples before reduction
    - `ppl_discard_percentile_higher`: Removes the higher percentile of samples before reduction
    - `ppl_z_interp_mode`: Noise interpolation mode in PPL
    - `input1_model_z_type`: Type of noise accepted by the input1 generator model
    - `input1_model_z_size`: Dimensionality of noise accepted by the input1 generator model
    - `input1_model_num_classes`: Number of classes for conditional generation (0 for unconditional) accepted by the input1 generator model
    - `input1_model_num_samples`: Number of samples to draw from input1 generator model, when it is provided as a path to ONNX model. This option affects the following metrics: ISC, FID, KID
    - `input2_model_z_type`: Type of noise accepted by the input2 generator model
    - `input2_model_z_size`: Dimensionality of noise accepted by the input2 generator model
    - `input2_model_num_classes`: Number of classes for conditional generation (0 for unconditional) accepted by the input2 generator model
    - `input2_model_num_samples`: Number of samples to draw from input2 generator model, when it is provided as a path to ONNX model. This option affects the following metrics: ISC, FID, KID
- Command line
  - `--ppl`: Calculate PPL (Perceptual Path Length)
  - `--ppl-epsilon`: Interpolation step size in PPL
  - `--ppl-reduction`: Reduction type to apply to the per-sample output values
  - `--ppl-sample-similarity`: Name of the sample similarity to use in PPL metric computation
  - `--ppl-sample-similarity-resize`: Force samples to this size when computing similarity, unless set to None 
  - `--ppl-sample-similarity-dtype`: Check samples are of compatible dtype when computing similarity, unless set to None.
  - `--ppl-discard-percentile-lower`: Removes the lower percentile of samples before reduction
  - `--ppl-discard-percentile-higher`: Removes the higher percentile of samples before reduction
  - `--ppl-z-interp-mode`: Noise interpolation mode in PPL
  - `--input1-model-z-type`: Type of noise accepted by the input1 generator model
  - `--input1-model-z-size`: Dimensionality of noise accepted by the input1 generator model
  - `--input1-model-num-classes`: Number of classes for conditional generation (0 for unconditional) accepted by the input1 generator model
  - `--input1-model-num-samples`: Number of samples to draw from input1 generator model, when it is provided as a path to ONNX model. This option affects the following metrics: ISC, FID, KID
  - `--input2-model-z-type`: Type of noise accepted by the input2 generator model
  - `--input2-model-z-size`: Dimensionality of noise accepted by the input2 generator model
  - `--input2-model-num-classes`: Number of classes for conditional generation (0 for unconditional) accepted by the input2 generator model
  - `--input2-model-num-samples`: Number of samples to draw from input2 generator model, when it is provided as a path to ONNX model. This option affects the following metrics: ISC, FID, KID
- Support generative model modules as inputs to all metrics  
- ONNX and PTH (JIT) model loading via command line functionality to support framework-agnostic metrics calculation
- Noise source types and latent vector interpolation methods can now be registered and dispatched similar to registered inputs
- Registered inputs: `stl10-train`, `stl10-test`, `stl10-unlabeled`
- Registered noise source types: `normal`, `uniform_0_1`, `unit` 
- Registered latent vector interpolation methods: `lerp`, `slerp_any`, `slerp` 
- Example SNGAN training and evaluation script (`examples/sngan_cifar10.py`)
- Test for LPIPS fidelity as compared to StyleGAN PyTorch implementation
- Test for feature extraction layer
- Unrecognized command line arguments warning
- Added ReadTheDocs documentation

### Changed in 0.3.0
- API
  - First input positional argument of `calculate_metrics` is now expected as a value to kwarg `input1`
  - Second input (optional) positional argument of `calculate_metrics` is now expected as a value to kwarg argument 
  `input2`
  - `cache_input1_name` renamed to `input1_cache_name`  
  - `cache_input2_name` renamed to `input2_cache_name`  
  - `rng_seed` default value from 2020 to 2021
- Command line
  - First input positional argument is now expected as a value to the key `--input1`
  - Second input (optional) positional argument is now expected as a value to the key `--input2`
  - `--datasets-downloaded` renamed to `--no-datasets-download`
  - `--samples-alphanumeric` renamed to `--no-samples-shuffle`
  - `--cache-input1-name` renamed to `--input1-cache-name`  
  - `--cache-input2-name` renamed to `--input2-cache-name`  
  - `--rng-seed` default value from 2020 to 2021
- Change `torch.save` to an atomic saving operation in all functions of the caching layer, which makes it 
  safe to use torch-fidelity in multiprocessing environment, such as a compute cluster with a shared file system.

### Fixed in 0.3.0
- [#15](https://github.com/toshas/torch-fidelity/issues/15): Fix '64', '192', and '768' feature layers usage in all metrics
- [#8](https://github.com/toshas/torch-fidelity/issues/8): Fix a missing exception for when KID subset size is larger than the number of samples in one of the inputs
- Fix a missing check that the elements of inputs are actually instances of `torch.Tensor`

## [0.2.0] - 2020-05-05
### Added in 0.2.0
- Initial release with Inception Score (ISC), Frechet Inception Distance (FID),
  and Kernel Inception Distance (KID) metrics
- Numerical precision unit tests for all three metrics
- Command line tool and Python API
