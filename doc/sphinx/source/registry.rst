Registry
========

The registry is a collection of input datasets, feature extractors, sample similarity measures, noise sources, and
interpolation sources. See :ref:`Extensibility` and :ref:`API` to learn how to add new items to the registry and how to use them
for metrics calculation. A number of entities have been pre-registered, and can be resolved in both CLI and API modes
by their names:

Preregistered inputs
--------------------

Can be used as values to :paramref:`~torch_fidelity.calculate_metrics.input1` and
:paramref:`~torch_fidelity.calculate_metrics.input2` arguments:

- `cifar10-train` - CIFAR-10 training split with 50000 images
- `cifar10-val` - CIFAR-10 validation split with 10000 images
- `cifar100-train` - CIFAR-100 training split with 50000 images
- `cifar100-val` - CIFAR-100 validation split with 10000 images
- `stl10-train` - STL-10 training split with 500 images
- `stl10-test` - STL-10 testing split with 800 images
- `stl10-unlabeled` - STL-10 unlabeled split with 100000 images

Preregistered feature extractors
--------------------------------

Can be used as values to the :paramref:`~torch_fidelity.calculate_metrics.feature_extractor` argument:

- `inception-v3-compat` - a standard InceptionV3 feature extractor from the original reference implementations of the
  Inception Score. This feature extractor is carefully ported to reproduce the original extractor's bilinear
  interpolation and neural architecture quirks.
- `vgg16` - a legacy VGG-based feature extractor used in the reference implementation of the Precision and Recall metrics.
- `clip-rn50`, `clip-rn101`, `clip-rn50x4`, `clip-rn50x16`, `clip-rn50x64`, `clip-vit-b-32`, `clip-vit-b-16`, `clip-vit-l-14`, `clip-vit-l-14-336px` - a set of modern CLIP-based feature extractors for evaluation of more realistic image generators, such as DDPMs.
- `dinov2-vit-s-14`, `dinov2-vit-b-14`, `dinov2-vit-l-14`, `dinov2-vit-g-14` - a set of modern self-supervised feature extractors,
  also suitable for state-of-the-art image generators evaluation.

Preregistered sample similarities
---------------------------------

Can be used as values to the :paramref:`~torch_fidelity.calculate_metrics.ppl_sample_similarity` argument:

- `lpips-vgg16` - a standard LPIPS sample similarity measure, based on a pre-trained VGG-16 and deep feature
  aggregation.

Preregistered noise source types
--------------------------------

Can be used as values to :paramref:`~torch_fidelity.calculate_metrics.input1_model_z_type` and
:paramref:`~torch_fidelity.calculate_metrics.input2_model_z_type` arguments:

- `normal` - standard normal distribution
- `unit` - uniform distribution on a unit sphere
- `uniform_0_1` - standard uniform distribution

Preregistered interpolation methods
-----------------------------------

Can be used as values to the :paramref:`~torch_fidelity.calculate_metrics.ppl_z_interp_mode` argument):

- `lerp` - linear interpolation
- `slerp_any` - spherical interpolation of `normal` samples
- `slerp_unit` - spherical interpolation of `unit` samples
