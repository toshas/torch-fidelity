Extensibility
=============

It is possible to implement and register a new input, feature extractor, sample similarity, noise
source type, or interpolation method before using using them in :func:`~torch_fidelity.calculate_metrics`:

Register a new input
--------------------

1. Subclass a new dataset (e.g., ``NewDataset``) from :class:`~torch:torch.utils.data.Dataset` class (refer to
   :class:`torch_fidelity.datasets.Cifar10_RGB` for an example),
2. Register it under some new name (`new-ds`):
   :func:`register_dataset('new-ds', lambda root, download: NewDataset(root, download)) <torch_fidelity.register_dataset>`,
3. Pass `"new-ds"` as a value of either :paramref:`~torch_fidelity.calculate_metrics.input1` or
   :paramref:`~torch_fidelity.calculate_metrics.input2` keyword arguments to :func:`~torch_fidelity.calculate_metrics`.

Register a new feature extractor
--------------------------------

1. Subclass a new feature extractor (e.g., ``NewFeatureExtractor``) from :class:`torch_fidelity.FeatureExtractorBase`
   class, implement all methods and properties,
2. Register it under some new name (`new-fe`):
   :func:`register_feature_extractor('new-fe', NewFeatureExtractor) <torch_fidelity.register_feature_extractor>`,
3. Pass `"new-fe"` as a value of :paramref:`~torch_fidelity.calculate_metrics.feature_extractor` keyword argument to
   :func:`~torch_fidelity.calculate_metrics`.

Register a new sample similarity measure
----------------------------------------

1. Subclass a new sample similarity (e.g., ``NewSampleSimilarity``) from :class:`torch_fidelity.SampleSimilarityBase`
   class, implement all methods and properties,
2. Register it under some new name (`new-ss`):
   :func:`register_sample_similarity('new-ss', NewSampleSimilarity) <torch_fidelity.register_sample_similarity>`,
3. Pass `"new-ss"` as a value of :paramref:`~torch_fidelity.calculate_metrics.ppl_sample_similarity` keyword argument to
   :func:`~torch_fidelity.calculate_metrics`.

Register a new noise source type
--------------------------------

1. Prepare a new function for drawing a sample from a multivariate distribution of a given shape, e.g.,
   ``def random_new(rng, shape): pass``,
2. Register it under some new name (`new-ns`):
   :func:`register_noise_source('new-ns', random_new) <torch_fidelity.register_noise_source>`,
3. Pass `"new-ns"` as a value of either :paramref:`~torch_fidelity.calculate_metrics.input1_model_z_type` or
   :paramref:`~torch_fidelity.calculate_metrics.input2_model_z_type` keyword arguments to
   :func:`~torch_fidelity.calculate_metrics`.

Register a new interpolation method
-----------------------------------

1. Prepare a new sample interpolation function, e.g., ``def new_interp(a, b, t): pass``,
2. Register it under some new name (`new-interp`):
   :func:`register_interpolation('new-interp', new_interp) <torch_fidelity.register_interpolation>`,
3. Pass `"new-interp"` as a value of :paramref:`~torch_fidelity.calculate_metrics.ppl_z_interp_mode` keyword arguments
   to :func:`~torch_fidelity.calculate_metrics`.
