API
===

Here you can find description of functions and their keyword arguments.

torch_fidelity module
---------------------

.. automodule:: torch_fidelity
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: calculate_metrics

.. autofunction:: register_dataset
.. autofunction:: register_feature_extractor
.. autofunction:: register_sample_similarity
.. autofunction:: register_noise_source
.. autofunction:: register_interpolation

.. autoclass:: FeatureExtractorBase
.. autoclass:: FeatureExtractorInceptionV3
.. autoclass:: FeatureExtractorVGG16
.. autoclass:: FeatureExtractorCLIP
.. autoclass:: FeatureExtractorDinoV2

.. autoclass:: GenerativeModelBase
.. autoclass:: GenerativeModelModuleWrapper
.. autoclass:: GenerativeModelONNX

.. autoclass:: SampleSimilarityBase
.. autoclass:: SampleSimilarityLPIPS
