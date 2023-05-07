torch-fidelity: High-fidelity performance metrics for generative models in PyTorch
==================================================================================

**torch-fidelity** provides **precise**, **efficient**, and **extensible** implementations of the popular metrics for
generative model evaluation, including:

* Inception Score (ISC_)
* Fréchet Inception Distance (FID_)
* Kernel Inception Distance (KID_)
* Perceptual Path Length (PPL_)
* Precision and Recall (PRC_)

.. _ISC: https://arxiv.org/pdf/1606.03498.pdf
.. _FID: https://arxiv.org/pdf/1706.08500.pdf
.. _KID: https://arxiv.org/pdf/1801.01401.pdf
.. _PPL: https://arxiv.org/pdf/1812.04948.pdf
.. _PRC: https://arxiv.org/pdf/1904.06991.pdf

**Numerical Precision**: Unlike many other reimplementations, the values produced by torch-fidelity match reference
implementations up to floating point's machine precision. This allows using torch-fidelity for reporting metrics in papers instead of
scattered and slow reference implementations.

**Efficiency**: Feature sharing between different metrics saves recomputation time, and an additional caching
level avoids recomputing features and statistics whenever possible. High efficiency allows using torch-fidelity in the
training loop, for example at the end of every epoch.

**Extensibility**: Going beyond 2D image generation is easy due to high modularity and abstraction of the metrics from
input data, models, and feature extractors. For example, one can swap out InceptionV3 feature extractor for a one
accepting 3D scan volumes, such as used in MRI.

**TLDR; fast and reliable GAN evaluation in PyTorch**

.. toctree::
   :maxdepth: 2
   :caption: Overview

   installation
   usage_cmd
   usage_api
   api
   registry
   extensibility
   miscellaneous
   precision
   changelog

Citation
--------

Citation is recommended to reinforce the evaluation protocol in works relying on torch-fidelity.
To ensure reproducibility, use the following BibTeX:

.. code-block:: bibtex

      @misc{obukhov2020torchfidelity,
        author={Anton Obukhov and Maximilian Seitzer and Po-Wei Wu and Semen Zhydenko and Jonathan Kyl and Elvis Yu-Jing Lin},
        year=2020,
        title={High-fidelity performance metrics for generative models in PyTorch},
        url={https://github.com/toshas/torch-fidelity},
        publisher={Zenodo},
        version={v0.3.0},
        doi={10.5281/zenodo.4957738},
        note={Version: 0.3.0, DOI: 10.5281/zenodo.4957738}
      }
