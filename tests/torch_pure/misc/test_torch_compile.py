import unittest

import torch

from tests import TimeTrackingTestCase
from torch_fidelity import calculate_metrics, KEY_METRIC_ISC_MEAN
from torch_fidelity.datasets import RandomlyGeneratedDataset


class TestTorchCompile(TimeTrackingTestCase):
    def _test_torch_compile(self, fe, cuda):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if int(torch.__version__.split(".")[0]) < 2 or not hasattr(torch, "compile") or not callable(torch.compile):
            raise RuntimeError("compile function not resolved")

        input1 = RandomlyGeneratedDataset(128, 3, 299, 299, dtype=torch.uint8, seed=2023)
        metrics_normal = calculate_metrics(
            input1=input1,
            isc=True,
            isc_splits=2,
            verbose=True,
            cache=False,
            feature_extractor=fe,
            feature_extractor_compile=False,
            cuda=cuda,
        )[KEY_METRIC_ISC_MEAN]
        metrics_compiled = calculate_metrics(
            input1=input1,
            isc=True,
            isc_splits=2,
            verbose=True,
            cache=False,
            feature_extractor=fe,
            feature_extractor_compile=True,
            cuda=cuda,
        )[KEY_METRIC_ISC_MEAN]

        self.assertAlmostEqual(
            metrics_normal,
            metrics_compiled,
            delta=1e-5,
            msg=f"Compilation affects metrics outputs: normal={metrics_normal}, compiled={metrics_compiled}",
        )

    def test_torch_compile_inceptionfe_cpu(self):
        self._test_torch_compile("inception-v3-compat", False)

    def test_torch_compile_inceptionfe_cuda(self):
        self._test_torch_compile("inception-v3-compat", True)

    def test_torch_compile_vgg16fe_cpu(self):
        self._test_torch_compile("vgg16", False)

    def test_torch_compile_vgg16fe_cuda(self):
        self._test_torch_compile("vgg16", True)

    def test_torch_compile_clipfe_cpu(self):
        self._test_torch_compile("clip-vit-b-32", False)

    def test_torch_compile_clipfe_cuda(self):
        self._test_torch_compile("clip-vit-b-32", True)

    def test_torch_compile_dinov2sfe_cpu(self):
        self._test_torch_compile("dinov2-vit-s-14", False)

    def test_torch_compile_dinov2sfe_cuda(self):
        self._test_torch_compile("dinov2-vit-s-14", True)

    def test_torch_compile_dinov2bfe_cpu(self):
        self._test_torch_compile("dinov2-vit-b-14", False)

    def test_torch_compile_dinov2bfe_cuda(self):
        self._test_torch_compile("dinov2-vit-b-14", True)

    def test_torch_compile_dinov2lfe_cpu(self):
        self._test_torch_compile("dinov2-vit-l-14", False)

    def test_torch_compile_dinov2lfe_cuda(self):
        self._test_torch_compile("dinov2-vit-l-14", True)

    def test_torch_compile_dinov2gfe_cpu(self):
        self._test_torch_compile("dinov2-vit-g-14", False)

    def test_torch_compile_dinov2gfe_cuda(self):
        self._test_torch_compile("dinov2-vit-g-14", True)


if __name__ == "__main__":
    unittest.main()
