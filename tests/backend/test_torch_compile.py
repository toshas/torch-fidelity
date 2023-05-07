import math
import unittest

import torch

from torch_fidelity import calculate_metrics, KEY_METRIC_ISC_MEAN
from torch_fidelity.datasets import RandomlyGeneratedDataset
from torch_fidelity.utils import create_feature_extractor


class TestTorchCompile(unittest.TestCase):

    def _test_torch_compile(self, fe, cuda):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        if int(torch.__version__.split('.')[0]) < 2 or not hasattr(torch, 'compile') or not callable(torch.compile):
            raise RuntimeError('compile function not resolved')

        feat = {
            'inception-v3-compat': '2048',
            'vgg16': 'fc2_relu',
            'clip-vit-b-32': 'clip',
        }[fe]

        dummy_feat_extractor = create_feature_extractor(
            fe, [feat], cuda=cuda, verbose=True, feature_extractor_compile=True
        )
        self.assertTrue(hasattr(dummy_feat_extractor, 'forward_pure'))

        input1 = RandomlyGeneratedDataset(128, 3, 299, 299, dtype=torch.uint8, seed=2023)
        metrics_normal = calculate_metrics(
            input1=input1, isc=True, isc_splits=2, verbose=True, cache=False,
            feature_extractor=fe, feature_extractor_compile=False, cuda=cuda
        )[KEY_METRIC_ISC_MEAN]
        metrics_compiled = calculate_metrics(
            input1=input1, isc=True, isc_splits=2, verbose=True, cache=False,
            feature_extractor=fe, feature_extractor_compile=True, cuda=cuda,
        )[KEY_METRIC_ISC_MEAN]

        discrepancy = math.fabs(metrics_normal - metrics_compiled)
        self.assertTrue(
            discrepancy < 1e-5,
            f'Compilation affects metrics outputs: {metrics_normal=}, {metrics_compiled=}',
        )

    def test_torch_compile_inceptionfe_cpu(self):
        self._test_torch_compile('inception-v3-compat', False)

    def test_torch_compile_inceptionfe_cuda(self):
        self._test_torch_compile('inception-v3-compat', True)

    def test_torch_compile_vgg16fe_cpu(self):
        self._test_torch_compile('vgg16', False)

    def test_torch_compile_vgg16fe_cuda(self):
        self._test_torch_compile('vgg16', True)

    def test_torch_compile_clipfe_cpu(self):
        self._test_torch_compile('clip-vit-b-32', False)

    def test_torch_compile_clipfe_cuda(self):
        self._test_torch_compile('clip-vit-b-32', True)


if __name__ == '__main__':
    unittest.main()
