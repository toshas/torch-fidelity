import math
import unittest

import torch

from torch_fidelity import calculate_metrics, KEY_METRIC_ISC_MEAN
from torch_fidelity.datasets import RandomlyGeneratedDataset


class TestBatchSizeIndependence(unittest.TestCase):

    def _test_batch_size_independence(self, fe, num_samples, dtype, cuda):
        if cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        input1 = RandomlyGeneratedDataset(num_samples, 3, 299, 299, dtype=torch.uint8, seed=2023)
        metrics_b_1 = calculate_metrics(
            input1=input1, isc=True, isc_splits=2, verbose=True, cache=False,
            feature_extractor=fe, feature_extractor_internal_dtype=dtype, batch_size=1, cuda=cuda
        )[KEY_METRIC_ISC_MEAN]
        metrics_b_all = calculate_metrics(
            input1=input1, isc=True, isc_splits=2, verbose=True, cache=False,
            feature_extractor=fe, feature_extractor_internal_dtype=dtype, batch_size=num_samples, cuda=cuda
        )[KEY_METRIC_ISC_MEAN]
        discrepancy = math.fabs(metrics_b_1 - metrics_b_all)
        self.assertTrue(
            discrepancy < 1e-5,
            f'Batch size affects metrics outputs: size_1 gives {metrics_b_1}, size_all gives {metrics_b_all}',
        )

    def test_batch_size_independence_inceptionfe_4_fp32_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 4, 'float32', False)

    def test_batch_size_independence_inceptionfe_8_fp32_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 8, 'float32', False)

    def test_batch_size_independence_inceptionfe_16_fp32_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 16, 'float32', False)

    def test_batch_size_independence_inceptionfe_32_fp32_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 32, 'float32', False)

    def test_batch_size_independence_inceptionfe_64_fp32_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 64, 'float32', False)

    def test_batch_size_independence_inceptionfe_128_fp32_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 128, 'float32', False)

    def test_batch_size_independence_inceptionfe_4_fp32_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 4, 'float32', True)

    def test_batch_size_independence_inceptionfe_8_fp32_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 8, 'float32', True)

    def test_batch_size_independence_inceptionfe_16_fp32_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 16, 'float32', True)

    def test_batch_size_independence_inceptionfe_32_fp32_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 32, 'float32', True)

    def test_batch_size_independence_inceptionfe_64_fp32_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 64, 'float32', True)

    def test_batch_size_independence_inceptionfe_128_fp32_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 128, 'float32', True)

    def test_batch_size_independence_inceptionfe_4_fp64_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 4, 'float64', False)

    def test_batch_size_independence_inceptionfe_8_fp64_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 8, 'float64', False)

    def test_batch_size_independence_inceptionfe_16_fp64_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 16, 'float64', False)

    def test_batch_size_independence_inceptionfe_32_fp64_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 32, 'float64', False)

    def test_batch_size_independence_inceptionfe_64_fp64_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 64, 'float64', False)

    def test_batch_size_independence_inceptionfe_128_fp64_cpu(self):
        self._test_batch_size_independence('inception-v3-compat', 128, 'float64', False)

    def test_batch_size_independence_inceptionfe_4_fp64_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 4, 'float64', True)

    def test_batch_size_independence_inceptionfe_8_fp64_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 8, 'float64', True)

    def test_batch_size_independence_inceptionfe_16_fp64_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 16, 'float64', True)

    def test_batch_size_independence_inceptionfe_32_fp64_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 32, 'float64', True)

    def test_batch_size_independence_inceptionfe_64_fp64_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 64, 'float64', True)

    def test_batch_size_independence_inceptionfe_128_fp64_cuda(self):
        self._test_batch_size_independence('inception-v3-compat', 128, 'float64', True)

    def test_batch_size_independence_clipfe_4_fp32_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 4, 'float32', False)

    def test_batch_size_independence_clipfe_8_fp32_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 8, 'float32', False)

    def test_batch_size_independence_clipfe_16_fp32_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 16, 'float32', False)

    def test_batch_size_independence_clipfe_32_fp32_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 32, 'float32', False)

    def test_batch_size_independence_clipfe_64_fp32_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 64, 'float32', False)

    def test_batch_size_independence_clipfe_128_fp32_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 128, 'float32', False)

    def test_batch_size_independence_clipfe_4_fp32_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 4, 'float32', True)

    def test_batch_size_independence_clipfe_8_fp32_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 8, 'float32', True)

    def test_batch_size_independence_clipfe_16_fp32_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 16, 'float32', True)

    def test_batch_size_independence_clipfe_32_fp32_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 32, 'float32', True)

    def test_batch_size_independence_clipfe_64_fp32_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 64, 'float32', True)

    def test_batch_size_independence_clipfe_128_fp32_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 128, 'float32', True)

    def test_batch_size_independence_clipfe_4_fp64_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 4, 'float64', False)

    def test_batch_size_independence_clipfe_8_fp64_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 8, 'float64', False)

    def test_batch_size_independence_clipfe_16_fp64_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 16, 'float64', False)

    def test_batch_size_independence_clipfe_32_fp64_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 32, 'float64', False)

    def test_batch_size_independence_clipfe_64_fp64_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 64, 'float64', False)

    def test_batch_size_independence_clipfe_128_fp64_cpu(self):
        self._test_batch_size_independence('clip-vit-b-32', 128, 'float64', False)

    def test_batch_size_independence_clipfe_4_fp64_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 4, 'float64', True)

    def test_batch_size_independence_clipfe_8_fp64_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 8, 'float64', True)

    def test_batch_size_independence_clipfe_16_fp64_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 16, 'float64', True)

    def test_batch_size_independence_clipfe_32_fp64_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 32, 'float64', True)

    def test_batch_size_independence_clipfe_64_fp64_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 64, 'float64', True)

    def test_batch_size_independence_clipfe_128_fp64_cuda(self):
        self._test_batch_size_independence('clip-vit-b-32', 128, 'float64', True)


if __name__ == '__main__':
    unittest.main()
