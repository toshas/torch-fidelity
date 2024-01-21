import unittest

import torch

from tests import TimeTrackingTestCase
from torch_fidelity import calculate_metrics
from torch_fidelity.datasets import RandomlyGeneratedDataset
from torch_fidelity.metric_fid import KEY_METRIC_FID


class TestFeatureLayer(TimeTrackingTestCase):
    def _test_fid_feature_layer(self, feature_size):
        input1 = RandomlyGeneratedDataset(10, 3, 299, 299, dtype=torch.uint8, seed=2021)
        input2 = RandomlyGeneratedDataset(10, 3, 299, 299, dtype=torch.uint8, seed=2022)
        metrics = calculate_metrics(
            input1=input1, input2=input2, fid=True, feature_layer_fid=feature_size, verbose=False
        )
        self.assertTrue(metrics[KEY_METRIC_FID] > 0)

    def test_fid_feature_layer_64_numeric(self):
        with self.assertRaises(ValueError) as e:
            self._test_fid_feature_layer(64)

    def test_fid_feature_layer_65(self):
        with self.assertRaises(ValueError) as e:
            self._test_fid_feature_layer("65")

    def test_fid_feature_layer_64(self):
        self._test_fid_feature_layer("64")

    def test_fid_feature_layer_192(self):
        self._test_fid_feature_layer("192")

    def test_fid_feature_layer_768(self):
        self._test_fid_feature_layer("768")

    def test_fid_feature_layer_2048(self):
        self._test_fid_feature_layer("2048")


if __name__ == "__main__":
    unittest.main()
