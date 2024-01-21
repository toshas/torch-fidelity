import unittest

import torch

from tests import TimeTrackingTestCase
from torch_fidelity.feature_extractor_dinov2 import FeatureExtractorDinoV2


class TestDINOv2FE(TimeTrackingTestCase):
    def test_dinov2_fe(self):
        fe_us = FeatureExtractorDinoV2("dinov2-vit-b-14", ["dinov2"])
        resolution = 32
        seed = 2023
        rng = torch.Generator()
        rng.manual_seed(seed)
        x = torch.randint(256, (1, 3, resolution, resolution), dtype=torch.uint8, generator=rng)
        feat_us = fe_us(x)[0]
        checksum = feat_us.sum().item()
        expected = -34.6249580383
        self.assertAlmostEqual(checksum, expected, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
