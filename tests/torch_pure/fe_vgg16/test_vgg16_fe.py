import unittest

import torch
from torch.hub import load_state_dict_from_url

from tests import TimeTrackingTestCase
from torch_fidelity.feature_extractor_vgg16 import FeatureExtractorVGG16


class TestVGG16FE(TimeTrackingTestCase):
    def _test_vgg16_fe_res(self, fe_us, fe_nv, res, seed=2023):
        rng = torch.Generator()
        rng.manual_seed(seed)
        x = torch.randint(256, (1, 3, res, res), dtype=torch.uint8, generator=rng)
        feat_us = fe_us(x)[0]
        feat_nv = fe_nv(x, return_features=True)
        diff = (feat_nv - feat_us).abs().max().item()
        self.assertLess(diff, 1e-4, f"res={res}")

    def test_vgg16_fe(self):
        url_nv = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        fe_us = FeatureExtractorVGG16("vgg16", ["fc2_relu"])
        fe_nv = load_state_dict_from_url(url_nv, map_location="cpu", progress=True)

        for i in (32, 64, 128, 256, 224, 512):
            self._test_vgg16_fe_res(fe_us, fe_nv, i)


if __name__ == "__main__":
    unittest.main()
