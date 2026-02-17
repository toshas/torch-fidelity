import os
import tempfile
import unittest
import warnings

import torch
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from tests import TimeTrackingTestCase
from torch_fidelity.utils import prepare_input_from_id, create_sample_similarity

# VGG16+LPIPS compiled module with pretrained weights
#   Distributed under NVIDIA Source Code License: https://nvlabs.github.io/stylegan2-ada-pytorch/license.html
URL_VGG16_LPIPS_STYLEGAN = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"


class LPIPS_reference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # NVIDIA's VGG16 checkpoint is a TorchScript archive, which triggers
        # warnings from torch.load; suppress since we load it intentionally.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.vgg16 = load_state_dict_from_url(URL_VGG16_LPIPS_STYLEGAN, file_name="vgg16_stylegan.pth")

    def forward(self, in0, in1):
        out0 = self.vgg16(in0, resize_images=False, return_lpips=True)
        out1 = self.vgg16(in1, resize_images=False, return_lpips=True)
        out = (out0 - out1).square().sum(dim=-1)
        return out


class TestLPIPS(TimeTrackingTestCase):
    @staticmethod
    def _get_sample(batch, index=0, size=None):
        ds = prepare_input_from_id("cifar10-val", datasets_root=tempfile.gettempdir())
        x = torch.cat([ds[i].unsqueeze(0) for i in range(index, index + batch)], dim=0)
        if size is not None:
            x = F.interpolate(x.float(), size=(size, size), mode="bicubic", align_corners=False).to(torch.uint8)
        return x

    def _test_lpips_raw(self, batch, size):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        x = self._get_sample(batch, 0, size)
        y = self._get_sample(batch, 5000, size)
        m_nv = LPIPS_reference()
        m_us = create_sample_similarity("lpips-vgg16", cuda=cuda)
        if cuda:
            x = x.cuda()
            y = y.cuda()
            m_nv.cuda()
        lpips_nv = m_nv(x, y)
        lpips_us = m_us(x, y)
        self.assertEqual(lpips_nv.shape, lpips_us.shape)
        l1 = (lpips_nv - lpips_us).abs().max().item()
        # cuDNN selects different convolution algorithms for different batch sizes,
        # producing slightly different floating-point results on CUDA
        threshold = 1e-4 if cuda else 1e-5
        self.assertLess(l1, threshold)

    def test_lpips_1_32(self):
        return self._test_lpips_raw(1, 32)

    def test_lpips_2_32(self):
        return self._test_lpips_raw(2, 32)

    def test_lpips_128_32(self):
        return self._test_lpips_raw(128, 32)

    def test_lpips_129_32(self):
        return self._test_lpips_raw(129, 32)

    def test_lpips_1024_32(self):
        return self._test_lpips_raw(1024, 32)

    def test_lpips_1_100(self):
        return self._test_lpips_raw(1, 100)

    def test_lpips_2_100(self):
        return self._test_lpips_raw(2, 100)

    def test_lpips_128_100(self):
        return self._test_lpips_raw(128, 100)

    def test_lpips_1_299(self):
        return self._test_lpips_raw(1, 299)

    def test_lpips_2_299(self):
        return self._test_lpips_raw(2, 299)


if __name__ == "__main__":
    unittest.main()
