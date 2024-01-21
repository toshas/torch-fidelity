import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from tests import TimeTrackingTestCase
from tests.tf1.reference.reference_lpips import LPIPS_reference
from torch_fidelity.utils import prepare_input_from_id, create_sample_similarity


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
        self.assertLess(l1, 1e-5)

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
