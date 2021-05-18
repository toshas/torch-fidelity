import unittest

import torch
import torch.nn.functional as F

from reference.reference_lpips import LPIPS_reference
from torch_fidelity.lpips import LPIPS_VGG16
from torch_fidelity.utils import prepare_inputs_as_datasets


class TestLPIPS(unittest.TestCase):
    @staticmethod
    def _get_sample(batch, size, index=0):
        ds = prepare_inputs_as_datasets('cifar10-val')
        x = torch.cat([ds[i].unsqueeze(0) for i in range(index, index+batch)], dim=0)
        x = F.interpolate(x.float(), size=(size, size), mode='bicubic', align_corners=False).to(torch.uint8)
        return x

    def _test_lpips(self, batch, size):
        x = self._get_sample(batch, size, 0)
        y = self._get_sample(batch, size, 5000)
        lpips_nv = LPIPS_reference()(x, y)
        lpips_us = LPIPS_VGG16()(x, y)
        self.assertEqual(lpips_nv.shape, lpips_us.shape)
        l1 = (lpips_nv - lpips_us).abs().max().item()
        self.assertLess(l1, 1e-5)

    def test_lpips_1_32(self):
        return self._test_lpips(1, 32)

    def test_lpips_2_32(self):
        return self._test_lpips(2, 32)

    def test_lpips_128_32(self):
        return self._test_lpips(128, 32)

    def test_lpips_129_32(self):
        return self._test_lpips(129, 32)

    def test_lpips_1024_32(self):
        return self._test_lpips(1024, 32)

    def test_lpips_1_100(self):
        return self._test_lpips(1, 100)

    def test_lpips_2_100(self):
        return self._test_lpips(2, 100)

    def test_lpips_128_100(self):
        return self._test_lpips(128, 100)

    def test_lpips_1_299(self):
        return self._test_lpips(1, 299)

    def test_lpips_2_299(self):
        return self._test_lpips(2, 299)


if __name__ == '__main__':
    unittest.main()
