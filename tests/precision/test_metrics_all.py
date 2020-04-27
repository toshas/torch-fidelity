import os
import tempfile
import unittest

import torch
from torchvision.transforms import Compose

from torch_fidelity import *
from torch_fidelity.datasets import Cifar10_RGB, TransformPILtoRGBTensor


class TransformAddNoise:
    def __call__(self, img):
        assert torch.is_tensor(img)
        img = img.float()
        img += torch.randn_like(img) * 64
        img = img.to(torch.uint8)
        return img


class TestMetricsAll(unittest.TestCase):
    def test_all(self):
        cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''

        input_1 = 'cifar10-train'
        input_2 = Cifar10_RGB(tempfile.gettempdir(), train=True, transform=Compose((
            TransformPILtoRGBTensor(),
            TransformAddNoise()
        )), download=True)
        input_2.name = None

        kwargs = default_kwargs()
        kwargs['cuda'] = cuda
        kwargs['isc'] = True
        kwargs['fid'] = True
        kwargs['kid'] = True

        isc = calculate_isc(input_1, **kwargs)
        fid = calculate_fid(input_1, input_2, **kwargs)
        kid = calculate_kid(input_1, input_2, **kwargs)
        all = calculate_metrics(input_1, input_2, **kwargs)

        self.assertEqual(isc[KEY_METRIC_ISC_MEAN], all[KEY_METRIC_ISC_MEAN])
        self.assertEqual(fid[KEY_METRIC_FID], all[KEY_METRIC_FID])
        self.assertEqual(kid[KEY_METRIC_KID_MEAN], all[KEY_METRIC_KID_MEAN])


if __name__ == '__main__':
    unittest.main()
