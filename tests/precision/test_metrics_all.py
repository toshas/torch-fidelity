import os
import tempfile
import unittest

import torch
from torchvision.transforms import Compose

from torch_fidelity import calculate_metrics
from torch_fidelity.datasets import Cifar10_RGB, TransformPILtoRGBTensor
from torch_fidelity.metric_fid import calculate_fid, KEY_METRIC_FID
from torch_fidelity.metric_isc import calculate_isc, KEY_METRIC_ISC_MEAN
from torch_fidelity.metric_kid import calculate_kid, KEY_METRIC_KID_MEAN


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

        isc = calculate_isc(input_1, cuda=cuda)
        fid = calculate_fid(input_1, input_2, cuda=cuda)
        kid = calculate_kid(input_1, input_2, cuda=cuda)

        all = calculate_metrics(input_1, input_2, cuda=cuda, isc=True, fid=True, kid=True)

        self.assertEqual(isc[KEY_METRIC_ISC_MEAN], all[KEY_METRIC_ISC_MEAN])
        self.assertEqual(fid[KEY_METRIC_FID], all[KEY_METRIC_FID])
        self.assertEqual(kid[KEY_METRIC_KID_MEAN], all[KEY_METRIC_KID_MEAN])


if __name__ == '__main__':
    unittest.main()
