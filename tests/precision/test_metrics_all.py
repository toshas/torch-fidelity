import os
import subprocess
import tempfile
import unittest

import torch

from torch_fidelity import calculate_metrics
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

        limit = 5000
        input_1 = os.path.join(tempfile.gettempdir(), f'cifar10-train-img-{limit}')
        input_2 = os.path.join(tempfile.gettempdir(), f'cifar10-valid-img-noise-{limit}')

        res = subprocess.run(
            ('python3', 'utils/util_dump_dataset_as_images.py', 'cifar10-train', input_1,
             '-l', str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)
        res = subprocess.run(
            ('python3', 'utils/util_dump_dataset_as_images.py', 'cifar10-valid', input_2,
             '-l', str(limit), '-n'),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        kwargs = {
            'cuda': cuda,
            'cache_input1_name': 'test_input_1',
            'cache_input2_name': 'test_input_2',
            'save_cpu_ram': True,
        }

        all = calculate_metrics(input_1, input_2, isc=True, fid=True, kid=True, **kwargs)

        isc = calculate_isc(input_1, **kwargs)
        fid = calculate_fid(input_1, input_2, **kwargs)
        kid = calculate_kid(input_1, input_2, **kwargs)

        self.assertEqual(isc[KEY_METRIC_ISC_MEAN], all[KEY_METRIC_ISC_MEAN])
        self.assertEqual(fid[KEY_METRIC_FID], all[KEY_METRIC_FID])
        self.assertEqual(kid[KEY_METRIC_KID_MEAN], all[KEY_METRIC_KID_MEAN])


if __name__ == '__main__':
    unittest.main()
