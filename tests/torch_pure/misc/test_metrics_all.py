import os
import subprocess
import tempfile
import unittest

import torch

from tests import TimeTrackingTestCase
from torch_fidelity import calculate_metrics
from torch_fidelity.metric_fid import calculate_fid, KEY_METRIC_FID
from torch_fidelity.metric_isc import calculate_isc, KEY_METRIC_ISC_MEAN
from torch_fidelity.metric_kid import calculate_kid, KEY_METRIC_KID_MEAN
from torch_fidelity.metric_prc import calculate_prc, KEY_METRIC_PRECISION, KEY_METRIC_RECALL


class TransformAddNoise:
    def __call__(self, img):
        assert torch.is_tensor(img)
        img = img.float()
        img += torch.randn_like(img) * 64
        img = img.to(torch.uint8)
        return img


class TestMetricsAll(TimeTrackingTestCase):
    def test_all(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""

        limit = 5000
        input_1 = os.path.join(tempfile.gettempdir(), f"cifar10-train-img-{limit}")
        input_2 = os.path.join(tempfile.gettempdir(), f"cifar10-valid-img-{limit}")

        res = subprocess.run(
            ("python3", "utils/util_dump_dataset_as_images.py", "cifar10-train", input_1, "-l", str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)
        res = subprocess.run(
            ("python3", "utils/util_dump_dataset_as_images.py", "cifar10-valid", input_2, "-l", str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        kwargs = {
            "cuda": cuda,
            "input1_cache_name": "test_input_1",
            "input2_cache_name": "test_input_2",
            "save_cpu_ram": True,
        }

        all = calculate_metrics(input1=input_1, input2=input_2, isc=True, fid=True, kid=True, prc=True, **kwargs)

        self.assertGreater(all[KEY_METRIC_ISC_MEAN], 0)
        self.assertGreater(all[KEY_METRIC_FID], 0)
        self.assertGreater(all[KEY_METRIC_PRECISION], 0)
        self.assertGreater(all[KEY_METRIC_RECALL], 0)

        isc = calculate_isc(1, **kwargs)
        fid = calculate_fid(**kwargs)
        kid = calculate_kid(**kwargs)
        prc = calculate_prc(**kwargs)

        self.assertEqual(isc[KEY_METRIC_ISC_MEAN], all[KEY_METRIC_ISC_MEAN])
        self.assertEqual(fid[KEY_METRIC_FID], all[KEY_METRIC_FID])
        self.assertEqual(kid[KEY_METRIC_KID_MEAN], all[KEY_METRIC_KID_MEAN])
        self.assertEqual(prc[KEY_METRIC_PRECISION], all[KEY_METRIC_PRECISION])
        self.assertEqual(prc[KEY_METRIC_RECALL], all[KEY_METRIC_RECALL])


if __name__ == "__main__":
    unittest.main()
