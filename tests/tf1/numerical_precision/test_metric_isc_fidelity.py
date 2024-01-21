import os
import subprocess
import sys
import tempfile
import unittest

from tests import TimeTrackingTestCase
from torch_fidelity.helpers import json_decode_string
from torch_fidelity.metric_isc import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD


class TestMetricIscFidelity(TimeTrackingTestCase):
    @staticmethod
    def call_ref_isc(input, cuda, determinism):
        args = ["python3", "tests/tf1/reference/reference_metric_isc_ttur.py", input]
        if cuda:
            args.append("--gpu")
            args.append(os.environ["CUDA_VISIBLE_DEVICES"])
        if determinism:
            args.append("--determinism")
        args.append("--json")
        args.append("--silent")
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=None)
        return res

    @staticmethod
    def call_fidelity_isc(input):
        args = ["python3", "-m", "torch_fidelity.fidelity", "--isc", "--json", "--save-cpu-ram", "--input1", input]
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res

    def test_isc_pt_tf_fidelity(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        limit = 100
        cifar10_root = os.path.join(tempfile.gettempdir(), f"cifar10-train-img-{limit}")

        res = subprocess.run(
            ("python3", "utils/util_dump_dataset_as_images.py", "cifar10-train", cifar10_root, "-l", str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        print(f"Running reference ISC...", file=sys.stderr)
        res_ref = self.call_ref_isc(cifar10_root, cuda, determinism=True)
        self.assertEqual(res_ref.returncode, 0, msg=res_ref)
        res_ref = json_decode_string(res_ref.stdout.decode())
        print("Reference ISC result:", res_ref, file=sys.stderr)

        print(f"Running fidelity ISC...", file=sys.stderr)
        res_fidelity = self.call_fidelity_isc(cifar10_root)
        self.assertEqual(res_fidelity.returncode, 0, msg=res_fidelity)
        res_fidelity = json_decode_string(res_fidelity.stdout.decode())
        print("Fidelity ISC result:", res_fidelity, file=sys.stderr)

        err_abs_mean = abs(res_ref[KEY_METRIC_ISC_MEAN] - res_fidelity[KEY_METRIC_ISC_MEAN])
        err_abs_std = abs(res_ref[KEY_METRIC_ISC_STD] - res_fidelity[KEY_METRIC_ISC_STD])
        print(f"Error absolute mean={err_abs_mean} std={err_abs_std}")

        err_rel_mean = err_abs_mean / res_ref[KEY_METRIC_ISC_MEAN]
        err_rel_std = err_abs_std / res_ref[KEY_METRIC_ISC_STD]
        print(f"Error relative mean={err_rel_mean} std={err_rel_std}")

        self.assertLess(err_rel_mean, 5e-3)
        self.assertLess(err_rel_std, 5e-1)

        self.assertAlmostEqual(res_fidelity[KEY_METRIC_ISC_MEAN], 4.51089784587268, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
