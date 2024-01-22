import os
import subprocess
import sys
import tempfile
import unittest

from tests import TimeTrackingTestCase
from torch_fidelity.helpers import json_decode_string
from torch_fidelity.metric_fid import KEY_METRIC_FID


class TestMetricFidDeterminism(TimeTrackingTestCase):
    @staticmethod
    def call_ref_fid(input_1, input_2, cuda, determinism):
        args = ["python3", "tests/tf1/reference/reference_metric_fid_ttur.py", input_1, input_2]
        if cuda:
            args.append("--gpu")
            args.append(os.environ["CUDA_VISIBLE_DEVICES"])
        if determinism:
            args.append("--determinism")
        args.append("--json")
        args.append("--silent")
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=None)
        return res

    def test_fid_reference_determinism(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""

        limit = 5000
        cifar10train_root = os.path.join(tempfile.gettempdir(), f"cifar10-train-img-{limit}")
        cifar10valid_root = os.path.join(tempfile.gettempdir(), f"cifar10-valid-img-{limit}")

        res = subprocess.run(
            ("python3", "utils/util_dump_dataset_as_images.py", "cifar10-train", cifar10train_root, "-l", str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)
        res = subprocess.run(
            ("python3", "utils/util_dump_dataset_as_images.py", "cifar10-valid", cifar10valid_root, "-l", str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        num_sample_runs = 2
        fid_nondet, fid_det = [], []

        def sample_runs(fid, prefix, determinism):
            for i in range(num_sample_runs):
                print(f"{prefix} run {i+1} of {num_sample_runs}...", file=sys.stderr)
                res = self.call_ref_fid(cifar10train_root, cifar10valid_root, cuda, determinism)
                self.assertEqual(res.returncode, 0, msg=res)
                out = json_decode_string(res.stdout.decode())
                fid.append(out[KEY_METRIC_FID])

        sample_runs(fid_nondet, "Non-deterministic", False)
        sample_runs(fid_det, "Deterministic", True)

        print("fid_nondet", fid_nondet, file=sys.stderr)
        print("fid_det", fid_det, file=sys.stderr)

        if cuda:
            self.assertGreater(min(fid_nondet) + 1e-4, max(fid_nondet))
        else:
            self.assertGreaterEqual(max(fid_nondet), min(fid_nondet))
        self.assertEqual(max(fid_det), min(fid_det))


if __name__ == "__main__":
    unittest.main()
