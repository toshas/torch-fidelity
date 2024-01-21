import os
import subprocess
import sys
import tempfile
import unittest

from tests import TimeTrackingTestCase
from torch_fidelity.helpers import json_decode_string
from torch_fidelity.metric_kid import KEY_METRIC_KID_MEAN, KEY_METRIC_KID_STD


class TestMetricKidDeterminism(TimeTrackingTestCase):
    @staticmethod
    def call_ref_kid(input_1, input_2, cuda, determinism):
        args = [
            # fmt: off
            "python3", "tests/tf1/reference/reference_metric_kid_mmdgan.py",
            input_1,
            input_2,
            "--no-inception",
            "--do-mmd",
            # fmt: on
        ]
        if cuda:
            args.append("--gpu")
            args.append(os.environ["CUDA_VISIBLE_DEVICES"])
        if determinism:
            args.append("--determinism")
        args.append("--json")
        args.append("--silent")
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=None)
        return res

    def test_kid_reference_determinism(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        limit = 5000
        cifar10trainorig_root = os.path.join(tempfile.gettempdir(), f"cifar10-train-img-{limit}")
        cifar10validnoise_root = os.path.join(tempfile.gettempdir(), f"cifar10-valid-img-noise-{limit}")
        cifar10trainorig_codes = os.path.join(tempfile.gettempdir(), f"cifar10-train-img-codes-{limit}.pth")

        res = subprocess.run(
            (
                # fmt: off
                "python3", "utils/util_dump_dataset_as_images.py",
                "cifar10-train",
                cifar10trainorig_root,
                "-l", str(limit),
                # fmt: on
            ),
        )
        self.assertEqual(res.returncode, 0, msg=res)
        res = subprocess.run(
            (
                # fmt: off
                "python3", "utils/util_dump_dataset_as_images.py",
                "cifar10-valid",
                cifar10validnoise_root,
                "-l", str(limit),
                "-n",
                # fmt: on
            ),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        # this partucular reference app needs features pre-extracted
        if not os.path.exists(cifar10trainorig_codes):
            args = [
                # fmt: off
                "python3", "tests/tf1/reference/reference_metric_kid_mmdgan.py",
                cifar10trainorig_root,
                "--save-codes", cifar10trainorig_codes,
                "--determinism",
                # fmt: on
            ]
            if cuda:
                args.append("--gpu")
                args.append(os.environ["CUDA_VISIBLE_DEVICES"])
            res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(res.returncode, 0, msg=res)

        num_sample_runs = 2
        kid_mean_nondet, kid_std_nondet = [], []
        kid_mean_det, kid_std_det = [], []

        def sample_runs(kid_mean, kid_std, prefix, determinism):
            for i in range(num_sample_runs):
                print(f"{prefix} run {i+1} of {num_sample_runs}...", file=sys.stderr)
                res = self.call_ref_kid(cifar10validnoise_root, cifar10trainorig_codes, cuda, determinism)
                self.assertEqual(res.returncode, 0, msg=res)
                res = json_decode_string(res.stdout.decode())
                kid_mean.append(res[KEY_METRIC_KID_MEAN])
                kid_std.append(res[KEY_METRIC_KID_STD])

        sample_runs(kid_mean_nondet, kid_std_nondet, "Non-deterministic", False)
        sample_runs(kid_mean_det, kid_std_det, "Deterministic", True)

        print("kid_mean_nondet", kid_mean_nondet, file=sys.stderr)
        print("kid_std_nondet", kid_std_nondet, file=sys.stderr)
        print("kid_mean_det", kid_mean_det, file=sys.stderr)
        print("kid_std_det", kid_std_det, file=sys.stderr)

        if cuda:
            self.assertGreater(min(kid_mean_nondet) + 1e-5, max(kid_mean_nondet))
        else:
            self.assertGreaterEqual(max(kid_mean_nondet), min(kid_mean_nondet))
        self.assertEqual(max(kid_mean_det), min(kid_mean_det))


if __name__ == "__main__":
    unittest.main()
