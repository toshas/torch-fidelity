import os
import subprocess
import sys
import tempfile
import unittest

from tests import TimeTrackingTestCase
from torch_fidelity.helpers import json_decode_string
from torch_fidelity.metric_fid import KEY_METRIC_FID

from cleanfid import fid as cleanfid_fid


class TestMetricFidClipFidelity(TimeTrackingTestCase):
    @staticmethod
    def call_ref_fid(input_1, input_2, cuda):
        res = cleanfid_fid.compute_fid(
            input_1, input_2, mode="legacy_pytorch", model_name="clip_vit_b_32", device="cuda" if cuda else "cpu"
        )
        return res

    @staticmethod
    def call_fidelity_fid(input_1, input_2, cuda):
        args = [
            # fmt: off
            "python3", "-m", "torch_fidelity.fidelity",
            "--fid",
            "--json",
            "--save-cpu-ram",
            "--feature-extractor", "clip-vit-b-32",
            "--input1", input_1,
            "--input2", input_2,
            # fmt: on
        ]
        if cuda:
            args += ["-g", "0"]
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res

    def test_fid_pt_clean_fidelity(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        limit = 5000
        cifar10train_root = os.path.join(tempfile.gettempdir(), f"cifar10-train-img-{limit}")
        cifar10valid_root = os.path.join(tempfile.gettempdir(), f"cifar10-valid-img-{limit}")

        res = subprocess.run(
            (
                # fmt: off
                "python3", "utils/util_dump_dataset_as_images.py",
                "cifar10-train",
                cifar10train_root,
                "-l", str(limit),
                "-r", "224",
                # fmt: on
            ),
        )
        self.assertEqual(res.returncode, 0, msg=res)
        res = subprocess.run(
            (
                # fmt: off
                "python3", "utils/util_dump_dataset_as_images.py",
                "cifar10-valid",
                cifar10valid_root,
                "-l", str(limit),
                "-r", "224",
                # fmt: on
            ),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        print(f"Running reference FID...", file=sys.stderr)
        res_ref = self.call_ref_fid(cifar10train_root, cifar10valid_root, cuda)
        print("Reference FID result:", res_ref, file=sys.stderr)

        print(f"Running fidelity FID...", file=sys.stderr)
        res_fidelity = self.call_fidelity_fid(cifar10train_root, cifar10valid_root, cuda)
        self.assertEqual(res_fidelity.returncode, 0, msg=res_fidelity)
        res_fidelity = json_decode_string(res_fidelity.stdout.decode())
        print("Fidelity FID result:", res_fidelity, file=sys.stderr)

        err_abs = abs(res_ref - res_fidelity[KEY_METRIC_FID])
        print(f"Error absolute={err_abs}")

        err_rel = err_abs / res_ref
        print(f"Error relative={err_rel}")

        self.assertLess(err_rel, 1e-3)

        self.assertAlmostEqual(res_fidelity[KEY_METRIC_FID], 0.492375, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
