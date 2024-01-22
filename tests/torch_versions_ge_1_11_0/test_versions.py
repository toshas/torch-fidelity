import json
import os
import subprocess
import unittest

from tests import TimeTrackingTestCase


class TestVersions(TimeTrackingTestCase):
    def test_no_preinstalled(self):
        try:
            import torch
        except ImportError:
            self.assertTrue(True)
            return
        self.assertTrue(False, f"torch version detected: {torch.__version__}")

    def _test_generic(self, version_torch=None, version_torchvision=None):
        if version_torch is not None:
            version_torch = "==" + version_torch
        else:
            version_torch = ""
        if version_torchvision is not None:
            version_torchvision = "==" + version_torchvision
        else:
            version_torchvision = ""
        try:
            print(f"Setting up torch={version_torch} torchvision={version_torchvision}")
            res = os.system(f'bash -c "pip3 install -U torch{version_torch} torchvision{version_torchvision}"')
            self.assertEqual(res, 0, msg=res)

            res = subprocess.run(
                (
                    # fmt: off
                    "python3", "-m", "torch_fidelity.fidelity",
                    "--input1", "/tmp/cifar10-train-5000",
                    "--input2", "/tmp/cifar10-valid-5000",
                    "--gpu", "0",
                    "--isc",
                    "--fid",
                    "--kid",
                    "--prc",
                    "--silent",
                    "--json",
                    # fmt: on
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"RETCODE:\n{res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n")
            self.assertEqual(res.returncode, 0, msg="Non-zero return code")
            self.assertTrue("Warning" not in res.stdout, msg="Warning in stdout")
            self.assertTrue("Warning" not in res.stderr, msg="Warning in stderr")
            metrics = json.loads(res.stdout)
            self.assertAlmostEqual(metrics["inception_score_mean"], 10.75051, delta=1e-4)
            self.assertAlmostEqual(metrics["inception_score_std"], 0.5778723, delta=1e-4)
            self.assertAlmostEqual(metrics["frechet_inception_distance"], 10.32333, delta=1e-4)
            self.assertAlmostEqual(metrics["kernel_inception_distance_mean"], -2.907863e-05, delta=1e-7)
            self.assertAlmostEqual(metrics["kernel_inception_distance_std"], 0.0001023118, delta=1e-7)
            self.assertAlmostEqual(metrics["precision"], 0.6908, delta=1e-3)
            self.assertAlmostEqual(metrics["recall"], 0.6852, delta=1e-3)
            self.assertAlmostEqual(metrics["f_score"], 0.6879886, delta=1e-3)

            res = subprocess.run(
                (
                    # fmt: off
                    "python3", "-m", "torch_fidelity.fidelity",
                    "--input1", "/tmp/cifar10-train-5000",
                    "--gpu", "0",
                    "--isc",
                    "--silent",
                    "--json",
                    "--feature-extractor", "clip-vit-b-32",
                    # fmt: on
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"RETCODE:\n{res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n")
            self.assertEqual(res.returncode, 0, msg="Non-zero return code")
            self.assertTrue("Warning" not in res.stdout, msg="Warning in stdout")
            self.assertTrue("Warning" not in res.stderr, msg="Warning in stderr")
            metrics = json.loads(res.stdout)
            self.assertAlmostEqual(metrics["inception_score_mean"], 1.034257746757046, delta=1e-5)
            self.assertAlmostEqual(metrics["inception_score_std"], 0.00041031304234824675, delta=1e-8)

            res = subprocess.run(
                (
                    # fmt: off
                    "python3", "-m", "torch_fidelity.fidelity",
                    "--input1", "/tmp/cifar10-train-5000",
                    "--gpu", "0",
                    "--isc",
                    "--silent",
                    "--json",
                    "--feature-extractor", "dinov2-vit-b-14",
                    # fmt: on
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"RETCODE:\n{res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n")
            self.assertEqual(res.returncode, 0, msg="Non-zero return code")
            self.assertTrue("Warning" not in res.stdout, msg="Warning in stdout")
            self.assertTrue("Warning" not in res.stderr, msg="Warning in stderr")
            metrics = json.loads(res.stdout)
            self.assertAlmostEqual(metrics["inception_score_mean"], 3.701889394966352, delta=1e-4)
            self.assertAlmostEqual(metrics["inception_score_std"], 0.051992151563281505, delta=1e-5)
        finally:
            print(f"Teardown")
            os.system(f'bash -c "pip3 uninstall -y torch torchvision ; rm -rf ~/.cache/torch/hub"')

    def test__torch_latest__torchvision_latest(self):
        self._test_generic()

    def test__torch_2_1_1__torchvision_0_16_1(self):
        self._test_generic("2.1.1", "0.16.1")

    def test__torch_2_0_1__torchvision_0_15_2(self):
        self._test_generic("2.0.1", "0.15.2")

    def test__torch_1_13_1__torchvision_0_14_1(self):
        self._test_generic("1.13.1", "0.14.1")

    def test__torch_1_12_1__torchvision_0_13_1(self):
        self._test_generic("1.12.1", "0.13.1")

    def test__torch_1_11_0__torchvision_0_12_0(self):
        self._test_generic("1.11.0", "0.12.0")


if __name__ == "__main__":
    unittest.main()
