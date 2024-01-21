import json
import subprocess
import unittest
from pathlib import Path

from tests import TimeTrackingTestCase


class SmokeTests(TimeTrackingTestCase):
    def test_latest(self):
        path_base = Path(__file__).parent.parent.parent / "data"
        path_input1 = str(path_base / "cifar10-train-256")
        path_input2 = str(path_base / "cifar10-valid-256")

        res = subprocess.run(
            (
                # fmt: off
                "python3", "-m", "torch_fidelity.fidelity",
                "--input1", path_input1,
                "--input2", path_input2,
                "--isc",
                "--fid",
                "--kid",
                "--prc",
                "--kid-subset-size", "64",
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
        self.assertAlmostEqual(metrics["inception_score_mean"], 6.675409920681458, delta=1e-3)
        self.assertAlmostEqual(metrics["inception_score_std"], 0.9399683668381174, delta=1e-3)
        self.assertAlmostEqual(metrics["frechet_inception_distance"], 110.28082617202443, delta=1e-2)
        self.assertAlmostEqual(metrics["kernel_inception_distance_mean"], -0.0006792521855187905, delta=1e-4)
        self.assertAlmostEqual(metrics["kernel_inception_distance_std"], 0.0017778231588294379, delta=1e-4)
        self.assertAlmostEqual(metrics["precision"], 0.7109375, delta=1e-3)
        self.assertAlmostEqual(metrics["recall"], 0.71484375, delta=1e-3)
        self.assertAlmostEqual(metrics["f_score"], 0.7128852739726027, delta=1e-3)

        res = subprocess.run(
            (
                # fmt: off
                "python3", "-m", "torch_fidelity.fidelity",
                "--input1", path_input1,
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
        self.assertAlmostEqual(metrics["inception_score_mean"], 1.0322264848429967, delta=1e-3)
        self.assertAlmostEqual(metrics["inception_score_std"], 0.0012455888960011387, delta=1e-3)

        res = subprocess.run(
            (
                # fmt: off
                "python3", "-m", "torch_fidelity.fidelity",
                "--input1", path_input1,
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
        self.assertAlmostEqual(metrics["inception_score_mean"], 3.2955061842255757, delta=1e-3)
        self.assertAlmostEqual(metrics["inception_score_std"], 0.23961402932647136, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
