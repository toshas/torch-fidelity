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
                'python3', '-m', 'torch_fidelity.fidelity',
                    '--input1', path_input1,
                    '--input2', path_input2,
                    '--isc', '--fid', '--kid', '--prc',
                    '--silent',
                    '--json',
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f'RETCODE:\n{res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n')
        # self.assertEqual(res.returncode, 0, msg="Non-zero return code")
        # self.assertTrue('Warning' not in res.stdout, msg="Warning in stdout")
        # self.assertTrue('Warning' not in res.stderr, msg="Warning in stderr")
        # metrics = json.loads(res.stdout)
        # self.assertAlmostEqual(metrics['inception_score_mean'], 10.75051, delta=1e-4)
        # self.assertAlmostEqual(metrics['inception_score_std'], 0.5778723, delta=1e-4)
        # self.assertAlmostEqual(metrics['frechet_inception_distance'], 10.32333, delta=1e-4)
        # self.assertAlmostEqual(metrics['kernel_inception_distance_mean'], -2.907863e-05, delta=1e-7)
        # self.assertAlmostEqual(metrics['kernel_inception_distance_std'], 0.0001023118, delta=1e-7)
        # self.assertAlmostEqual(metrics['precision'], 0.6908, delta=1e-3)
        # self.assertAlmostEqual(metrics['recall'], 0.6852, delta=1e-3)
        # self.assertAlmostEqual(metrics['f_score'], 0.6879886, delta=1e-3)

        res = subprocess.run(
            (
                'python3', '-m', 'torch_fidelity.fidelity',
                    '--input1', path_input1,
                    '--gpu', '0',
                    '--isc',
                    '--silent',
                    '--json',
                    '--feature-extractor', 'clip-vit-b-32',
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f'RETCODE:\n{res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n')
        # self.assertEqual(res.returncode, 0, msg="Non-zero return code")
        # self.assertTrue('Warning' not in res.stdout, msg="Warning in stdout")
        # self.assertTrue('Warning' not in res.stderr, msg="Warning in stderr")
        # metrics = json.loads(res.stdout)
        # self.assertAlmostEqual(metrics['inception_score_mean'], 1.034257746757046, delta=1e-5)
        # self.assertAlmostEqual(metrics['inception_score_std'], 0.00041031304234824675, delta=1e-8)

        res = subprocess.run(
            (
                'python3', '-m', 'torch_fidelity.fidelity',
                    '--input1', path_input1,
                    '--gpu', '0',
                    '--isc',
                    '--silent',
                    '--json',
                    '--feature-extractor', 'dinov2-vit-b-14',
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f'RETCODE:\n{res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n')
        # self.assertEqual(res.returncode, 0, msg="Non-zero return code")
        # self.assertTrue('Warning' not in res.stdout, msg="Warning in stdout")
        # self.assertTrue('Warning' not in res.stderr, msg="Warning in stderr")
        # metrics = json.loads(res.stdout)
        # self.assertAlmostEqual(metrics['inception_score_mean'], 3.701889394966352, delta=1e-4)
        # self.assertAlmostEqual(metrics['inception_score_std'], 0.051992151563281505, delta=1e-5)


if __name__ == '__main__':
    unittest.main()
