import json
import os
import subprocess
import unittest

from tests import TimeTrackingTestCase


class SmokeTests(TimeTrackingTestCase):
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

        path_input1 = "cifar10-train-256"
        path_input2 = "cifar10-valid-256"

        try:
            print(f'Setting up torch={version_torch} torchvision={version_torchvision}')
            res = os.system(f'pip3 install -U torch{version_torch} torchvision{version_torchvision}')
            self.assertEqual(res, 0, msg=res)

            if not (os.path.isdir(path_input1) and os.path.isdir(path_input2)):
                res = os.system(f'python3 utils/util_dump_dataset_as_images.py cifar10-train {path_input1} -l 256')
                self.assertEqual(res, 0, msg=res)
                res = os.system(f'python3 utils/util_dump_dataset_as_images.py cifar10-valid {path_input2} -l 256')
                self.assertEqual(res, 0, msg=res)

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
            metrics = json.loads(res.stdout)
            print(f'RETCODE:\n{res.returncode}\nSTDOUT:\n{json.dumps(metrics, indent=4)}\nSTDERR:\n{res.stderr}\n')
            # self.assertEqual(res.returncode, 0, msg="Non-zero return code")
            # self.assertTrue('Warning' not in res.stdout, msg="Warning in stdout")
            # self.assertTrue('Warning' not in res.stderr, msg="Warning in stderr")
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
            metrics = json.loads(res.stdout)
            print(f'RETCODE:\n{res.returncode}\nSTDOUT:\n{json.dumps(metrics, indent=4)}\nSTDERR:\n{res.stderr}\n')
            # self.assertEqual(res.returncode, 0, msg="Non-zero return code")
            # self.assertTrue('Warning' not in res.stdout, msg="Warning in stdout")
            # self.assertTrue('Warning' not in res.stderr, msg="Warning in stderr")
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
        finally:
            print(f'Teardown')
            os.system(f'bash -c "pip3 uninstall -y torch torchvision"')

    def test__torch_latest__torchvision_latest(self):
        self._test_generic()


if __name__ == '__main__':
    unittest.main()
