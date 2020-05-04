import os
import subprocess
import sys
import tempfile
import unittest

from torch_fidelity.helpers import json_decode_string
from torch_fidelity.metric_kid import KEY_METRIC_KID_MEAN, KEY_METRIC_KID_STD


class TestMetricKidFidelity(unittest.TestCase):
    @staticmethod
    def call_ref_kid(input_1, input_2, cuda, determinism):
        args = [
            'python3', 'tests/reference/reference_metric_kid_mmdgan.py', input_1, input_2,
            '--no-inception', '--do-mmd'
        ]
        if cuda:
            args.append('--gpu')
            args.append(os.environ['CUDA_VISIBLE_DEVICES'])
        if determinism:
            args.append('--determinism')
        args.append('--json')
        args.append('--silent')
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=None)
        return res

    @staticmethod
    def call_fidelity_kid(input_1, input_2):
        args = ['python3', '-m', 'torch_fidelity.fidelity', '--kid', '--json', '--save-cpu-ram', input_1, input_2]
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res

    def test_kid_pt_tf_fidelity(self):
        cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
        limit = 5000
        cifar10trainorig_root = os.path.join(tempfile.gettempdir(), f'cifar10-train-img-{limit}')
        cifar10validnoise_root = os.path.join(tempfile.gettempdir(), f'cifar10-valid-img-noise-{limit}')
        cifar10trainorig_codes = os.path.join(tempfile.gettempdir(), f'cifar10-train-codes-{limit}.pth')

        res = subprocess.run(
            ('python3', 'utils/util_dump_dataset_as_images.py', 'cifar10-train', cifar10trainorig_root,
             '-l', str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)
        res = subprocess.run(
            ('python3', 'utils/util_dump_dataset_as_images.py', 'cifar10-valid', cifar10validnoise_root,
             '-l', str(limit), '-n'),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        # this partucular reference app needs features pre-extracted
        if not os.path.exists(cifar10trainorig_codes):
            args = [
                'python3', 'tests/reference/reference_metric_kid_mmdgan.py', cifar10trainorig_root,
                '--save-codes', cifar10trainorig_codes, '--determinism'
            ]
            if cuda:
                args.append('--gpu')
                args.append(os.environ['CUDA_VISIBLE_DEVICES'])
            res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(res.returncode, 0, msg=res)

        print(f'Running reference KID...', file=sys.stderr)
        res_ref = self.call_ref_kid(cifar10validnoise_root, cifar10trainorig_codes, cuda, determinism=True)
        self.assertEqual(res_ref.returncode, 0, msg=res_ref)
        res_ref = json_decode_string(res_ref.stdout.decode())
        print('Reference KID result:', res_ref, file=sys.stderr)

        print(f'Running fidelity KID...', file=sys.stderr)
        res_fidelity = self.call_fidelity_kid(cifar10validnoise_root, cifar10trainorig_root)
        self.assertEqual(res_fidelity.returncode, 0, msg=res_fidelity)
        res_fidelity = json_decode_string(res_fidelity.stdout.decode())
        print('Fidelity KID result:', res_fidelity, file=sys.stderr)

        err_abs_mean = abs(res_ref[KEY_METRIC_KID_MEAN] - res_fidelity[KEY_METRIC_KID_MEAN])
        err_abs_std = abs(res_ref[KEY_METRIC_KID_STD] - res_fidelity[KEY_METRIC_KID_STD])
        print(f'Error absolute mean={err_abs_mean} std={err_abs_std}')

        err_rel_mean = err_abs_mean / max(1e-6, abs(res_ref[KEY_METRIC_KID_MEAN]))
        err_rel_std = err_abs_std / res_ref[KEY_METRIC_KID_STD]
        print(f'Error relative mean={err_rel_mean} std={err_rel_std}')

        self.assertLess(err_rel_mean, 1e-6)
        self.assertLess(err_rel_std, 1e-4)

        self.assertAlmostEqual(res_fidelity[KEY_METRIC_KID_MEAN], 0.4718520, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
