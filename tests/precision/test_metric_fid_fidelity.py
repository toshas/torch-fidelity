import os
import subprocess
import sys
import tempfile
import unittest

from torch_fidelity.metric_fid import KEY_METRIC_FID
from torch_fidelity.utils import json_decode_string


class TestMetricFidFidelity(unittest.TestCase):
    @staticmethod
    def call_ref_fid(input_1, input_2, cuda, determinism):
        args = ['python3', 'tests/reference/reference_metric_fid_ttur.py', input_1, input_2]
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
    def call_fidelity_fid(input_1, input_2):
        args = ['python3', '-m', 'torch_fidelity.fidelity', '--fid', '--json', '--save-cpu-ram', input_1, input_2]
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res

    def test_fid_pt_tf_fidelity(self):
        cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
        limit = 5000
        cifar10train_root = os.path.join(tempfile.gettempdir(), f'cifar10-train-img-{limit}')
        cifar10valid_root = os.path.join(tempfile.gettempdir(), f'cifar10-valid-img-{limit}')

        res = subprocess.run(
            ('python3', 'utils/util_dump_dataset_as_images.py', 'cifar10-train', cifar10train_root,
             '-l', str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)
        res = subprocess.run(
            ('python3', 'utils/util_dump_dataset_as_images.py', 'cifar10-valid', cifar10valid_root,
             '-l', str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        print(f'Running reference FID...', file=sys.stderr)
        res_ref = self.call_ref_fid(cifar10train_root, cifar10valid_root, cuda, determinism=True)
        self.assertEqual(res_ref.returncode, 0, msg=res_ref)
        res_ref = json_decode_string(res_ref.stdout.decode())
        print('Reference FID result:', res_ref, file=sys.stderr)

        print(f'Running fidelity FID...', file=sys.stderr)
        res_fidelity = self.call_fidelity_fid(cifar10train_root, cifar10valid_root)
        self.assertEqual(res_fidelity.returncode, 0, msg=res_fidelity)
        res_fidelity = json_decode_string(res_fidelity.stdout.decode())
        print('Fidelity FID result:', res_fidelity, file=sys.stderr)

        err_abs = abs(res_ref[KEY_METRIC_FID] - res_fidelity[KEY_METRIC_FID])
        print(f'Error absolute={err_abs}')

        err_rel = err_abs / res_ref[KEY_METRIC_FID]
        print(f'Error relative={err_rel}')

        self.assertLess(err_rel, 1e-6)

        self.assertAlmostEqual(res_fidelity[KEY_METRIC_FID], 10.3233274, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
