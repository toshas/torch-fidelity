import os
import subprocess
import sys
import tempfile
import unittest

from torch_fidelity.metric_isc import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD
from torch_fidelity.utils import json_decode_string


class TestMetricIscFidelity(unittest.TestCase):
    @staticmethod
    def call_ref_isc(input, cuda, determinism):
        args = ['python3', 'tests/reference/reference_metric_isc_ttur.py', input]
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
    def call_fidelity_isc(input):
        args = ['python3', '-m', 'torch_fidelity.fidelity', '--isc', '--json', input]
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res

    def test_isc_pt_tf_fidelity(self):
        cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''
        cifar10_root = os.path.join(tempfile.gettempdir(), 'cifar10-train-img')

        res = subprocess.run(
            ('python3', 'utils/util_dump_dataset_as_images.py', 'cifar10-train', cifar10_root),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        print(f'Running reference ISC...', file=sys.stderr)
        res_ref = self.call_ref_isc(cifar10_root, cuda, determinism=True)
        self.assertEqual(res_ref.returncode, 0, msg=res_ref)
        res_ref = json_decode_string(res_ref.stdout.decode())
        print('Reference ISC result:', res_ref, file=sys.stderr)

        print(f'Running fidelity ISC cached...', file=sys.stderr)
        res_fidelity = self.call_fidelity_isc('cifar10-train')
        self.assertEqual(res_fidelity.returncode, 0, msg=res_fidelity)
        res_fidelity = json_decode_string(res_fidelity.stdout.decode())
        print('Fidelity ISC result:', res_fidelity, file=sys.stderr)

        self.assertAlmostEqual(res_ref[KEY_METRIC_ISC_MEAN], res_fidelity[KEY_METRIC_ISC_MEAN], delta=1e-3)
        self.assertAlmostEqual(res_ref[KEY_METRIC_ISC_STD], res_fidelity[KEY_METRIC_ISC_STD], delta=5e-1)
        self.assertLess(res_fidelity[KEY_METRIC_ISC_STD], res_ref[KEY_METRIC_ISC_STD])

        self.assertAlmostEqual(res_fidelity[KEY_METRIC_ISC_MEAN], 11.236778, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
