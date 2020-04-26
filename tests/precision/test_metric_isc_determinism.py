import os
import subprocess
import sys
import tempfile
import unittest

from tfdeterminism import patch as patch_tensorflow_for_determinism

from torch_fidelity.metric_isc import KEY_METRIC_ISC_MEAN, KEY_METRIC_ISC_STD
from torch_fidelity.utils import json_decode_string


class TestMetricIscDeterminism(unittest.TestCase):
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

    def test_isc_reference_determinism(self):
        cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''

        limit = 10000
        cifar10_root = os.path.join(tempfile.gettempdir(), f'cifar10-train-img-{limit}')

        res = subprocess.run(
            ('python3', 'utils/util_dump_dataset_as_images.py', 'cifar10-train', cifar10_root, str(limit)),
        )
        self.assertEqual(res.returncode, 0, msg=res)

        num_sample_runs = 2
        isc_mean_nondet, isc_std_nondet = [], []
        isc_mean_det, isc_std_det = [], []

        def sample_runs(isc_mean, isc_std, prefix):
            for i in range(num_sample_runs):
                print(f'{prefix} run {i+1} of {num_sample_runs}...', file=sys.stderr)
                res = self.call_ref_isc(cifar10_root, cuda, determinism=False)
                self.assertEqual(res.returncode, 0, msg=res)
                out = json_decode_string(res.stdout.decode())
                isc_mean.append(out[KEY_METRIC_ISC_MEAN])
                isc_std.append(out[KEY_METRIC_ISC_STD])

        sample_runs(isc_mean_nondet, isc_std_nondet, 'Non-deterministic')

        print('ENABLING TENSORFLOW DETERMINISM', file=sys.stderr)
        patch_tensorflow_for_determinism()

        sample_runs(isc_mean_det, isc_std_det, 'Deterministic')

        print('isc_mean_nondet', isc_mean_nondet, file=sys.stderr)
        print('isc_std_nondet', isc_std_nondet, file=sys.stderr)
        print('isc_mean_det', isc_mean_det, file=sys.stderr)
        print('isc_std_det', isc_std_det, file=sys.stderr)

        self.assertGreater(max(isc_mean_nondet), min(isc_mean_nondet))
        self.assertEqual(max(isc_mean_det), min(isc_mean_det))


if __name__ == '__main__':
    unittest.main()
