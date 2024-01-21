import json
import os
import unittest

from tests import TimeTrackingTestCase


class TestMetricPplPrc(TimeTrackingTestCase):
    def test_ppl_prc(self):
        path_sg2ada = os.path.expanduser(os.path.expandvars("../workspace/torch-fidelity-stylegan2-ada"))
        res = os.system(f'bash -c "pip install -e . && cd {path_sg2ada} && sh run_fidelity_comparison_reduced.sh"')
        self.assertEqual(res, 0)

        def _read_metric(name, field):
            with open(os.path.join(path_sg2ada, f"metric-{name}.jsonl")) as fp:
                line = fp.readlines()[-1]
            return json.loads(line)["results"][field]

        ppl_original = _read_metric("ppl_zend_cifar10_original_epsexp_m4_dtype_u8", "ppl_zend")
        ppl_fidelity = _read_metric("ppl_zend_cifar10_fidelity_epsexp_m4_dtype_u8", "ppl_zend")
        prc_original_precision = _read_metric("pr50k3_original", "pr50k3_precision")
        prc_original_recall = _read_metric("pr50k3_original", "pr50k3_recall")
        prc_fidelity_precision = _read_metric("pr50k3_fidelity", "pr50k3_precision")
        prc_fidelity_recall = _read_metric("pr50k3_fidelity", "pr50k3_recall")

        rel_ppl = abs(ppl_original - ppl_fidelity) / ppl_original
        diff_precision = abs(prc_original_precision - prc_fidelity_precision)
        diff_recall = abs(prc_original_recall - prc_fidelity_recall)

        print(f"ppl: orig={ppl_original} fidelity={ppl_fidelity} reldiff={rel_ppl}")
        print(f"precision: orig={prc_original_precision} fidelity={prc_fidelity_precision} absdiff={diff_precision}")
        print(f"recall: orig={prc_original_recall} fidelity={prc_fidelity_recall} absdiff={diff_recall}")

        self.assertLess(rel_ppl, 1e-2)
        self.assertLess(diff_precision, 1e-2)
        self.assertLess(diff_recall, 1e-2)


if __name__ == "__main__":
    unittest.main()
