import os
import subprocess
import sys
import tempfile
import unittest

from torch.hub import download_url_to_file

from tests import TimeTrackingTestCase
from torch_fidelity.helpers import json_decode_string
from torch_fidelity import KEY_METRIC_PPL_MEAN

URL_SNGAN_MODEL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/example-sngan-cifar10-generator.pth"
)


class TestMetricPplGenModel(TimeTrackingTestCase):
    @staticmethod
    def call_fidelity_ppl(input, nsamples):
        args = [
            # fmt: off
            "python3", "-m", "torch_fidelity.fidelity",
            "--ppl",
            "--json",
            "--save-cpu-ram",
            "--input1", input,
            "--input1-model-z-size", "128",
            "--input1-model-num-samples", str(nsamples),
            # fmt: on
        ]
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res

    def test_ppl_gen_model(self):
        model = os.path.basename(URL_SNGAN_MODEL)
        model = os.path.realpath(os.path.join(tempfile.gettempdir(), model))
        download_url_to_file(URL_SNGAN_MODEL, model, progress=True)
        self.assertTrue(os.path.isfile(model))

        print(f"Running fidelity PPL...", file=sys.stderr)
        res_fidelity = self.call_fidelity_ppl(model, 100)
        self.assertEqual(res_fidelity.returncode, 0, msg=res_fidelity)
        res_fidelity = json_decode_string(res_fidelity.stdout.decode())
        print("Fidelity PPL result:", res_fidelity, file=sys.stderr)

        self.assertAlmostEqual(res_fidelity[KEY_METRIC_PPL_MEAN], 2560.187255859375, delta=5)


if __name__ == "__main__":
    unittest.main()
