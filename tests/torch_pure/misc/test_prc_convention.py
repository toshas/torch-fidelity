import unittest

import torch

from tests import TimeTrackingTestCase
from torch_fidelity.metric_prc import calculate_precision_recall_full, calculate_precision_recall_part


class TestPrcConvention(TimeTrackingTestCase):
    """Smoke tests verifying precision/recall convention: features_1=generated, features_2=real.

    Uses a mode-collapse scenario: generated=tight cluster, real=wide spread.
    Precision (fraction of generated in real manifold) should be HIGH.
    Recall (fraction of real in generated manifold) should be LOW.
    """

    @staticmethod
    def _make_asymmetric_features(seed=42):
        rng = torch.Generator().manual_seed(seed)
        features_gen = torch.randn(200, 16, generator=rng) * 0.1
        features_real = torch.randn(200, 16, generator=rng) * 3.0
        return features_gen, features_real

    def test_full_precision_high_recall_low(self):
        gen, real = self._make_asymmetric_features()
        precision, recall = calculate_precision_recall_full(gen, real)
        self.assertGreater(precision, 0.8, f"Precision {precision} should be high (generated cluster inside real manifold)")
        self.assertLess(recall, 0.3, f"Recall {recall} should be low (real manifold not covered by tight cluster)")

    def test_part_precision_high_recall_low(self):
        gen, real = self._make_asymmetric_features()
        precision, recall = calculate_precision_recall_part(gen, real, batch_size=50)
        self.assertGreater(precision, 0.8, f"Precision {precision} should be high (generated cluster inside real manifold)")
        self.assertLess(recall, 0.3, f"Recall {recall} should be low (real manifold not covered by tight cluster)")

    def test_full_vs_part_equal(self):
        gen, real = self._make_asymmetric_features()
        p_full, r_full = calculate_precision_recall_full(gen, real)
        p_part, r_part = calculate_precision_recall_part(gen, real, batch_size=50)
        self.assertAlmostEqual(p_full, p_part, places=6, msg="Precision mismatch between full and part")
        self.assertAlmostEqual(r_full, r_part, places=6, msg="Recall mismatch between full and part")


if __name__ == "__main__":
    unittest.main()
