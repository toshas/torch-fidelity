import math
import unittest

import torch

from tests import TimeTrackingTestCase
from torch_fidelity.metric_mind import KEY_METRIC_MIND, mind_features_to_metric


class TestMind(TimeTrackingTestCase):
    """Unit tests for the Monge Inception Distance.

    For two iso-Gaussian samples X~N(0,I_d), Y~N(mu*1_d, I_d) projected onto a unit direction u,
    u^T X ~ N(0,1) and u^T Y ~ N(u^T(mu*1_d), 1), so W_2^2 between the two 1D distributions equals
    (u^T mu*1_d)^2. Averaging over u~U(S^{d-1}) gives E[(u^T mu*1_d)^2] = ||mu*1_d||^2/d = mu^2.
    With alpha = 3d, expected MIND ~ 3d * mu^2 in the large-sample large-M limit.
    """

    @staticmethod
    def _gaussian(n, d, mean=0.0, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(n, d, generator=g) + mean

    def test_identity_is_zero(self):
        X = self._gaussian(1024, 32, seed=1)
        out = mind_features_to_metric(X, X, mind_num_projections=256, cuda=False, rng_seed=2020, verbose=False)
        self.assertEqual(out[KEY_METRIC_MIND], 0.0)

    def test_symmetric_under_input_swap(self):
        X = self._gaussian(1024, 32, seed=1)
        Y = self._gaussian(1024, 32, mean=0.5, seed=2)
        out_xy = mind_features_to_metric(X, Y, mind_num_projections=256, cuda=False, rng_seed=2020, verbose=False)
        out_yx = mind_features_to_metric(Y, X, mind_num_projections=256, cuda=False, rng_seed=2020, verbose=False)
        self.assertEqual(out_xy[KEY_METRIC_MIND], out_yx[KEY_METRIC_MIND])

    def test_reproducible_with_same_seed(self):
        X = self._gaussian(1024, 32, seed=1)
        Y = self._gaussian(1024, 32, mean=0.5, seed=2)
        a = mind_features_to_metric(X, Y, mind_num_projections=256, cuda=False, rng_seed=2020, verbose=False)
        b = mind_features_to_metric(X, Y, mind_num_projections=256, cuda=False, rng_seed=2020, verbose=False)
        self.assertEqual(a[KEY_METRIC_MIND], b[KEY_METRIC_MIND])

    def test_closed_form_gaussian_shift(self):
        # For samples large enough and M large enough, MIND should approach 3d*mu^2 for an
        # isotropic shift of magnitude mu per coordinate. The expected finite-sample error from
        # the difference of empirical means scales as ~2*mu/sqrt(n) per coordinate, giving a
        # ~3% relative discrepancy at n=4000, mu=1.0 even in the infinite-M limit; we allow 10%.
        d = 64
        for mu in (1.0, 3.0):
            X = self._gaussian(4000, d, seed=11)
            Y = self._gaussian(4000, d, mean=mu, seed=22)
            out = mind_features_to_metric(
                X, Y, mind_num_projections=2000, cuda=False, rng_seed=2020, verbose=False
            )
            expected = 3.0 * d * mu * mu
            rel_err = math.fabs(out[KEY_METRIC_MIND] - expected) / expected
            self.assertLess(rel_err, 0.10, f"MIND {out[KEY_METRIC_MIND]} far from expected {expected} for mu={mu}")

    def test_monotone_in_shift(self):
        # Larger distribution shift => larger MIND.
        d = 32
        X = self._gaussian(1024, d, seed=1)
        Y_small = self._gaussian(1024, d, mean=0.5, seed=2)
        Y_big = self._gaussian(1024, d, mean=2.0, seed=2)
        m_small = mind_features_to_metric(X, Y_small, mind_num_projections=512, cuda=False, rng_seed=0, verbose=False)
        m_big = mind_features_to_metric(X, Y_big, mind_num_projections=512, cuda=False, rng_seed=0, verbose=False)
        self.assertGreater(m_big[KEY_METRIC_MIND], m_small[KEY_METRIC_MIND])

    def test_unequal_sample_sizes(self):
        # Should run and produce a finite, non-negative value.
        X = self._gaussian(2000, 16, seed=1)
        Y = self._gaussian(1200, 16, mean=1.0, seed=2)
        out = mind_features_to_metric(X, Y, mind_num_projections=256, cuda=False, rng_seed=2020, verbose=False)
        self.assertTrue(math.isfinite(out[KEY_METRIC_MIND]))
        self.assertGreaterEqual(out[KEY_METRIC_MIND], 0.0)


if __name__ == "__main__":
    unittest.main()
