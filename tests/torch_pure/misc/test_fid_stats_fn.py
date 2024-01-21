import unittest
import warnings

import numpy as np
import scipy.linalg
import torch

from tests import TimeTrackingTestCase
from torch_fidelity.helpers import vprint
from torch_fidelity.metric_fid import fid_statistics_to_metric as impl_lib, KEY_METRIC_FID, fid_features_to_statistics


class TestFIDStatsFunction(TimeTrackingTestCase):
    @staticmethod
    def impl_ref_torch_fidelity_less_equal_ver_0_3_0(stat_1, stat_2, verbose):
        eps = 1e-6

        mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
        mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
        assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
        assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            vprint(
                verbose,
                f"WARNING: fid calculation produces singular product; adding {eps} to diagonal of cov estimates",
            )
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=verbose)

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                vprint(verbose, "WARNING: imaginary component {}".format(m))
                covmean = np.array([[np.nan]])
            else:
                covmean = covmean.real

        tr_covmean = np.trace(covmean)

        out = {
            KEY_METRIC_FID: float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean),
        }

        vprint(verbose, f"Frechet Inception Distance: {out[KEY_METRIC_FID]}")

        return out

    @staticmethod
    def impl_buggy_sometimes_nan(stat_1, stat_2, verbose):
        # for a short period this was used after merging https://github.com/toshas/torch-fidelity/pull/50
        # this was a numpy adaptation of the proposed torch code.
        # the bug was introduced because unlike torch.linalg.eigvals which always returns complex, np.linalg.eigvals
        # returns real if there is no complex part, and then the subsequent sqrt fails.
        mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
        mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
        assert mu1.ndim == 1 and mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
        assert sigma1.ndim == 2 and sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

        diff = mu1 - mu2

        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

        tr_covmean = np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2))).real)
        fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

        out = {
            KEY_METRIC_FID: fid,
        }

        vprint(verbose, f"Frechet Inception Distance: {out[KEY_METRIC_FID]}")

        return out

    def _test_fid_stat(self, stat_1, stat_2, msg, verbose):
        out_lib = impl_lib(stat_1, stat_2, verbose=verbose)[KEY_METRIC_FID]
        self.assertEqual(out_lib, out_lib, msg=msg)  # check that the library does not return NaN
        out_ref = self.impl_ref_torch_fidelity_less_equal_ver_0_3_0(stat_1, stat_2, verbose=verbose)[KEY_METRIC_FID]
        if not np.isnan(out_ref):
            self.assertAlmostEqual(out_lib, out_ref, places=5, msg=msg)

    def _test_fid_stat_expect_nan_with_impl_buggy(self, stat_1, stat_2, verbose):
        out_buggy_nan = self.impl_buggy_sometimes_nan(stat_1, stat_2, verbose=verbose)[KEY_METRIC_FID]
        self.assertNotEqual(out_buggy_nan, out_buggy_nan)

    @staticmethod
    def make_stat(N, C, seed):
        rng = np.random.RandomState(seed)
        feat_1 = rng.randn(N, C)
        feat_2 = rng.randn(N, C)
        stat_1 = fid_features_to_statistics(torch.tensor(feat_1))
        stat_2 = fid_features_to_statistics(torch.tensor(feat_2))
        return stat_1, stat_2

    @staticmethod
    def make_stat_for_buggy_nan_impl():
        return TestFIDStatsFunction.make_stat(4, 4, 1)

    def test_fid(self):
        verbose = False
        seed = 2023

        stat_1, stat_2 = self.make_stat_for_buggy_nan_impl()
        self._test_fid_stat_expect_nan_with_impl_buggy(stat_1, stat_2, verbose)

        for N_log2 in range(1, 16):
            N = 2**N_log2
            for C_log2 in range(1, 9):
                C = 2**C_log2
                msg = f"seed={seed} N={N} C={C}"
                stat_1, stat_2 = self.make_stat(N, C, seed)
                self._test_fid_stat(stat_1, stat_2, msg, verbose)


if __name__ == "__main__":
    unittest.main()
