import os

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from tqdm import tqdm

from utils import glob_image_paths, get_features

KEY_METRIC_KID_MEAN = 'kernel_inception_distance_mean'
KEY_METRIC_KID_STD = 'kernel_inception_distance_std'


def kid_features_to_metric(
        features_1, features_2, n_subsets=50, subset_size=1000, rng_seed=2020,
        degree=3, gamma=None, coef0=1
):
    mmds = np.zeros(n_subsets)
    rng = np.random.RandomState(rng_seed)

    for i in tqdm(range(n_subsets)):
        f1 = features_1[rng.choice(len(features_1), subset_size, replace=False)]
        f2 = features_2[rng.choice(len(features_2), subset_size, replace=False)]
        o = polynomial_mmd(f1, f2, degree=degree, gamma=gamma, coef0=coef0)
        mmds[i] = o

    return {
        KEY_METRIC_KID_MEAN: float(np.mean(mmds)),
        KEY_METRIC_KID_STD: float(np.std(mmds)),
    }

def polynomial_mmd(features_1, features_2, degree=3, gamma=None, coef0=1):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = features_1
    Y = features_2

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return mmd2(K_XX, K_XY, K_YY)


def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased'):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    return mmd2






from feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

KID_INCEPTION_FEATURES = '2048'

# kid_features_to_metric(
#         features_1, features_2, n_subsets=50, subset_size=1000, rng_seed=2020,
#         degree=3, gamma=None, coef0=1
# )

def kid_listimages_to_metric(files, model, batch_size=50, cuda=False, verbose=True):
    features = get_features(files, model, batch_size, cuda, verbose)
    features = features[KID_INCEPTION_FEATURES]
    return kid_features_to_metric(features)


def kid_path_to_statistics(path, glob_recursively, model, batch_size, cuda, verbose):
    if path.endswith('.npz'):
        stat = fid_cache_to_statistics(path)
    else:
        files = glob_image_paths(path, glob_recursively, verbose)
        stat = fid_listimages_to_statistics(files, model, batch_size, cuda, verbose)
    return stat


def kid_create_model(cuda=True, model_weights_path=None):
    model = FeatureExtractorInceptionV3(
        [KID_INCEPTION_FEATURES],
        normalize_input=False,
        inception_weights_path=model_weights_path
    )
    model.eval()
    if cuda:
        model.cuda()
    return model



def kid_paths_to_metric(paths, glob_recursively, batch_size, cuda, model_weights_path, verbose):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    model = kid_create_model(cuda, model_weights_path)

    stat_1 = fid_path_to_statistics(paths[0], glob_recursively, model, batch_size, cuda, verbose)
    stat_2 = fid_path_to_statistics(paths[1], glob_recursively, model, batch_size, cuda, verbose)
    metric = fid_statistics_to_metric(stat_1, stat_2, verbose)

    return metric
