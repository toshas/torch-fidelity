import numpy as np
import torch
from sklearn.metrics.pairwise import polynomial_kernel
from tqdm import tqdm

from utils import create_feature_extractor, extract_featuresdict_from_input_cached

KEY_METRIC_KID_MEAN = 'kernel_inception_distance_mean'
KEY_METRIC_KID_STD = 'kernel_inception_distance_std'


def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased'):
    # based on https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # changed to not compute the full kernel matrix at once
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
        assert mmd_est in ('unbiased', 'u-statistic')
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    return mmd2


def polynomial_mmd(features_1, features_2, degree, gamma, coef0):
    X = features_1
    Y = features_2
    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    return mmd2(K_XX, K_XY, K_YY)


def kid_features_to_metric(features_1, features_2, **kwargs):
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2

    features_1 = features_1.cpu().numpy()
    features_2 = features_2.cpu().numpy()

    kid_subsets = kwargs['kid_subsets']
    kid_subset_size = kwargs['kid_subset_size']

    mmds = np.zeros(kid_subsets)
    rng = np.random.RandomState(kwargs['rng_seed'])

    for i in tqdm(range(kid_subsets)):
        f1 = features_1[rng.choice(len(features_1), kid_subset_size, replace=False)]
        f2 = features_2[rng.choice(len(features_2), kid_subset_size, replace=False)]
        o = polynomial_mmd(f1, f2, degree=kwargs['kid_degree'], gamma=kwargs['kid_gamma'], coef0=kwargs['kid_coef0'])
        mmds[i] = o

    return {
        KEY_METRIC_KID_MEAN: float(np.mean(mmds)),
        KEY_METRIC_KID_STD: float(np.std(mmds)),
    }


def kid_alone(input_1, input_2, **kwargs):
    feat_layer_name = kwargs['feature_layer_kid']
    feat_extractor = create_feature_extractor(kwargs['feature_extractor'], [feat_layer_name], **kwargs)

    featuresdict_1 = extract_featuresdict_from_input_cached(input_1, feat_extractor, **kwargs)
    featuresdict_2 = extract_featuresdict_from_input_cached(input_2, feat_extractor, **kwargs)

    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]

    metric = kid_features_to_metric(features_1, features_2, **kwargs)
    return metric
