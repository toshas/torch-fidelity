import numpy as np
import scipy.linalg
import torch

from torch_fidelity.helpers import get_kwarg, vprint
from torch_fidelity.utils import get_input_cacheable_name, cache_lookup_one_recompute_on_miss, \
    extract_featuresdict_from_input_cached, create_feature_extractor

KEY_METRIC_FID = 'frechet_inception_distance'


def fid_features_to_statistics(features):
    assert torch.is_tensor(features) and features.dim() == 2
    features = features.numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {
        'mu': mu,
        'sigma': sigma,
    }


def fid_statistics_to_metric(stat_1, stat_2, verbose):
    eps = 1e-6

    vprint(verbose, 'Computing Frechet Inception Distance')

    mu1, sigma1 = stat_1['mu'], stat_1['sigma']
    mu2, sigma2 = stat_2['mu'], stat_2['sigma']
    assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        vprint(verbose,
            f'WARNING: fid calculation produces singular product; '
            f'adding {eps} to diagonal of cov estimates'
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=verbose)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            assert False, 'Imaginary component {}'.format(m)
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return {
        KEY_METRIC_FID: float(fid),
    }


def fid_featuresdict_to_statistics(featuresdict, feat_layer_name):
    features = featuresdict[feat_layer_name]
    statistics = fid_features_to_statistics(features)
    return statistics


def fid_featuresdict_to_statistics_cached(
        featuresdict, cacheable_input_name, feat_extractor, feat_layer_name, **kwargs
):

    def fn_recompute():
        return fid_featuresdict_to_statistics(featuresdict, feat_layer_name)

    if cacheable_input_name is not None:
        feat_extractor_name = feat_extractor.get_name()
        cached_name = f'{cacheable_input_name}-{feat_extractor_name}-stat-fid-{feat_layer_name}'
        stat = cache_lookup_one_recompute_on_miss(cached_name, fn_recompute, **kwargs)
    else:
        stat = fn_recompute()
    return stat


def fid_input_to_statistics(input, cacheable_input_name, feat_extractor, feat_layer_name, **kwargs):
    featuresdict = extract_featuresdict_from_input_cached(input, cacheable_input_name, feat_extractor, **kwargs)
    return fid_featuresdict_to_statistics(featuresdict, feat_layer_name)


def fid_input_to_statistics_cached(input, cacheable_input_name, feat_extractor, feat_layer_name, **kwargs):

    def fn_recompute():
        return fid_input_to_statistics(input, cacheable_input_name, feat_extractor, feat_layer_name, **kwargs)

    if cacheable_input_name is not None:
        feat_extractor_name = feat_extractor.get_name()
        cached_name = f'{cacheable_input_name}-{feat_extractor_name}-stat-fid-{feat_layer_name}'
        stat = cache_lookup_one_recompute_on_miss(cached_name, fn_recompute, **kwargs)
    else:
        stat = fn_recompute()
    return stat


def fid_inputs_to_metric(input_1, input_2, feat_extractor, feat_layer_name, **kwargs):
    verbose = get_kwarg('verbose', kwargs)

    cacheable_input1_name = get_input_cacheable_name(input_1, get_kwarg('cache_input1_name', kwargs))
    cacheable_input2_name = get_input_cacheable_name(input_2, get_kwarg('cache_input2_name', kwargs))

    vprint(verbose, f'Extracting statistics from input_1')
    stats_1 = fid_input_to_statistics_cached(input_1, cacheable_input1_name, feat_extractor, feat_layer_name, **kwargs)

    vprint(verbose, f'Extracting statistics from input_2')
    stats_2 = fid_input_to_statistics_cached(input_2, cacheable_input2_name, feat_extractor, feat_layer_name, **kwargs)

    metric = fid_statistics_to_metric(stats_1, stats_2, get_kwarg('verbose', kwargs))
    return metric


def calculate_fid(input_1, input_2, **kwargs):
    feat_layer_name = get_kwarg('feature_layer_fid', kwargs)
    feat_extractor = create_feature_extractor(
        get_kwarg('feature_extractor', kwargs),
        [feat_layer_name],
        **kwargs
    )
    metric = fid_inputs_to_metric(input_1, input_2, feat_extractor, feat_layer_name, **kwargs)
    return metric
