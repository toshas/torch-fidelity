# Functions fid_features_to_statistics and fid_statistics_to_metric are adapted from
#   https://github.com/bioinf-jku/TTUR/blob/master/fid.py commit id d4baae8
#   Distributed under Apache License 2.0: https://github.com/bioinf-jku/TTUR/blob/master/LICENSE

import numpy as np
import torch

from torch_fidelity.helpers import get_kwarg, vprint
from torch_fidelity.utils import (
    get_cacheable_input_name,
    cache_lookup_one_recompute_on_miss,
    extract_featuresdict_from_input_id_cached,
    create_feature_extractor,
    resolve_feature_extractor,
    resolve_feature_layer_for_metric,
)

KEY_METRIC_FID = "frechet_inception_distance"


def fid_features_to_statistics(features):
    assert torch.is_tensor(features) and features.dim() == 2
    features = features.numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {
        "mu": mu,
        "sigma": sigma,
    }


def fid_statistics_to_metric(stat_1, stat_2, verbose):
    mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
    mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
    assert mu1.ndim == 1 and mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.ndim == 2 and sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    diff = mu1 - mu2
    tr_covmean = np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    out = {KEY_METRIC_FID: fid}

    vprint(verbose, f"Frechet Inception Distance: {out[KEY_METRIC_FID]:.7g}")

    return out


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
        cached_name = f"{cacheable_input_name}-{feat_extractor_name}-stat-fid-{feat_layer_name}"
        stat = cache_lookup_one_recompute_on_miss(cached_name, fn_recompute, **kwargs)
    else:
        stat = fn_recompute()
    return stat


def fid_input_id_to_statistics(input_id, feat_extractor, feat_layer_name, **kwargs):
    featuresdict = extract_featuresdict_from_input_id_cached(input_id, feat_extractor, **kwargs)
    return fid_featuresdict_to_statistics(featuresdict, feat_layer_name)


def fid_input_id_to_statistics_cached(input_id, feat_extractor, feat_layer_name, **kwargs):
    def fn_recompute():
        return fid_input_id_to_statistics(input_id, feat_extractor, feat_layer_name, **kwargs)

    cacheable_input_name = get_cacheable_input_name(input_id, **kwargs)

    if cacheable_input_name is not None:
        feat_extractor_name = feat_extractor.get_name()
        cached_name = f"{cacheable_input_name}-{feat_extractor_name}-stat-fid-{feat_layer_name}"
        stat = cache_lookup_one_recompute_on_miss(cached_name, fn_recompute, **kwargs)
    else:
        stat = fn_recompute()
    return stat


def fid_inputs_to_metric(feat_extractor, **kwargs):
    feat_layer_name = resolve_feature_layer_for_metric("fid", **kwargs)
    verbose = get_kwarg("verbose", kwargs)

    vprint(verbose, f"Extracting statistics from input 1")
    stats_1 = fid_input_id_to_statistics_cached(1, feat_extractor, feat_layer_name, **kwargs)

    vprint(verbose, f"Extracting statistics from input 2")
    stats_2 = fid_input_id_to_statistics_cached(2, feat_extractor, feat_layer_name, **kwargs)

    metric = fid_statistics_to_metric(stats_1, stats_2, get_kwarg("verbose", kwargs))
    return metric


def calculate_fid(**kwargs):
    kwargs["fid"] = True
    feature_extractor = resolve_feature_extractor(**kwargs)
    feat_layer_name = resolve_feature_layer_for_metric("fid", **kwargs)
    feat_extractor = create_feature_extractor(feature_extractor, [feat_layer_name], **kwargs)
    metric = fid_inputs_to_metric(feat_extractor, **kwargs)
    return metric
