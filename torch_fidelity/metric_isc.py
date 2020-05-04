import numpy as np
import torch

from torch_fidelity.helpers import get_kwarg, vprint
from torch_fidelity.utils import extract_featuresdict_from_input_cached, create_feature_extractor, \
    get_input_cacheable_name

KEY_METRIC_ISC_MEAN = 'inception_score_mean'
KEY_METRIC_ISC_STD = 'inception_score_std'


def isc_features_to_metric(feature, splits=10, shuffle=True, rng_seed=2020):
    assert torch.is_tensor(feature) and feature.dim() == 2
    N, C = feature.shape
    if shuffle:
        rng = np.random.RandomState(rng_seed)
        feature = feature[rng.permutation(N), :]
    feature = feature.double()

    p = feature.softmax(dim=1)
    log_p = feature.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * N // splits): ((i + 1) * N // splits), :]
        log_p_chunk = log_p[(i * N // splits): ((i + 1) * N // splits), :]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = p_chunk * (log_p_chunk - q_chunk.log())
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)

    return {
        KEY_METRIC_ISC_MEAN: float(np.mean(scores)),
        KEY_METRIC_ISC_STD: float(np.std(scores)),
    }


def isc_featuresdict_to_metric(featuresdict, feat_layer_name, **kwargs):
    vprint(get_kwarg('verbose', kwargs), 'Computing Inception Score')

    features = featuresdict[feat_layer_name]

    metric = isc_features_to_metric(
        features,
        get_kwarg('isc_splits', kwargs),
        get_kwarg('samples_shuffle', kwargs),
        get_kwarg('rng_seed', kwargs),
    )

    return metric


def isc_input_to_metric(input, cacheable_input_name, feat_extractor, feat_layer_name, **kwargs):
    featuresdict = extract_featuresdict_from_input_cached(input, cacheable_input_name, feat_extractor, **kwargs)
    return isc_featuresdict_to_metric(featuresdict, feat_layer_name, **kwargs)


def calculate_isc(input_1, **kwargs):
    feat_layer_name = get_kwarg('feature_layer_isc', kwargs)
    feat_extractor = create_feature_extractor(
        get_kwarg('feature_extractor', kwargs),
        [feat_layer_name],
        **kwargs
    )
    cacheable_input1_name = get_input_cacheable_name(input_1, get_kwarg('cache_input1_name', kwargs))
    metric = isc_input_to_metric(input_1, cacheable_input1_name, feat_extractor, feat_layer_name, **kwargs)
    return metric
