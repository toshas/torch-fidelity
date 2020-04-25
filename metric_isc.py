import numpy as np
import torch

from utils import extract_featuresdict_from_input_cached, create_feature_extractor

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
    features = featuresdict[feat_layer_name]
    metric = isc_features_to_metric(
        features,
        kwargs['isc_splits'],
        kwargs['shuffle_on'],
        kwargs['rng_seed'],
    )
    return metric


def isc_input_to_metric(input, feat_extractor, feat_layer_name, **kwargs):
    featuresdict = extract_featuresdict_from_input_cached(input, feat_extractor, **kwargs)
    return isc_featuresdict_to_metric(featuresdict, feat_layer_name, **kwargs)


def isc_alone(input, **kwargs):
    feat_layer_name = kwargs['feature_layer_isc']
    feat_extractor = create_feature_extractor(
        kwargs['feature_extractor'],
        [feat_layer_name],
        cuda=kwargs['cuda'],
        **kwargs
    )
    metric = isc_input_to_metric(input, feat_extractor, feat_layer_name, **kwargs)
    return metric
