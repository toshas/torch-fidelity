import numpy as np
import torch

from torch_fidelity.helpers import get_kwarg, vprint
from torch_fidelity.utils import (
    extract_featuresdict_from_input_id_cached,
    create_feature_extractor,
    resolve_feature_extractor,
    resolve_feature_layer_for_metric,
)

KEY_METRIC_ISC_MEAN = "inception_score_mean"
KEY_METRIC_ISC_STD = "inception_score_std"


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
        p_chunk = p[(i * N // splits) : ((i + 1) * N // splits), :]
        log_p_chunk = log_p[(i * N // splits) : ((i + 1) * N // splits), :]
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

    out = isc_features_to_metric(
        features,
        get_kwarg("isc_splits", kwargs),
        get_kwarg("samples_shuffle", kwargs),
        get_kwarg("rng_seed", kwargs),
    )

    vprint(
        get_kwarg("verbose", kwargs), f"Inception Score: {out[KEY_METRIC_ISC_MEAN]:.7g} ± {out[KEY_METRIC_ISC_STD]:.7g}"
    )

    return out


def isc_input_id_to_metric(input_id, feat_extractor, feat_layer_name, **kwargs):
    featuresdict = extract_featuresdict_from_input_id_cached(input_id, feat_extractor, **kwargs)
    return isc_featuresdict_to_metric(featuresdict, feat_layer_name, **kwargs)


def calculate_isc(input_id, **kwargs):
    kwargs["isc"] = True
    feature_extractor = resolve_feature_extractor(**kwargs)
    feat_layer_name = resolve_feature_layer_for_metric("isc", **kwargs)
    feat_extractor = create_feature_extractor(feature_extractor, [feat_layer_name], **kwargs)
    metric = isc_input_id_to_metric(input_id, feat_extractor, feat_layer_name, **kwargs)
    return metric
