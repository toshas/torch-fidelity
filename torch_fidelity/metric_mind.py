# Implementation of MIND (Monge Inception Distance), proposed in
#   Berthet et al., "MIND: Monge Inception Distance for Generative Models Evaluation",
#   https://arxiv.org/abs/2605.06797
# MIND approximates the sliced 2-Wasserstein distance between feature distributions:
#   MIND(p_g, p_r) = (alpha / (n * M)) * sum_i sum_j |sort(u_i^T X)_j - sort(u_i^T Y)_j|^2
# where u_i are M random unit-norm directions and alpha = 3 * d (d = feature dimension).

import torch

from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.utils import (
    create_feature_extractor,
    extract_featuresdict_from_input_id_cached,
    resolve_feature_extractor,
    resolve_feature_layer_for_metric,
)

KEY_METRIC_MIND = "monge_inception_distance"


def mind_features_to_metric(features_1, features_2, **kwargs):
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    assert features_1.shape[1] == features_2.shape[1]

    num_projections = get_kwarg("mind_num_projections", kwargs)
    cuda = get_kwarg("cuda", kwargs)
    rng_seed = get_kwarg("rng_seed", kwargs)
    verbose = get_kwarg("verbose", kwargs)

    vassert(isinstance(num_projections, int) and num_projections > 0, "mind_num_projections must be a positive integer")

    n1, n2 = features_1.shape[0], features_2.shape[0]
    d = features_1.shape[1]
    n = min(n1, n2)

    cpu_g = torch.Generator(device="cpu").manual_seed(rng_seed)

    if n1 != n2:
        if n1 > n:
            idx1 = torch.randperm(n1, generator=cpu_g)[:n]
            features_1 = features_1[idx1]
        if n2 > n:
            idx2 = torch.randperm(n2, generator=cpu_g)[:n]
            features_2 = features_2[idx2]

    directions = torch.randn(num_projections, d, generator=cpu_g, dtype=torch.float32)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    features_1 = features_1.to(device=device, dtype=torch.float32)
    features_2 = features_2.to(device=device, dtype=torch.float32)
    directions = directions.to(device=device)

    # Projections: (n, M)
    proj_1 = features_1 @ directions.t()
    proj_2 = features_2 @ directions.t()

    # Sort each projection (column) independently to align order statistics.
    proj_1, _ = proj_1.sort(dim=0)
    proj_2, _ = proj_2.sort(dim=0)

    # 1D squared 2-Wasserstein per direction, then mean over directions.
    sliced_w2_sq = ((proj_1 - proj_2) ** 2).mean(dim=0)
    alpha = 3.0 * d
    mind = float(alpha * sliced_w2_sq.mean().item())

    out = {KEY_METRIC_MIND: mind}
    vprint(verbose, f"Monge Inception Distance: {out[KEY_METRIC_MIND]:.7g}")
    return out


def mind_featuresdict_to_metric(featuresdict_1, featuresdict_2, feat_layer_name, **kwargs):
    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]
    return mind_features_to_metric(features_1, features_2, **kwargs)


def calculate_mind(**kwargs):
    kwargs["mind"] = True
    feature_extractor = resolve_feature_extractor(**kwargs)
    feat_layer_name = resolve_feature_layer_for_metric("mind", **kwargs)
    feat_extractor = create_feature_extractor(feature_extractor, [feat_layer_name], **kwargs)
    featuresdict_1 = extract_featuresdict_from_input_id_cached(1, feat_extractor, **kwargs)
    featuresdict_2 = extract_featuresdict_from_input_id_cached(2, feat_extractor, **kwargs)
    return mind_featuresdict_to_metric(featuresdict_1, featuresdict_2, feat_layer_name, **kwargs)
