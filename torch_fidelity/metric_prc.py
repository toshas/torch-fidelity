import torch

from torch_fidelity.helpers import get_kwarg, vprint
from torch_fidelity.utils import (
    create_feature_extractor,
    extract_featuresdict_from_input_id_cached,
    resolve_feature_extractor,
    resolve_feature_layer_for_metric,
)

KEY_METRIC_PRECISION = "precision"
KEY_METRIC_RECALL = "recall"
KEY_METRIC_F_SCORE = "f_score"


def calc_cdist_part(features_1, features_2, batch_size=10000):
    dists = []
    for feat2_batch in features_2.split(batch_size):
        dists.append(torch.cdist(features_1, feat2_batch).cpu())
    return torch.cat(dists, dim=1)


def calculate_precision_recall_part(features_1, features_2, neighborhood=3, batch_size=10000):
    # Precision
    dist_nn_1 = []
    for feat_1_batch in features_1.split(batch_size):
        dist_nn_1.append(calc_cdist_part(feat_1_batch, features_1, batch_size).kthvalue(neighborhood + 1).values)
    dist_nn_1 = torch.cat(dist_nn_1)
    precision = []
    for feat_2_batch in features_2.split(batch_size):
        dist_2_1_batch = calc_cdist_part(feat_2_batch, features_1, batch_size)
        precision.append((dist_2_1_batch <= dist_nn_1).any(dim=1).float())
    precision = torch.cat(precision).mean().item()
    # Recall
    dist_nn_2 = []
    for feat_2_batch in features_2.split(batch_size):
        dist_nn_2.append(calc_cdist_part(feat_2_batch, features_2, batch_size).kthvalue(neighborhood + 1).values)
    dist_nn_2 = torch.cat(dist_nn_2)
    recall = []
    for feat_1_batch in features_1.split(batch_size):
        dist_1_2_batch = calc_cdist_part(feat_1_batch, features_2, batch_size)
        recall.append((dist_1_2_batch <= dist_nn_2).any(dim=1).float())
    recall = torch.cat(recall).mean().item()
    return precision, recall


def calc_cdist_full(features_1, features_2, batch_size=10000):
    dists = []
    for feat1_batch in features_1.split(batch_size):
        dists_batch = []
        for feat2_batch in features_2.split(batch_size):
            dists_batch.append(torch.cdist(feat1_batch, feat2_batch).cpu())
        dists.append(torch.cat(dists_batch, dim=1))
    return torch.cat(dists, dim=0)


def calculate_precision_recall_full(features_1, features_2, neighborhood=3, batch_size=10000):
    dist_nn_1 = calc_cdist_full(features_1, features_1, batch_size).kthvalue(neighborhood + 1).values
    dist_nn_2 = calc_cdist_full(features_2, features_2, batch_size).kthvalue(neighborhood + 1).values
    dist_2_1 = calc_cdist_full(features_2, features_1, batch_size)
    dist_1_2 = dist_2_1.T
    # Precision
    precision = (dist_2_1 <= dist_nn_1).any(dim=1).float().mean().item()
    # Recall
    recall = (dist_1_2 <= dist_nn_2).any(dim=1).float().mean().item()
    return precision, recall


def prc_features_to_metric(features_1, features_2, **kwargs):
    # Convention: features_1 is REAL, features_2 is GENERATED. This important for the notion of precision/recall only.
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    assert features_1.shape[1] == features_2.shape[1]

    neighborhood = get_kwarg("prc_neighborhood", kwargs)
    batch_size = get_kwarg("prc_batch_size", kwargs)
    save_cpu_ram = get_kwarg("save_cpu_ram", kwargs)
    verbose = get_kwarg("verbose", kwargs)

    calculate_precision_recall_fn = calculate_precision_recall_part if save_cpu_ram else calculate_precision_recall_full
    precision, recall = calculate_precision_recall_fn(features_1, features_2, neighborhood, batch_size)
    f_score = 2 * precision * recall / max(1e-5, precision + recall)

    out = {
        KEY_METRIC_PRECISION: precision,
        KEY_METRIC_RECALL: recall,
        KEY_METRIC_F_SCORE: f_score,
    }

    vprint(verbose, f"Precision: {out[KEY_METRIC_PRECISION]:.7g}")
    vprint(verbose, f"Recall: {out[KEY_METRIC_RECALL]:.7g}")
    vprint(verbose, f"F-score: {out[KEY_METRIC_F_SCORE]:.7g}")

    return out


def prc_featuresdict_to_metric(featuresdict_1, featuresdict_2, feat_layer_name, **kwargs):
    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]
    metric = prc_features_to_metric(features_1, features_2, **kwargs)
    return metric


def calculate_prc(**kwargs):
    kwargs["prc"] = True
    feature_extractor = resolve_feature_extractor(**kwargs)
    feat_layer_name = resolve_feature_layer_for_metric("prc", **kwargs)
    feat_extractor = create_feature_extractor(feature_extractor, [feat_layer_name], **kwargs)
    featuresdict_1 = extract_featuresdict_from_input_id_cached(1, feat_extractor, **kwargs)
    featuresdict_2 = extract_featuresdict_from_input_id_cached(2, feat_extractor, **kwargs)
    metric = prc_featuresdict_to_metric(featuresdict_1, featuresdict_2, feat_layer_name, **kwargs)
    return metric
