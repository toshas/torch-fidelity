import torch

from torch_fidelity.helpers import get_kwarg, vprint
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached

KEY_METRIC_PRECISION = 'precision'
KEY_METRIC_RECALL = 'recall'


def calc_cdist_part(features_1, features_2, batch_size=10000):
    dists = []
    for feat2_batch in features_2.split(batch_size):
        dists.append(torch.cdist(features_1, feat2_batch).cpu())
    return torch.cat(dists, dim=1)


def calculate_precision_recall_part(features_r, features_g, NN_k=3, batch_size=10000):
    # Precision
    dist_NN_r = []
    for feat_r_batch in features_r.split(batch_size):
        dist_NN_r.append(calc_cdist_part(feat_r_batch, features_r, batch_size).kthvalue(NN_k+1).values)
    dist_NN_r = torch.cat(dist_NN_r)
    precision = []
    for feat_g_batch in features_g.split(batch_size):
        dist_g_r_batch = calc_cdist_part(feat_g_batch, features_r, batch_size)
        precision.append((dist_g_r_batch <= dist_NN_r).any(dim=1).float())
    precision = torch.cat(precision).mean().item()
    # Recall
    dist_NN_g = []
    for feat_g_batch in features_g.split(batch_size):
        dist_NN_g.append(calc_cdist_part(feat_g_batch, features_g, batch_size).kthvalue(NN_k+1).values)
    dist_NN_g = torch.cat(dist_NN_g)
    recall = []
    for feat_r_batch in features_r.split(batch_size):
        dist_r_g_batch = calc_cdist_part(feat_r_batch, features_g, batch_size)
        recall.append((dist_r_g_batch <= dist_NN_g).any(dim=1).float())
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


def calculate_precision_recall_full(features_r, features_g, NN_k=3, batch_size=10000):
    dist_NN_r = calc_cdist_full(features_r, features_r, batch_size).kthvalue(NN_k+1).values
    dist_NN_g = calc_cdist_full(features_g, features_g, batch_size).kthvalue(NN_k+1).values
    dist_g_r = calc_cdist_full(features_g, features_r, batch_size)
    dist_r_g = dist_g_r.T
    # Precision
    precision = (dist_g_r <= dist_NN_r).any(dim=1).float().mean().item()
    # Recall
    recall = (dist_r_g <= dist_NN_g).any(dim=1).float().mean().item()
    return precision, recall


def prc_features_to_metric(features_1, features_2, **kwargs):
    # ASSUMING features_1 is of REAL, and features_2 is of GENERATED
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    assert features_1.shape[1] == features_2.shape[1]

    NN_k = get_kwarg('prc_NN_k', kwargs)
    batch_size = get_kwarg('prc_batch_size', kwargs)
    save_cpu_ram = get_kwarg('save_cpu_ram', kwargs)
    verbose = get_kwarg('verbose', kwargs)

    if save_cpu_ram:
        precision, recall = calculate_precision_recall_part(features_1, features_2, NN_k, batch_size)
    else:
        precision, recall = calculate_precision_recall_full(features_1, features_2, NN_k, batch_size)

    out = {
        KEY_METRIC_PRECISION: precision,
        KEY_METRIC_RECALL: recall,
    }

    vprint(verbose, f'Precision: {out[KEY_METRIC_PRECISION]}')
    vprint(verbose, f'Recall: {out[KEY_METRIC_RECALL]}')

    return out


def prc_featuresdict_to_metric(featuresdict_1, featuresdict_2, feat_layer_name, **kwargs):
    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]
    metric = prc_features_to_metric(features_1, features_2, **kwargs)
    return metric


def calculate_prc(**kwargs):
    feature_extractor = get_kwarg('feature_extractor', kwargs)
    feat_layer_name = get_kwarg('feature_layer_prc', kwargs)
    feat_extractor = create_feature_extractor(feature_extractor, [feat_layer_name], **kwargs)
    featuresdict_1 = extract_featuresdict_from_input_id_cached(1, feat_extractor, **kwargs)
    featuresdict_2 = extract_featuresdict_from_input_id_cached(2, feat_extractor, **kwargs)
    metric = prc_featuresdict_to_metric(featuresdict_1, featuresdict_2, feat_layer_name, **kwargs)
    return metric
