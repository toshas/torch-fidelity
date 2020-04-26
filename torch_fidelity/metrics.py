from torch_fidelity.metric_fid import fid_inputs_to_metric, fid_featuresdict_to_statistics_cached, \
    fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_featuresdict_to_metric
from torch_fidelity.metric_kid import kid_featuresdict_to_metric
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_cached


def calculate_metrics(input_1, input_2=None, **kwargs):
    have_isc, have_fid, have_kid = kwargs['isc'], kwargs['fid'], kwargs['kid']
    assert have_isc or have_fid or have_kid, 'At least one of "isc", "fid", "kid" metrics must be specified'
    assert (not have_fid) and (not have_kid) or input_2 is not None, \
        'Both inputs are required for "fid" and "kid" metrics'

    feature_layer_isc, feature_layer_fid, feature_layer_kid = (None,) * 3
    feature_layers = []
    if have_isc:
        feature_layer_isc = kwargs['feature_layer_isc']
        feature_layers.append(feature_layer_isc)
    if have_fid:
        feature_layer_fid = kwargs['feature_layer_fid']
        feature_layers.append(feature_layer_fid)
    if have_kid:
        feature_layer_kid = kwargs['feature_layer_kid']
        feature_layers.append(feature_layer_kid)

    feat_extractor = create_feature_extractor(
        kwargs['feature_extractor'], feature_layers, cuda=kwargs['cuda'], **kwargs
    )

    # isc: input - featuresdict(cached) - metric
    # fid: input - featuresdict(cached) - statistics(cached) - metric
    # kid: input - featuresdict(cached) - metric

    metrics = {}

    if (not have_isc) and have_fid and (not have_kid):
        # shortcut for a case when statistics are cached and features are not required on at least one input
        metric_fid = fid_inputs_to_metric(input_1, input_2, feat_extractor, feature_layer_fid, **kwargs)
        metrics.update(metric_fid)
        return metrics

    featuresdict_1 = extract_featuresdict_from_input_cached(input_1, feat_extractor, **kwargs)
    featuresdict_2 = None
    if input_2 is not None:
        featuresdict_2 = extract_featuresdict_from_input_cached(input_2, feat_extractor, **kwargs)

    if have_isc:
        metric_isc = isc_featuresdict_to_metric(featuresdict_1, feature_layer_isc, **kwargs)
        metrics.update(metric_isc)

    if have_fid:
        fid_stats_1 = fid_featuresdict_to_statistics_cached(
            featuresdict_1, input, feat_extractor, feature_layer_fid, **kwargs
        )
        fid_stats_2 = fid_featuresdict_to_statistics_cached(
            featuresdict_2, input, feat_extractor, feature_layer_fid, **kwargs
        )
        metric_fid = fid_statistics_to_metric(fid_stats_1, fid_stats_2, kwargs['verbose'])
        metrics.update(metric_fid)

    if have_kid:
        metric_kid = kid_featuresdict_to_metric(featuresdict_1, featuresdict_2, feature_layer_kid, **kwargs)
        metrics.update(metric_kid)

    return metrics
