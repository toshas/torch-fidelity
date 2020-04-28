import sys

from torch_fidelity.defaults import get_kwarg
from torch_fidelity.metric_fid import fid_inputs_to_metric, fid_featuresdict_to_statistics_cached, \
    fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_featuresdict_to_metric
from torch_fidelity.metric_kid import kid_featuresdict_to_metric
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_cached, \
    get_input_cacheable_name


def calculate_metrics(input_1, input_2=None, **kwargs):
    have_isc, have_fid, have_kid = get_kwarg('isc', kwargs), get_kwarg('fid', kwargs), get_kwarg('kid', kwargs)
    assert have_isc or have_fid or have_kid, 'At least one of "isc", "fid", "kid" metrics must be specified'
    assert (not have_fid) and (not have_kid) or input_2 is not None, \
        'Both inputs are required for "fid" and "kid" metrics'
    verbose = get_kwarg('verbose', kwargs)

    feature_layer_isc, feature_layer_fid, feature_layer_kid = (None,) * 3
    feature_layers = set()
    if have_isc:
        feature_layer_isc = get_kwarg('feature_layer_isc', kwargs)
        feature_layers.add(feature_layer_isc)
    if have_fid:
        feature_layer_fid = get_kwarg('feature_layer_fid', kwargs)
        feature_layers.add(feature_layer_fid)
    if have_kid:
        feature_layer_kid = get_kwarg('feature_layer_kid', kwargs)
        feature_layers.add(feature_layer_kid)

    feat_extractor = create_feature_extractor(
        get_kwarg('feature_extractor', kwargs), list(feature_layers), **kwargs
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

    cacheable_input1_name = get_input_cacheable_name(input_1, get_kwarg('cache_input1_name', kwargs))
    cacheable_input2_name = get_input_cacheable_name(input_2, get_kwarg('cache_input2_name', kwargs))

    if verbose:
        print(f'Extracting features from input_1', file=sys.stderr)
    featuresdict_1 = extract_featuresdict_from_input_cached(input_1, cacheable_input1_name, feat_extractor, **kwargs)
    featuresdict_2 = None
    if input_2 is not None:
        if verbose:
            print(f'Extracting features from input_2', file=sys.stderr)
        featuresdict_2 = extract_featuresdict_from_input_cached(
            input_2, cacheable_input2_name, feat_extractor, **kwargs
        )

    if have_isc:
        metric_isc = isc_featuresdict_to_metric(featuresdict_1, feature_layer_isc, **kwargs)
        metrics.update(metric_isc)

    if have_fid:
        fid_stats_1 = fid_featuresdict_to_statistics_cached(
            featuresdict_1, cacheable_input1_name, feat_extractor, feature_layer_fid, **kwargs
        )
        fid_stats_2 = fid_featuresdict_to_statistics_cached(
            featuresdict_2, cacheable_input2_name, feat_extractor, feature_layer_fid, **kwargs
        )
        metric_fid = fid_statistics_to_metric(fid_stats_1, fid_stats_2, get_kwarg('verbose', kwargs))
        metrics.update(metric_fid)

    if have_kid:
        metric_kid = kid_featuresdict_to_metric(featuresdict_1, featuresdict_2, feature_layer_kid, **kwargs)
        metrics.update(metric_kid)

    return metrics
