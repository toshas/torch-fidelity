from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.metric_fid import fid_inputs_to_metric, fid_featuresdict_to_statistics_cached, \
    fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_featuresdict_to_metric
from torch_fidelity.metric_kid import kid_featuresdict_to_metric
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_cached, \
    get_input_cacheable_name


def calculate_metrics(input_1, input_2=None, **kwargs):
    r"""
    Calculate metrics for the given inputs.
    Args:
        input_1: str or torch.util.data.Dataset
            First positional input, can be either a Dataset instance, or a string containing a path to a directory
            of images, or one of the registered input sources (see registry.py).
        input_2: str or torch.util.data.Dataset
            Second positional input (not used in unary metrics, such as "isc"), can be either a Dataset instance, or a
            string containing a path to a directory of images, or one of the registered input sources (see registry.py).
        cuda: bool (default: True)
            Sets executor device to GPU.
        batch_size: int (default: 64)
            Batch size used to process images; the larger the more memory is used on the executor device (see "cuda"
            argument).
        isc: bool (default: False)
            Calculate ISC (Inception Score).
        fid: bool (default: False)
            Calculate FID (Frechet Inception Distance).
        kid: bool (default: False)
            Calculate KID (Kernel Inception Distance).
        feature_extractor: str (default: inception-v3-compat)
            Name of the feature extractor (see registry.py).
        feature_layer_isc: str (default: logits_unbiased)
            Name of the feature layer to use with ISC metric.
        feature_layer_fid: str (default: 2048)
            Name of the feature layer to use with FID metric.
        feature_layer_kid: str (default: 2048)
            Name of the feature layer to use with KID metric.
        feature_extractor_weights_path: str (default: None)
            Path to feature extractor weights (downloaded if None).
        isc_splits: int (default: 10)
            Number of splits in ISC.
        kid_subsets: int (default: 100)
            Number of subsets in KID.
        kid_subset_size: int (default: 1000)
            Subset size in KID.
        kid_degree: int (default: 3)
            Degree of polynomial kernel in KID.
        kid_gamma: float (default: None)
            Polynomial kernel gamma in KID (automatic if None).
        kid_coef0: float (default: 1)
            Polynomial kernel coef0 in KID.
        samples_shuffle: bool (default: True)
            Perform random samples shuffling before computing splits.
        samples_find_deep: bool (default: False)
            Find all samples in paths recursively.
        samples_find_ext: str (default: png,jpg,jpeg)
            List of extensions to look for when traversing input path.
        samples_ext_lossy: str (default: jpg,jpeg)
            List of extensions to warn about lossy compression.
        datasets_root: str (default: None)
            Path to built-in torchvision datasets root. Defaults to $ENV_TORCH_HOME/fidelity_datasets.
        datasets_download: bool (default: True)
            Download torchvision datasets to dataset_root.
        cache_root: str (default: None)
            Path to file cache for features and statistics. Defaults to $ENV_TORCH_HOME/fidelity_cache.
        cache: bool (default: True)
            Use file cache for features and statistics.
        cache_input1_name: str (default: None)
            Assigns a cache entry to input1 (if a path) and forces caching of features on it if not None.
        cache_input2_name: str (default: None)
            Assigns a cache entry to input2 (if a path) and forces caching of features on it if not None.
        rng_seed: int (default: 2020)
            Random numbers generator seed for all operations involving randomness.
        save_cpu_ram: bool (default: False)
            Use less CPU RAM at the cost of speed.
        verbose: bool (default: True)
            Output progress information to STDERR.

    Return: a dictionary of metrics.
    """

    have_isc, have_fid, have_kid = get_kwarg('isc', kwargs), get_kwarg('fid', kwargs), get_kwarg('kid', kwargs)
    vassert(have_isc or have_fid or have_kid, 'At least one of "isc", "fid", "kid" metrics must be specified')
    vassert(
        (not have_fid) and (not have_kid) or input_2 is not None, 'Both inputs are required for "fid" and "kid" metrics'
    )
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

    vprint(verbose, f'Extracting features from input_1')
    featuresdict_1 = extract_featuresdict_from_input_cached(input_1, cacheable_input1_name, feat_extractor, **kwargs)
    featuresdict_2 = None
    if input_2 is not None:
        cacheable_input2_name = get_input_cacheable_name(input_2, get_kwarg('cache_input2_name', kwargs))
        vprint(verbose, f'Extracting features from input_2')
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
