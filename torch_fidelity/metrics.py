from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.metric_fid import fid_inputs_to_metric, fid_featuresdict_to_statistics_cached, \
    fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_featuresdict_to_metric
from torch_fidelity.metric_kid import kid_featuresdict_to_metric
from torch_fidelity.metric_prc import prc_featuresdict_to_metric
from torch_fidelity.metric_ppl import calculate_ppl
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached, \
    get_cacheable_input_name, resolve_feature_extractor, resolve_feature_layer_for_metric


def calculate_metrics_one_feature_extractor(**kwargs):
    verbose = get_kwarg('verbose', kwargs)
    input1, input2 = get_kwarg('input1', kwargs), get_kwarg('input2', kwargs)

    have_isc = get_kwarg('isc', kwargs)
    have_fid = get_kwarg('fid', kwargs)
    have_kid = get_kwarg('kid', kwargs)
    have_prc = get_kwarg('prc', kwargs)
    have_ppl = get_kwarg('ppl', kwargs)

    have_unary = have_isc or have_ppl
    have_binary = have_fid or have_kid or have_prc
    have_any = have_unary or have_binary
    have_other_than_ppl = have_isc or have_binary
    have_only_fid = (not have_isc) and have_fid and (not have_kid) and (not have_prc)

    need_input1 = True
    need_input2 = have_binary

    vassert(have_any, 'At least one metric must be specified')
    vassert(input1 is not None or not need_input1, 'First input is required for all metrics')
    vassert(input2 is not None or not need_input2, 'Second input is required for binary metrics')

    metrics = {}

    if have_other_than_ppl:
        feature_extractor = resolve_feature_extractor(**kwargs)
        feature_layer_isc, feature_layer_fid, feature_layer_kid, feature_layer_prc = (None,) * 4
        feature_layers = set()
        if have_isc:
            feature_layer_isc = resolve_feature_layer_for_metric('isc', **kwargs)
            feature_layers.add(feature_layer_isc)
        if have_fid:
            feature_layer_fid = resolve_feature_layer_for_metric('fid', **kwargs)
            feature_layers.add(feature_layer_fid)
        if have_kid:
            feature_layer_kid = resolve_feature_layer_for_metric('kid', **kwargs)
            feature_layers.add(feature_layer_kid)
        if have_prc:
            feature_layer_prc = resolve_feature_layer_for_metric('prc', **kwargs)
            feature_layers.add(feature_layer_prc)

        feat_extractor = create_feature_extractor(feature_extractor, list(feature_layers), **kwargs)

        # isc: input - featuresdict(cached) - metric
        # fid: input - featuresdict(cached) - statistics(cached) - metric
        # kid: input - featuresdict(cached) - metric

        if have_only_fid:
            # shortcut for a case when statistics are cached and features are not required on at least one input
            metric_fid = fid_inputs_to_metric(feat_extractor, **kwargs)
            metrics.update(metric_fid)
            return metrics

        vprint(verbose, f'Extracting features from input1')
        featuresdict_1 = extract_featuresdict_from_input_id_cached(1, feat_extractor, **kwargs)
        featuresdict_2 = None
        if input2 is not None:
            vprint(verbose, f'Extracting features from input2')
            featuresdict_2 = extract_featuresdict_from_input_id_cached(2, feat_extractor, **kwargs)

        if have_isc:
            metric_isc = isc_featuresdict_to_metric(featuresdict_1, feature_layer_isc, **kwargs)
            metrics.update(metric_isc)

        if have_fid:
            cacheable_input1_name = get_cacheable_input_name(1, **kwargs)
            cacheable_input2_name = get_cacheable_input_name(2, **kwargs)
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

        if have_prc:
            metric_prc = prc_featuresdict_to_metric(featuresdict_1, featuresdict_2, feature_layer_prc, **kwargs)
            metrics.update(metric_prc)

    if have_ppl:
        metric_ppl = calculate_ppl(1, **kwargs)
        metrics.update(metric_ppl)

    return metrics


def calculate_metrics(**kwargs):
    """
    Calculates metrics for the given inputs. Keyword arguments:

    .. _ISC: https://arxiv.org/pdf/1606.03498.pdf
    .. _FID: https://arxiv.org/pdf/1706.08500.pdf
    .. _KID: https://arxiv.org/pdf/1801.01401.pdf
    .. _PPL: https://arxiv.org/pdf/1812.04948.pdf
    .. _PRC: https://arxiv.org/pdf/1904.06991.pdf

    Args:

        input1 (str or torch.utils.data.Dataset or GenerativeModelBase):
            First input, which can be either of the following values:

            - Name of a registered input. See :ref:`registry <Registry>` for the complete list of preregistered
              inputs, and :meth:`register_dataset` for registering a new input. The following options refine the
              behavior wrt dataset location and downloading:
              :paramref:`~calculate_metrics.datasets_root`,
              :paramref:`~calculate_metrics.datasets_download`.
            - Path to a directory with samples. The following options refine the behavior wrt directory
              traversal and samples filtering:
              :paramref:`~calculate_metrics.samples_find_deep`,
              :paramref:`~calculate_metrics.samples_find_ext`, and
              :paramref:`~calculate_metrics.samples_ext_lossy`.
            - Path to a generative model in the :obj:`ONNX<torch:torch.onnx>` or `PTH` (:obj:`JIT<torch:torch.jit>`)
              format. This option also requires the following kwargs:
              :paramref:`~calculate_metrics.input1_model_z_type`,
              :paramref:`~calculate_metrics.input1_model_z_size`, and
              :paramref:`~calculate_metrics.input1_model_num_classes`.
            - Instance of :class:`~torch:torch.utils.data.Dataset` encapsulating a fixed set of samples.
            - Instance of :class:`GenerativeModelBase`, implementing the generative model.

            Default: `None`.

        input2 (str or torch.utils.data.Dataset or GenerativeModelBase):
            Second input, which can be either of the following values:

            - Name of a registered input. See :ref:`registry <Registry>` for the complete list of preregistered
              inputs, and :meth:`register_dataset` for registering a new input. The following options refine the
              behavior wrt dataset location and downloading:
              :paramref:`~calculate_metrics.datasets_root`,
              :paramref:`~calculate_metrics.datasets_download`.
            - Path to a directory with samples. The following options refine the behavior wrt directory
              traversal and samples filtering:
              :paramref:`~calculate_metrics.samples_find_deep`,
              :paramref:`~calculate_metrics.samples_find_ext`, and
              :paramref:`~calculate_metrics.samples_ext_lossy`.
            - Path to a generative model in the :obj:`ONNX<torch:torch.onnx>` or `PTH` (:obj:`JIT<torch:torch.jit>`)
              format. This option also requires the following kwargs:
              :paramref:`~calculate_metrics.input2_model_z_type`,
              :paramref:`~calculate_metrics.input2_model_z_size`, and
              :paramref:`~calculate_metrics.input2_model_num_classes`.
            - Instance of :class:`~torch:torch.utils.data.Dataset` encapsulating a fixed set of samples.
            - Instance of :class:`GenerativeModelBase`, implementing the generative model.

            Default: `None`.

        cuda (bool): Sets executor device to GPU. Default: `True`.

        batch_size (int): Batch size used to process images; the larger the more memory is used on the executor device
            (see :paramref:`~calculate_metrics.cuda`). Default: `64`.

        isc (bool): Calculate ISC_ (Inception Score). Default: `False`.

        fid (bool): Calculate FID_ (Frechet Inception Distance). Default: `False`.

        kid (bool): Calculate KID_ (Kernel Inception Distance). Default: `False`.

        prc (bool): Calculate PRC_ (Precision and Recall). Default: `False`.

        ppl (bool): Calculate PPL_ (Perceptual Path Length). Default: `False`.

        feature_extractor (str): Name of the feature extractor (see :ref:`registry <Registry>`). Default: `None`
            (defined by the chosen set of metrics to compute).

        feature_layer_isc (str): Name of the feature layer to use with ISC metric. Default: `None` (defined by the
            chosen feature extractor).

        feature_layer_fid (str): Name of the feature layer to use with FID metric. Default: `None` (defined by the
            chosen feature extractor).

        feature_layer_kid (str): Name of the feature layer to use with KID metric. Default: `None` (defined by the
            chosen feature extractor).

        feature_layer_prc (str): Name of the feature layer to use with PRC metric. Default: `None` (defined by the
            chosen feature extractor).

        feature_extractor_weights_path (str): Path to feature extractor weights (downloaded if `None`). Default: `None`.

        feature_extractor_internal_dtype (str): dtype to use inside the feature extractor. Default: `None` (defined by
            the chosen feature extractor).

        feature_extractor_compile (bool): Compile feature extractor (experimental: may have negative effect on the
            metrics numerical precision). Default: False.

        isc_splits (int): Number of splits in ISC. Default: `10`.

        kid_subsets (int): Number of subsets in KID. Default: `100`.

        kid_subset_size (int): Subset size in KID. Default: `1000`.

        kid_degree (int): Degree of polynomial kernel in KID. Default: `3`.

        kid_gamma (float): Polynomial kernel gamma in KID (automatic if `None`). Default: `None`.

        kid_coef0 (float): Polynomial kernel coef0 in KID. Default: `1.0`.

        ppl_epsilon (float): Interpolation step size in PPL. Default: `1e-4`.

        ppl_reduction (str): Reduction type to apply to the per-sample output values. Default: `mean`.

        ppl_sample_similarity (str): Name of the sample similarity to use in PPL metric computation (see :ref:`registry
            <Registry>`). Default: `lpips-vgg16`.

        ppl_sample_similarity_resize (int): Force samples to this size when computing similarity, unless set to `None`.
            Default: `64`.

        ppl_sample_similarity_dtype (str): Check samples are of compatible dtype when computing similarity, unless set
            to `None`. Default: `uint8`.

        ppl_discard_percentile_lower (int): Removes the lower percentile of samples before reduction. Default: `1`.

        ppl_discard_percentile_higher (int): Removes the higher percentile of samples before reduction. Default: `99`.

        ppl_z_interp_mode (str): Noise interpolation mode in PPL (see :ref:`registry <Registry>`). Default: `lerp`.

        prc_neighborhood (int): Number of nearest neighbours to consider in PRC. Default: `3`.

        prc_batch_size (int): Batch size in PRC. Default: `10000`.

        samples_shuffle (bool): Perform random samples shuffling before computing splits. Default: `True`.

        samples_find_deep (bool): Find all samples in paths recursively. Default: `False`.

        samples_find_ext (str): List of comma-separated extensions (no blanks) to look for when traversing input path.
            Default: `png,jpg,jpeg`.

        samples_ext_lossy (str): List of comma-separated extensions (no blanks) to warn about lossy compression.
            Default: `jpg,jpeg`.

        samples_resize_and_crop (int): Transform all images found in the directory to a given size and square shape.
            Default: 0 (disabled).

        datasets_root (str): Path to built-in torchvision datasets root. Default: `$ENV_TORCH_HOME/fidelity_datasets`.

        datasets_download (bool): Download torchvision datasets to :paramref:`~calculate_metrics.dataset_root`.
            Default: `True`.

        cache_root (str): Path to file cache for features and statistics. Default: `$ENV_TORCH_HOME/fidelity_cache`.

        cache (bool): Use file cache for features and statistics. Default: `True`.

        input1_cache_name (str): Assigns a cache entry to input1 (when not a registered input) and forces caching of
            features on it. Default: `None`.

        input1_model_z_type (str): Type of noise, only required when the input is a path to a generator model (see
            :ref:`registry <Registry>`). Default: `normal`.

        input1_model_z_size (int): Dimensionality of noise (only required when the input is a path to a generator
            model). Default: `None`.

        input1_model_num_classes (int): Number of classes for conditional (0 for unconditional) generation (only
            required when the input is a path to a generator model). Default: `0`.

        input1_model_num_samples (int): Number of samples to draw (only required when the input is a generator model).
            This option affects the following metrics: ISC, FID, KID. Default: `None`.

        input2_cache_name (str): Assigns a cache entry to input2 (when not a registered input) and forces caching of
            features on it. Default: `None`.

        input2_model_z_type (str): Type of noise, only required when the input is a path to a generator model (see
            :ref:`registry <Registry>`). Default: `normal`.

        input2_model_z_size (int): Dimensionality of noise (only required when the input is a path to a generator
            model). Default: `None`.

        input2_model_num_classes (int): Number of classes for conditional (0 for unconditional) generation (only
            required when the input is a path to a generator model). Default: `0`.

        input2_model_num_samples (int): Number of samples to draw (only required when the input is a generator model).
            This option affects the following metrics: FID, KID. Default: `None`.

        rng_seed (int): Random numbers generator seed for all operations involving randomness. Default: `2020`.

        save_cpu_ram (bool): Use less CPU RAM at the cost of speed. May not lead to improvement with every metric.
            Default: `False`.

        verbose (bool): Output progress information to STDERR. Default: `True`.

    Returns:

        : Dictionary of metrics with a subset of the following keys:

            - :const:`torch_fidelity.KEY_METRIC_ISC_MEAN`
            - :const:`torch_fidelity.KEY_METRIC_ISC_STD`
            - :const:`torch_fidelity.KEY_METRIC_FID`
            - :const:`torch_fidelity.KEY_METRIC_KID_MEAN`
            - :const:`torch_fidelity.KEY_METRIC_KID_STD`
            - :const:`torch_fidelity.KEY_METRIC_PPL_MEAN`
            - :const:`torch_fidelity.KEY_METRIC_PPL_STD`
            - :const:`torch_fidelity.KEY_METRIC_PPL_RAW`
            - :const:`torch_fidelity.KEY_METRIC_PRECISION`
            - :const:`torch_fidelity.KEY_METRIC_RECALL`
            - :const:`torch_fidelity.KEY_METRIC_F_SCORE`
    """

    have_isc = get_kwarg('isc', kwargs)
    have_fid = get_kwarg('fid', kwargs)
    have_kid = get_kwarg('kid', kwargs)
    have_prc = get_kwarg('prc', kwargs)
    fe_name = get_kwarg('feature_extractor', kwargs)

    have_default_fe_inception = have_isc or have_fid or have_kid
    have_default_fe_vgg = have_prc

    if fe_name is not None or not (have_default_fe_inception and have_default_fe_vgg):
        # using the same non-default feature extractor for all metrics except ppl, or using just one default extractor
        return calculate_metrics_one_feature_extractor(**kwargs)

    out = {}
    kwargs_subset = dict(**kwargs)
    kwargs_subset['prc'] = False
    out.update(calculate_metrics_one_feature_extractor(**kwargs_subset))
    kwargs_subset = dict(**kwargs)
    kwargs_subset['isc'] = False
    kwargs_subset['fid'] = False
    kwargs_subset['kid'] = False
    kwargs_subset['ppl'] = False
    out.update(calculate_metrics_one_feature_extractor(**kwargs_subset))

    return out
