from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.metric_fid import fid_inputs_to_metric, fid_featuresdict_to_statistics_cached, \
    fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_featuresdict_to_metric
from torch_fidelity.metric_kid import kid_featuresdict_to_metric
from torch_fidelity.metric_ppl import calculate_ppl
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached, \
    get_cacheable_input_name


def calculate_metrics(**kwargs):
    """
    Calculates metrics for the given inputs. Keyword arguments:

    .. _ISC: https://arxiv.org/pdf/1606.03498.pdf
    .. _FID: https://arxiv.org/pdf/1706.08500.pdf
    .. _KID: https://arxiv.org/pdf/1801.01401.pdf
    .. _PPL: https://arxiv.org/pdf/1812.04948.pdf

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

        ppl (bool): Calculate PPL_ (Perceptual Path Length). Default: `False`.

        feature_extractor (str): Name of the feature extractor (see :ref:`registry <Registry>`). Default:
            `inception-v3-compat`.

        feature_layer_isc (str): Name of the feature layer to use with ISC metric. Default: `logits_unbiased`.

        feature_layer_fid (str): Name of the feature layer to use with FID metric. Default: `"2048"`.

        feature_layer_kid (str): Name of the feature layer to use with KID metric. Default: `"2048"`.

        feature_extractor_weights_path (str): Path to feature extractor weights (downloaded if `None`). Default: `None`.

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

        samples_shuffle (bool): Perform random samples shuffling before computing splits. Default: `True`.

        samples_find_deep (bool): Find all samples in paths recursively. Default: `False`.

        samples_find_ext (str): List of comma-separated extensions (no blanks) to look for when traversing input path.
            Default: `png,jpg,jpeg`.

        samples_ext_lossy (str): List of comma-separated extensions (no blanks) to warn about lossy compression.
            Default: `jpg,jpeg`.

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
    """

    verbose = get_kwarg('verbose', kwargs)
    input1, input2 = get_kwarg('input1', kwargs), get_kwarg('input2', kwargs)

    have_isc = get_kwarg('isc', kwargs)
    have_fid = get_kwarg('fid', kwargs)
    have_kid = get_kwarg('kid', kwargs)
    have_ppl = get_kwarg('ppl', kwargs)

    need_input1 = have_isc or have_fid or have_kid or have_ppl
    need_input2 = have_fid or have_kid

    vassert(
        have_isc or have_fid or have_kid or have_ppl,
        'At least one of "isc", "fid", "kid", "ppl" metrics must be specified'
    )
    vassert(input1 is not None or not need_input1, 'First input is required for "isc", "fid", "kid", and "ppl" metrics')
    vassert(input2 is not None or not need_input2, 'Second input is required for "fid" and "kid" metrics')

    metrics = {}

    if have_isc or have_fid or have_kid:
        feature_extractor = get_kwarg('feature_extractor', kwargs)
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

        feat_extractor = create_feature_extractor(feature_extractor, list(feature_layers), **kwargs)

        # isc: input - featuresdict(cached) - metric
        # fid: input - featuresdict(cached) - statistics(cached) - metric
        # kid: input - featuresdict(cached) - metric

        if (not have_isc) and have_fid and (not have_kid):
            # shortcut for a case when statistics are cached and features are not required on at least one input
            metric_fid = fid_inputs_to_metric(feat_extractor, **kwargs)
            metrics.update(metric_fid)
        else:
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

    if have_ppl:
        metric_ppl = calculate_ppl(1, **kwargs)
        metrics.update(metric_ppl)

    return metrics
