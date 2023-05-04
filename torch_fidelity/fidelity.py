#!/usr/bin/env python3
import argparse
import json
import os
import sys

from torch_fidelity.defaults import DEFAULTS
from torch_fidelity.metrics import calculate_metrics
from torch_fidelity.registry import FEATURE_EXTRACTORS_REGISTRY, DATASETS_REGISTRY, SAMPLE_SIMILARITY_REGISTRY, \
    INTERPOLATION_REGISTRY, NOISE_SOURCE_REGISTRY


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input1', default=DEFAULTS['input1'], type=str,
                        help=f'First input, which can be either a path to a directory with samples, or one of the '
                             f'registered input sources ({DATASETS_REGISTRY.keys()}, or a path to a generative model '
                             f'in the ONNX or PTH (JIT) formats. In the latter case, the following arguments must also '
                             f'be provided: --input1-model-z-type, --input1-model-z-size, --input1-model-num-classes, '
                             f'and --input1-model-num-samples.')
    parser.add_argument('--input2', default=DEFAULTS['input2'], type=str,
                        help=f'Second input, which can be either a path to a directory with samples, or one of the '
                             f'registered input sources ({DATASETS_REGISTRY.keys()}, or a path to a generative model '
                             f'in the ONNX or PTH (JIT) format. In the latter case, the following arguments must also '
                             f'be provided: --input2-model-z-type, --input2-model-z-size, --input2-model-num-classes, '
                             f'and --input2-model-num-samples.')
    parser.add_argument('-b', '--batch-size', default=DEFAULTS['batch_size'], type=int,
                        help='Batch size to use')
    pgroup = parser.add_mutually_exclusive_group()
    pgroup.add_argument('-g', '--gpu', default=None, type=str,
                        help='Use CUDA (overrides CUDA_VISIBLE_DEVICES)')
    pgroup.add_argument('-c', '--cpu', action='store_true',
                        help='Use CPU despite capabilities')
    parser.add_argument('-j', '--json', action='store_true',
                        help='Print scores in JSON')
    parser.add_argument('-i', '--isc', action='store_true',
                        help='Calculate ISC (Inception Score)')
    parser.add_argument('-f', '--fid', action='store_true',
                        help='Calculate FID (Frechet Inception Distance)')
    parser.add_argument('-k', '--kid', action='store_true',
                        help='Calculate KID (Kernel Inception Distance)')
    parser.add_argument('-r', '--prc', action='store_true',
                        help='Calculate PRC (Precision and Recall)')
    parser.add_argument('-p', '--ppl', action='store_true',
                        help='Calculate PPL (Perceptual Path Length)')
    parser.add_argument('--feature-extractor', default=DEFAULTS['feature_extractor'], type=str,
                        choices=FEATURE_EXTRACTORS_REGISTRY.keys(),
                        help='Name of the feature extractor (default if None)')
    parser.add_argument('--feature-layer-isc', default=DEFAULTS['feature_layer_isc'], type=str,
                        help='Name of the feature layer to use with ISC metric (default if None)')
    parser.add_argument('--feature-layer-fid', default=DEFAULTS['feature_layer_fid'], type=str,
                        help='Name of the feature layer to use with FID metric (default if None)')
    parser.add_argument('--feature-layer-kid', default=DEFAULTS['feature_layer_kid'], type=str,
                        help='Name of the feature layer to use with KID metric (default if None)')
    parser.add_argument('--feature-layer-prc', default=DEFAULTS['feature_layer_prc'], type=str,
                        help='Name of the feature layer to use with PRC metrics (default if None)')
    parser.add_argument('--feature-extractor-weights-path',
                        default=DEFAULTS['feature_extractor_weights_path'], type=str,
                        help='Path to feature extractor weights (downloaded if None)')
    parser.add_argument('--feature-extractor-internal-dtype',
                        default=DEFAULTS['feature_extractor_internal_dtype'], type=str,
                        choices=['float32', 'float64'],
                        help='dtype to use inside the feature extractor (default if None)')
    parser.add_argument('--feature-extractor-compile', action='store_true',
                        help='Compile feature extractor (experimental: may have negative effect on metrics numerical '
                             'precision)')
    parser.add_argument('--isc-splits', default=DEFAULTS['isc_splits'], type=int,
                        help='Number of splits in ISC')
    parser.add_argument('--kid-subsets', default=DEFAULTS['kid_subsets'], type=int,
                        help='Number of subsets in KID')
    parser.add_argument('--kid-subset-size', default=DEFAULTS['kid_subset_size'], type=int,
                        help='Subset size in KID')
    parser.add_argument('--kid-degree', default=DEFAULTS['kid_degree'], type=int,
                        help='Degree of polynomial kernel in KID')
    parser.add_argument('--kid-gamma', default=DEFAULTS['kid_gamma'], type=float,
                        help='Polynomial kernel gamma in KID')
    parser.add_argument('--kid-coef0', default=DEFAULTS['kid_coef0'], type=float,
                        help='Polynomial kernel coef0 in KID')
    parser.add_argument('--ppl-epsilon', default=DEFAULTS['ppl_epsilon'], type=float,
                        help='Interpolation step size in PPL')
    parser.add_argument('--ppl-reduction', default=DEFAULTS['ppl_reduction'], type=str,
                        choices=('mean', 'none'),
                        help='Reduction type to apply to the per-sample output values')
    parser.add_argument('--ppl-sample-similarity', default=DEFAULTS['ppl_sample_similarity'], type=str,
                        choices=list(SAMPLE_SIMILARITY_REGISTRY.keys()),
                        help='Name of the sample similarity to use in PPL metric computation')
    parser.add_argument('--ppl-sample-similarity-resize', default=DEFAULTS['ppl_sample_similarity_resize'], type=int,
                        help='Force samples to this size when computing similarity, unless set to None')
    parser.add_argument('--ppl-sample-similarity-dtype', default=DEFAULTS['ppl_sample_similarity_dtype'], type=str,
                        help='Check samples are of compatible dtype when computing similarity, unless set to None')
    parser.add_argument('--ppl-discard-percentile-lower', default=DEFAULTS['ppl_discard_percentile_lower'], type=int,
                        help='Removes the lower percentile of samples before reduction')
    parser.add_argument('--ppl-discard-percentile-higher', default=DEFAULTS['ppl_discard_percentile_higher'], type=int,
                        help='Removes the higher percentile of samples before reduction')
    parser.add_argument('--ppl-z-interp-mode', default=DEFAULTS['ppl_z_interp_mode'], type=str,
                        choices=list(INTERPOLATION_REGISTRY.keys()),
                        help='Noise interpolation mode in PPL')
    parser.add_argument('--prc-neighborhood', default=DEFAULTS['prc_neighborhood'], type=int,
                        help='Number of nearest neighbours in PRC')
    parser.add_argument('--prc-batch-size', default=DEFAULTS['prc_batch_size'], type=int,
                        help='Batch size in PRC')
    parser.add_argument('--no-samples-shuffle', action='store_true',
                        help='Do not perform samples shuffling before computing splits')
    parser.add_argument('--samples-find-deep', action='store_true',
                        help='Find all samples in paths recursively')
    parser.add_argument('--samples-find-ext', default=DEFAULTS['samples_find_ext'], type=str,
                        help=f'List of extensions to look for when traversing input path')
    parser.add_argument('--samples-ext-lossy', default=DEFAULTS['samples_ext_lossy'], type=str,
                        help=f'List of extensions to warn about lossy compression')
    parser.add_argument('--samples-resize-and-crop', default=DEFAULTS['samples_resize_and_crop'], type=int,
                        help=f'Transform all images found in the directory to a given size and square shape')
    parser.add_argument('--datasets-root', default=DEFAULTS['datasets_root'], type=str,
                        help='Path to built-in torchvision datasets root. '
                             'Defaults to $ENV_TORCH_HOME/fidelity_datasets')
    parser.add_argument('--no-datasets-download', action='store_true',
                        help='Do not download torchvision datasets to dataset_root')
    parser.add_argument('--cache-root', default=DEFAULTS['cache_root'], type=str,
                        help='Path to file cache for features and statistics. '
                             'Defaults to $ENV_TORCH_HOME/fidelity_cache')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not use file cache for features and statistics')
    parser.add_argument('--input1-cache-name', default=DEFAULTS['input1_cache_name'], type=str,
                        help='Assigns a cache entry to input1 (when not a registered input) and forces caching of '
                             'features on it.')
    parser.add_argument('--input1-model-z-type', default=DEFAULTS['input1_model_z_type'], type=str,
                        choices=list(NOISE_SOURCE_REGISTRY.keys()),
                        help='Type of noise (only required when the input is a path to a generator model)')
    parser.add_argument('--input1-model-z-size', default=DEFAULTS['input1_model_z_size'], type=int,
                        help='Dimensionality of noise (only required when the input is a path to a generator model)')
    parser.add_argument('--input1-model-num-classes', default=DEFAULTS['input1_model_num_classes'], type=int,
                        help='Number of classes for conditional (0 for unconditional) generation (only required when '
                             'the input is a path to a generator model)')
    parser.add_argument('--input1-model-num-samples', default=DEFAULTS['input1_model_num_samples'], type=int,
                        help='Number of samples to draw (only required when the input is a generator model). '
                             'This option affects the following metrics: ISC, FID, KID')
    parser.add_argument('--input2-cache-name', default=DEFAULTS['input2_cache_name'], type=str,
                        help='Assigns a cache entry to input2 (when not a registered input) and forces caching of '
                             'features on it.')
    parser.add_argument('--input2-model-z-type', default=DEFAULTS['input2_model_z_type'], type=str,
                        choices=list(NOISE_SOURCE_REGISTRY.keys()),
                        help='Type of noise (only required when the input is a path to a generator model)')
    parser.add_argument('--input2-model-z-size', default=DEFAULTS['input2_model_z_size'], type=int,
                        help='Dimensionality of noise (only required when the input is a path to a generator model)')
    parser.add_argument('--input2-model-num-classes', default=DEFAULTS['input2_model_num_classes'], type=int,
                        help='Number of classes for conditional (0 for unconditional) generation (only required when '
                             'the input is a path to a generator model)')
    parser.add_argument('--input2-model-num-samples', default=DEFAULTS['input2_model_num_samples'], type=int,
                        help='Number of samples to draw (only required when the input is a generator model). '
                             'This option affects the following metrics: ISC, FID, KID')
    parser.add_argument('--rng-seed', default=DEFAULTS['rng_seed'], type=int,
                        help='Random numbers generator seed for all operations involving randomness')
    parser.add_argument('--save-cpu-ram', action='store_true',
                        help='Use less CPU RAM at the cost of speed')
    parser.add_argument('--silent', action='store_true',
                        help='Do not output progress information to STDERR')

    args, unknown = parser.parse_known_args()
    if type(unknown) is list and len(unknown) > 0:
        print(f'Ignoring unrecognized command line options: {unknown}', file=sys.stderr)
        print(f'  This may be due the command line options change in the most recent version,', file=sys.stderr)
        print(f'  Use ''fidelity --help'' to see the up-to-date command line options,', file=sys.stderr)
        print(f'  See https://github.com/toshas/torch-fidelity/blob/master/CHANGELOG.md', file=sys.stderr)

    args.verbose = not args.silent
    args.datasets_download = not args.no_datasets_download
    args.samples_shuffle = not args.no_samples_shuffle
    args.cache = not args.no_cache

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.cuda = not args.cpu and os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''

    metrics = calculate_metrics(**vars(args))

    if args.json:
        print(json.dumps(metrics, indent=4))
    else:
        print('\n'.join((f'{k}: {v:.7g}' for k, v in metrics.items())))


if __name__ == '__main__':
    main()
