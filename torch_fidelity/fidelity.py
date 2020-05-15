#!/usr/bin/env python3
import argparse
import json
import os

from torch_fidelity.defaults import DEFAULTS
from torch_fidelity.metrics import calculate_metrics
from torch_fidelity.registry import FEATURE_EXTRACTORS_REGISTRY, DATASETS_REGISTRY


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input1', default=DEFAULTS['input1'], type=str,
                        help=f'First path to samples or a registered input source (one of {DATASETS_REGISTRY.keys()})')
    parser.add_argument('--input2', default=DEFAULTS['input2'], type=str,
                        help=f'Second path to samples or a registered input source (one of {DATASETS_REGISTRY.keys()})')
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
    parser.add_argument('-p', '--ppl', action='store_true',
                        help='Calculate PPL (Perceptual Path Length)')
    parser.add_argument('-m', '--model', default=DEFAULTS['model'], type=str,
                        help='Path to generator model in ONNX format')
    parser.add_argument('--model-z-type', default=DEFAULTS['model_z_type'], type=str,
                        choices=('normal', 'uniform_0_1'),
                        help='Type of noise for generator model input')
    parser.add_argument('--model-z-size', default=DEFAULTS['model_z_size'], type=int,
                        help='Dimensionality of generator noise')
    parser.add_argument('--feature-extractor', default=DEFAULTS['feature_extractor'], type=str,
                        choices=FEATURE_EXTRACTORS_REGISTRY.keys(),
                        help='Name of the feature extractor')
    parser.add_argument('--feature-layer-isc', default=DEFAULTS['feature_layer_isc'], type=str,
                        help='Name of the feature layer to use with ISC metric')
    parser.add_argument('--feature-layer-fid', default=DEFAULTS['feature_layer_fid'], type=str,
                        help='Name of the feature layer to use with FID metric')
    parser.add_argument('--feature-layer-kid', default=DEFAULTS['feature_layer_kid'], type=str,
                        help='Name of the feature layer to use with KID metric')
    parser.add_argument('--feature-extractor-weights-path',
                        default=DEFAULTS['feature_extractor_weights_path'], type=str,
                        help='Path to feature extractor weights (downloaded if None)')
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
    parser.add_argument('--ppl-num-samples', default=DEFAULTS['ppl_num_samples'], type=int,
                        help='Number of samples to generate using the model in PPL')
    parser.add_argument('--ppl-epsilon', default=DEFAULTS['ppl_epsilon'], type=float,
                        help='Interpolation step size in PPL')
    parser.add_argument('--ppl-z-interp-mode', default=DEFAULTS['ppl_z_interp_mode'], type=str,
                        choices=('slerp', 'lerp'),
                        help='Noise interpolation mode in PPL')
    parser.add_argument('--no-samples-shuffle', action='store_true',
                        help='Do not perform samples shuffling before computing splits')
    parser.add_argument('--samples-find-deep', action='store_true',
                        help='Find all samples in paths recursively')
    parser.add_argument('--samples-find-ext', default=DEFAULTS['samples_find_ext'], type=str,
                        help=f'List of extensions to look for when traversing input path')
    parser.add_argument('--samples-ext-lossy', default=DEFAULTS['samples_ext_lossy'], type=str,
                        help=f'List of extensions to warn about lossy compression')
    parser.add_argument('--datasets-root', default=DEFAULTS['datasets_root'], type=str,
                        help='Path to built-in torchvision datasets root. Defaults to $ENV_TORCH_HOME/fidelity_datasets')
    parser.add_argument('--no-datasets-download', action='store_true',
                        help='Do not download torchvision datasets to dataset_root')
    parser.add_argument('--cache-root', default=DEFAULTS['cache_root'], type=str,
                        help='Path to file cache for features and statistics. Defaults to $ENV_TORCH_HOME/fidelity_cache')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not use file cache for features and statistics')
    parser.add_argument('--cache-input1-name', default=DEFAULTS['cache_input1_name'], type=str,
                        help='Assigns a cache entry to input1 (if a path) and forces caching of features on it')
    parser.add_argument('--cache-input2-name', default=DEFAULTS['cache_input2_name'], type=str,
                        help='Assigns a cache entry to input2 (if a path) and forces caching of features on it')
    parser.add_argument('--rng-seed', default=DEFAULTS['rng_seed'], type=int,
                        help='Random numbers generator seed for all operations involving randomness')
    parser.add_argument('--save-cpu-ram', action='store_true',
                        help='Use less CPU RAM at the cost of speed')
    parser.add_argument('--silent', action='store_true',
                        help='Do not output progress information to STDERR')

    args = parser.parse_args()

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
