#!/usr/bin/env python3
import argparse
import json
import os

from torch_fidelity.metric_fid import calculate_fid
from torch_fidelity.metric_isc import calculate_isc
from torch_fidelity.metric_kid import calculate_kid
from torch_fidelity.metrics import calculate_metrics
from torch_fidelity.registry import FEATURE_EXTRACTORS_REGISTRY, DATASETS_REGISTRY

DEFAULTS = {
    'cuda': True,
    'batch_size': 64,
    'isc': False,
    'fid': False,
    'kid': False,
    'feature_extractor': 'inception-v3-compat',
    'feature_layer_isc': 'logits_unbiased',
    'feature_layer_fid': '2048',
    'feature_layer_kid': '2048',
    'feature_extractor_weights_path': None,
    'isc_splits': 10,
    'kid_subsets': 100,
    'kid_subset_size': 1000,
    'kid_degree': 3,
    'kid_gamma': None,
    'kid_coef0': 1,
    'shuffle_on': True,
    'glob_recursively': False,
    'datasets_root': None,
    'datasets_download_on': True,
    'cache_root': None,
    'cache_off': False,
    'rng_seed': 2020,
    'verbose': True,
}


def default_kwargs():
    return DEFAULTS.copy()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input1', type=str,
                        help=f'First path to images or a registered input source (one of {DATASETS_REGISTRY.keys()})')
    parser.add_argument('input2', type=str, nargs='?', default=None,
                        help=f'Second (optional) path to images or a registered input source (one of '
                             f'{DATASETS_REGISTRY.keys()})')
    parser.add_argument('-b', '--batch-size', type=int, default=DEFAULTS['batch_size'],
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
    parser.add_argument('--feature-extractor', default=DEFAULTS['feature_extractor'], type=str,
                        choices=FEATURE_EXTRACTORS_REGISTRY.keys(),
                        help='Name of the feature extractor')
    parser.add_argument('--feature-layer-isc', default=DEFAULTS['feature_layer_isc'], type=str,
                        help='Name of the feature layer to use with ISC (Inception Score) metric')
    parser.add_argument('--feature-layer-fid', default=DEFAULTS['feature_layer_fid'], type=str,
                        help='Name of the feature layer to use with FID metric')
    parser.add_argument('--feature-layer-kid', default=DEFAULTS['feature_layer_kid'], type=str,
                        help='Name of the feature layer to use with KID metric')
    parser.add_argument('--feature-extractor-weights-path', default=DEFAULTS['feature_extractor_weights_path'], type=str,
                        help='Path to feature extractor weights')
    parser.add_argument('--isc-splits', default=DEFAULTS['isc_splits'], type=int,
                        help='Number of splits in Inception Score')
    parser.add_argument('--kid-subsets', default=DEFAULTS['kid_subsets'], type=int,
                        help='Number of subsets in KID')
    parser.add_argument('--kid-subset-size', default=DEFAULTS['kid_subset_size'], type=int,
                        help='Subset size in KID')
    parser.add_argument('--kid-degree', default=DEFAULTS['kid_degree'], type=int,
                        help='')
    parser.add_argument('--kid-gamma', default=DEFAULTS['kid_gamma'], type=float,
                        help='')
    parser.add_argument('--kid-coef0', default=DEFAULTS['kid_coef0'], type=float,
                        help='')
    parser.add_argument('--shuffle-off', action='store_true',
                        help='Do not perform samples shuffling using RNG before computing splits')
    parser.add_argument('--glob-recursively', action='store_true',
                        help='Find all images in paths recursively')
    parser.add_argument('--datasets-root', default=DEFAULTS['datasets_root'], type=str,
                        help='Path to built-in torchvision datasets root. Defaults to $ENV_TORCH_HOME/fidelity_datasets')
    parser.add_argument('--datasets-download-off', action='store_true',
                        help='Do not download torchvision datasets to dataset_root')
    parser.add_argument('--cache-root', default=DEFAULTS['cache_root'], type=str,
                        help='Path to file cache for features and statistics. Defaults to $ENV_TORCH_HOME/fidelity_cache')
    parser.add_argument('--cache-off', action='store_true',
                        help='Do not use file cache for features and statistics')
    parser.add_argument('--rng-seed', default=DEFAULTS['rng_seed'], type=int,
                        help='Random numbers generator seed for ISC and KID splits')
    parser.add_argument('--silent', dest='silent', action='store_true',
                        help='Do not output progress information to STDERR')

    args = parser.parse_args()

    assert args.isc or args.fid or args.kid, 'Specify one or a few metrics: --isc, --fid, --kid'

    args.verbose = not args.silent
    args.datasets_download_on = not args.datasets_download_off
    args.shuffle_on = not args.shuffle_off

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.cuda = not args.cpu and os.environ.get('CUDA_VISIBLE_DEVICES', '') != ''

    if args.isc and not args.fid and not args.kid:
        metrics = calculate_isc(args.input1, **vars(args))
    elif not args.isc and args.fid and not args.kid:
        metrics = calculate_fid(args.input1, args.input2, **vars(args))
    elif not args.isc and not args.fid and args.kid:
        metrics = calculate_kid(args.input1, args.input2, **vars(args))
    else:
        metrics = calculate_metrics(args.input1, input_2=args.input2, **vars(args))

    if args.json:
        print(json.dumps(metrics, indent=4))
    else:
        print(', '.join((f'{k}: {v}' for k, v in metrics.items())))


if __name__ == '__main__':
    main()
