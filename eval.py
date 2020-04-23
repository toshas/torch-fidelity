#!/usr/bin/env python3
import argparse
import json
import os

from metric_fid import fid_alone
from metric_isc import isc_alone
from registry import FEATURE_EXTRACTORS_REGISTRY

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help='Path to the generated images or to .npz statistic files')
parser.add_argument('-b', '--batch-size', type=int, default=64,
                    help='Batch size to use')
pgroup = parser.add_mutually_exclusive_group()
pgroup.add_argument('-g', '--gpu', default=None, type=str,
                    help='Use CUDA and override CUDA_VISIBLE_DEVICES')
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
parser.add_argument('--feature-extractor', default='inception-v3-compat', type=str,
                    choices=FEATURE_EXTRACTORS_REGISTRY.keys(),
                    help='Name of the feature extractor')
parser.add_argument('--feature-layer-isc', default='logits_unbiased', type=str,
                    help='Name of the feature layer to use with ISC (Inception Score) metric')
parser.add_argument('--feature-layer-fid', default='2048', type=str,
                    help='Name of the feature layer to use with FID metric')
parser.add_argument('--feature-layer-kid', default='2048', type=str,
                    help='Name of the feature layer to use with KID metric')
parser.add_argument('--feature-extractor-weights-path', default=None, type=str,
                    help='Path to feature extractor weights')
parser.add_argument('--isc-splits', default=10, type=int,
                    help='Number of splits in Inception Score')
parser.add_argument('--shuffle-off', action='store_true',
                    help='Do not perform samples shuffling using RNG before computing splits')
parser.add_argument('--glob-recursively', action='store_true',
                    help='Find all images in paths recursively')
parser.add_argument('--datasets-root', default=None, type=str,
                    help='Path to built-in torchvision datasets root. Defaults to $ENV_TORCH_HOME/fidelity_datasets')
parser.add_argument('--datasets-download-off', action='store_true',
                    help='Do not download torchvision datasets to dataset_root')
parser.add_argument('--cache-root', default=None, type=str,
                    help='Path to file cache for features and statistics. Defaults to $ENV_TORCH_HOME/fidelity_cache')
parser.add_argument('--cache-off', action='store_true',
                    help='Do not use file cache for features and statistics')
parser.add_argument('--rng-seed', default=2020, type=int,
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
    metrics = isc_alone(args.path[0], **vars(args))
elif not args.isc and args.fid and not args.kid:
    metrics = fid_alone(args.path[0], args.path[1], **vars(args))
else:
    raise NotImplementedError

if args.json:
    print(json.dumps(metrics, indent=4))
else:
    print(', '.join((f'{k}: {v}' for k, v in metrics.items())))
