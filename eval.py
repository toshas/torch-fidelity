#!/usr/bin/env python3
import argparse
import json
import os

from metric_FID import calculate_metric_of_paths
from metric_IS import calculate_metric_of_path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help='Path to the generated images or to .npz statistic files')
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('-m', '--model', default=None, type=str,
                    help='Path to Inception model weights, if downloading needs to be skipped')
parser.add_argument('-s', '--silent', action='store_true',
                    help='Verbose or silent progress bar and messages')
parser.add_argument('-j', '--json', action='store_true',
                    help='Print scores in JSON')

args = parser.parse_args()

if args.gpu != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

metrics = calculate_metric_of_paths(
    args.path,
    False,
    args.batch_size,
    args.gpu != '',
    args.model,
    not args.silent,
)


# metrics = calculate_metric_of_path(
#     args.path[0],
#     False,
#     args.batch_size,
#     args.gpu != '',
#     splits=10,
#     shuffle=True, shuffle_seed=2020,
#     model_path=None, verbose=True
# )

if args.json:
    print(json.dumps(metrics, indent=4))
else:
    print(', '.join((f'{k}: {v}' for k, v in metrics.items())))
