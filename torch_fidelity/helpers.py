import json
import sys

import torch

from torch_fidelity.defaults import DEFAULTS


def vassert(truecond, message):
    if not truecond:
        raise ValueError(message)


def vprint(verbose, message):
    if verbose:
        print(message, file=sys.stderr)


def get_kwarg(name, kwargs):
    return kwargs.get(name, DEFAULTS[name])


def json_decode_string(s):
    try:
        out = json.loads(s)
    except json.JSONDecodeError as e:
        print(f'Failed to decode JSON string: {s}', file=sys.stderr)
        raise
    return out


def text_to_dtype(name, default=None):
    DTYPES = {
        'uint8': torch.uint8,
        'float32': torch.float32,
        'float64': torch.float32,
    }
    if default in DTYPES:
        default = DTYPES[default]
    return DTYPES.get(name, default)
