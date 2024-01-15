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


class CleanStderr:
    def __init__(self, filter_phrases, stream=sys.stderr):
        self.filter_phrases = filter_phrases
        self.stream = stream

    def __enter__(self):
        sys.stderr = self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr = self.stream

    def write(self, msg):
        if not any(phrase in msg for phrase in self.filter_phrases):
            self.stream.write(msg)

    def flush(self):
        self.stream.flush()
