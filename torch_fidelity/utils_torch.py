import os
import tempfile

import torch

from torch_fidelity.helpers import vprint


def torch_maybe_compile(module, dummy_input, verbose):
    out = module
    try:
        compiled = torch.compile(module)
        try:
            compiled(dummy_input)
            vprint(verbose, "Feature extractor compiled")
            setattr(out, "forward_pure", out.forward)
            setattr(out, "forward", compiled)
        except Exception:
            vprint(verbose, "Feature extractor compiled, but failed to run. Falling back to pure torch")
    except Exception as e:
        vprint(verbose, "Feature extractor compilation failed. Falling back to pure torch")
    return out


def torch_atomic_save(what, path):
    path = os.path.expanduser(path)
    path_dir = os.path.dirname(path)
    fp = tempfile.NamedTemporaryFile(delete=False, dir=path_dir)
    try:
        torch.save(what, fp)
        fp.close()
        os.rename(fp.name, path)
    finally:
        fp.close()
        if os.path.exists(fp.name):
            os.remove(fp.name)
