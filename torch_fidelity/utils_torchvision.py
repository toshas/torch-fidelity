import sys
import warnings
from contextlib import redirect_stdout

import torchvision

from torch_fidelity.helpers import get_kwarg


def torchvision_load_pretrained_vgg16(**kwargs):
    verbose = get_kwarg("verbose", kwargs)
    with redirect_stdout(sys.stderr), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
        warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
        warnings.filterwarnings(
            "ignore",
            message="'torch.load' received a zip file that looks like a TorchScript "
            "archive dispatching to 'torch.jit.load'",
        )
        try:
            out = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1,
                progress=verbose,
            )
        except Exception:
            out = torchvision.models.vgg16(
                pretrained=True,
                progress=verbose,
            )
    return out
