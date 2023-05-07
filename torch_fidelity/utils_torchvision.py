import sys
import warnings
from contextlib import redirect_stdout

import torchvision


def torchvision_load_pretrained_vgg16():
    with redirect_stdout(sys.stderr), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
        warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
        try:
            out = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            out = torchvision.models.vgg16(pretrained=True)
    return out
