import os

from torch_fidelity.datasets import Cifar10_RGB, TransformPILtoRGBTensor
from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.helpers import vassert

DATASETS_REGISTRY = dict()
FEATURE_EXTRACTORS_REGISTRY = dict()


def register_dataset(name, fn_create):
    r"""
    Register a new input source (useful for ground truth or reference datasets).
    Args:
        name: str
            A unique name of the input source, which will be available for use as a positional input argument. See
            calculate_metrics function.
        fn_create: callable(root, download)
            A constructor of torch.util.data.Dataset instance. The passed arguments denote a possible root where the
            dataset may be downloaded.
    """
    vassert(type(name) is str, 'Dataset must be given a name')
    vassert(name.strip() == name, 'Name must not have leading or trailing whitespaces')
    vassert(os.path.sep not in name, 'Name must not contain path delimiters (slash/backslash)')
    vassert(name not in DATASETS_REGISTRY, f'Dataset "{name}" is already registered')
    vassert(
        callable(fn_create),
        'Dataset must be provided as a callable (function, lambda) with 2 bool arguments: root, download'
    )
    DATASETS_REGISTRY[name] = fn_create


def register_feature_extractor(name, cls):
    r"""
    Register a new feature extractor (useful for extending metrics beyond Inception 2D feature extractor).
    Args:
        name: str
            A unique name of the feature extractor, which will be available for use as a value of the
            "feature_extractor" argument. See calculate_metrics function.
        cls: subclass(FeatureExtractorBase)
            Name of a class subclassed from FeatureExtractorBase, implementing a new feature extractor.
    """
    vassert(type(name) is str, 'Feature extractor must be given a name')
    vassert(name.strip() == name, 'Name must not have leading or trailing whitespaces')
    vassert(os.path.sep not in name, 'Name must not contain path delimiters (slash/backslash)')
    vassert(name not in FEATURE_EXTRACTORS_REGISTRY, f'Feature extractor "{name}" is already registered')
    vassert(
        issubclass(cls, FeatureExtractorBase), 'Feature extractor class must be subclassed from FeatureExtractorBase'
    )
    FEATURE_EXTRACTORS_REGISTRY[name] = cls


register_dataset(
    'cifar10-train',
    lambda root, download: Cifar10_RGB(root, train=True, transform=TransformPILtoRGBTensor(), download=download)
)
register_dataset(
    'cifar10-val',
    lambda root, download: Cifar10_RGB(root, train=False, transform=TransformPILtoRGBTensor(), download=download)
)

register_feature_extractor('inception-v3-compat', FeatureExtractorInceptionV3)
