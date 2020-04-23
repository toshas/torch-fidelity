import os

from datasets import Cifar10_RGB, TransformPILtoRGBTensor
from feature_extractor_base import FeatureExtractorBase
from feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

DATASETS_REGISTRY = dict()
FEATURE_EXTRACTORS_REGISTRY = dict()


def register_dataset(name, ctor):
    assert type(name) is str, 'Dataset must be given a name'
    assert name.strip() == name, 'Name must not have leading or trailing whitespaces'
    assert os.path.sep not in name, 'Name must not contain path delimiters (slash/backslash)'
    assert name not in DATASETS_REGISTRY, f'Dataset "{name}" is already registered'
    assert callable(ctor), 'Dataset must be provided as a callable (function, lambda) with 2 bool arguments: ' \
                           'root, download'
    DATASETS_REGISTRY[name] = ctor


def register_feature_extractor(name, cls):
    assert type(name) is str, 'Feature extractor must be given a name'
    assert name.strip() == name, 'Name must not have leading or trailing whitespaces'
    assert os.path.sep not in name, 'Name must not contain path delimiters (slash/backslash)'
    assert name not in FEATURE_EXTRACTORS_REGISTRY, f'Feature extractor "{name}" is already registered'
    assert issubclass(cls, FeatureExtractorBase), 'Feature extractor class must be subclassed from FeatureExtractorBase'
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
