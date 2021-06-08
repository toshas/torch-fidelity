import os

from torch_fidelity.datasets import TransformPILtoRGBTensor, Cifar10_RGB, STL10_RGB
from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.helpers import vassert
from torch_fidelity.noise import random_normal, random_unit, random_uniform_0_1, batch_lerp, batch_slerp_any, \
    batch_slerp_unit
from torch_fidelity.sample_similarity_base import SampleSimilarityBase
from torch_fidelity.sample_similarity_lpips import SampleSimilarityLPIPS

DATASETS_REGISTRY = dict()
FEATURE_EXTRACTORS_REGISTRY = dict()
SAMPLE_SIMILARITY_REGISTRY = dict()
NOISE_SOURCE_REGISTRY = dict()
INTERPOLATION_REGISTRY = dict()


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


def register_sample_similarity(name, cls):
    r"""
    Register a new sample similarity (useful for extending sample similarity measures beyond LPIPS-VGG16 for 2D images).
    Args:
        name: str
            A unique name of the sample similarity class.
        cls: subclass(SampleSimilarityBase)
            Name of a class subclassed from SampleSimilarityBase, implementing a new sample similarity.
    """
    vassert(type(name) is str, 'Sample similarity must be given a name')
    vassert(name.strip() == name, 'Name must not have leading or trailing whitespaces')
    vassert(os.path.sep not in name, 'Name must not contain path delimiters (slash/backslash)')
    vassert(name not in SAMPLE_SIMILARITY_REGISTRY, f'Sample similarity "{name}" is already registered')
    vassert(
        issubclass(cls, SampleSimilarityBase), 'Sample similarity class must be subclassed from SampleSimilarityBase'
    )
    SAMPLE_SIMILARITY_REGISTRY[name] = cls


def register_noise_source(name, fn_generate):
    r"""
    Register a new noise source, which can generate samples to be used as inputs to generative models.
    Args:
        name: str
            A unique name of the noise source.
        fn_generate: callable(rng, shape)
            A generator of a random sample of specified type and shape. The passed arguments denote an initialized RNG
            instance and the desired sample shape.
    """
    vassert(type(name) is str, 'Noise source must be given a name')
    vassert(name.strip() == name, 'Name must not have leading or trailing whitespaces')
    vassert(os.path.sep not in name, 'Name must not contain path delimiters (slash/backslash)')
    vassert(name not in NOISE_SOURCE_REGISTRY, f'Noise source "{name}" is already registered')
    vassert(
        callable(fn_generate),
        'Noise source must be provided as a callable (function, lambda) with 2 arguments: rng, shape'
    )
    NOISE_SOURCE_REGISTRY[name] = fn_generate


def register_interpolation(name, fn_interpolate):
    r"""
    Register a new interpolation method, adhering to the interface `fn(a, b, t), where a and b are the endpoints,
    and t is a float in the [0,1] range.
    Args:
        name: str
            A unique name of the interpolation method.
        fn_interpolate: callable(a, b, t)
            An interpolation function of the specified interface.
    """
    vassert(type(name) is str, 'Interpolation must be given a name')
    vassert(name.strip() == name, 'Name must not have leading or trailing whitespaces')
    vassert(os.path.sep not in name, 'Name must not contain path delimiters (slash/backslash)')
    vassert(name not in INTERPOLATION_REGISTRY, f'Interpolation "{name}" is already registered')
    vassert(
        callable(fn_interpolate),
        'Interpolation must be provided as a callable (function, lambda) with 3 arguments: a, b, t'
    )
    INTERPOLATION_REGISTRY[name] = fn_interpolate


register_dataset(
    'cifar10-train',
    lambda root, download: Cifar10_RGB(root, train=True, transform=TransformPILtoRGBTensor(), download=download)
)
register_dataset(
    'cifar10-val',
    lambda root, download: Cifar10_RGB(root, train=False, transform=TransformPILtoRGBTensor(), download=download)
)
register_dataset(
    'stl10-train',
    lambda root, download: STL10_RGB(root, split='train', transform=TransformPILtoRGBTensor(), download=download)
)
register_dataset(
    'stl10-test',
    lambda root, download: STL10_RGB(root, split='test', transform=TransformPILtoRGBTensor(), download=download)
)
register_dataset(
    'stl10-unlabeled',
    lambda root, download: STL10_RGB(root, split='unlabeled', transform=TransformPILtoRGBTensor(), download=download)
)

register_feature_extractor('inception-v3-compat', FeatureExtractorInceptionV3)

register_sample_similarity('lpips-vgg16', SampleSimilarityLPIPS)

register_noise_source('normal', random_normal)
register_noise_source('unit', random_unit)
register_noise_source('uniform_0_1', random_uniform_0_1)

register_interpolation('lerp', batch_lerp)
register_interpolation('slerp_any', batch_slerp_any)
register_interpolation('slerp_unit', batch_slerp_unit)
