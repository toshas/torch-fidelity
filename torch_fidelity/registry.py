import os

from torch_fidelity.datasets import TransformPILtoRGBTensor, Cifar10_RGB, Cifar100_RGB, STL10_RGB
from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.feature_extractor_clip import FeatureExtractorCLIP
from torch_fidelity.feature_extractor_dinov2 import FeatureExtractorDinoV2
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.feature_extractor_vgg16 import FeatureExtractorVGG16
from torch_fidelity.helpers import vassert
from torch_fidelity.noise import (
    random_normal,
    random_unit,
    random_uniform_0_1,
    batch_lerp,
    batch_slerp_any,
    batch_slerp_unit,
)
from torch_fidelity.sample_similarity_base import SampleSimilarityBase
from torch_fidelity.sample_similarity_lpips import SampleSimilarityLPIPS

DATASETS_REGISTRY = dict()
FEATURE_EXTRACTORS_REGISTRY = dict()
SAMPLE_SIMILARITY_REGISTRY = dict()
NOISE_SOURCE_REGISTRY = dict()
INTERPOLATION_REGISTRY = dict()


def register_dataset(name, fn_create):
    """
    Registers a new input source.

    Args:

        name (str): Unique name of the input source.

        fn_create (callable): A constructor of a :class:`~torch:torch.utils.data.Dataset` instance. Callable arguments:

            - `root` (str): Location where the dataset files may be downloaded.
            - `download` (bool): Whether to perform downloading or rely on the cached version.
    """
    vassert(type(name) is str, "Dataset must be given a name")
    vassert(name.strip() == name, "Name must not have leading or trailing whitespaces")
    vassert(os.path.sep not in name, "Name must not contain path delimiters (slash/backslash)")
    vassert(name not in DATASETS_REGISTRY, f'Dataset "{name}" is already registered')
    vassert(
        callable(fn_create),
        "Dataset must be provided as a callable (function, lambda) with 2 bool arguments: root, download",
    )
    DATASETS_REGISTRY[name] = fn_create


def register_feature_extractor(name, cls):
    """
    Registers a new feature extractor.

    Args:

        name (str): Unique name of the feature extractor.

        cls (FeatureExtractorBase): Instance of :class:`FeatureExtractorBase`, implementing a new feature extractor.
    """
    vassert(type(name) is str, "Feature extractor must be given a name")
    vassert(name.strip() == name, "Name must not have leading or trailing whitespaces")
    vassert(os.path.sep not in name, "Name must not contain path delimiters (slash/backslash)")
    vassert(name not in FEATURE_EXTRACTORS_REGISTRY, f'Feature extractor "{name}" is already registered')
    vassert(
        issubclass(cls, FeatureExtractorBase), "Feature extractor class must be subclassed from FeatureExtractorBase"
    )
    FEATURE_EXTRACTORS_REGISTRY[name] = cls


def register_sample_similarity(name, cls):
    """
    Registers a new sample similarity measure.

    Args:

        name (str): Unique name of the sample similarity measure.

        cls (SampleSimilarityBase): Instance of :class:`SampleSimilarityBase`, implementing a new sample similarity
            measure.
    """
    vassert(type(name) is str, "Sample similarity must be given a name")
    vassert(name.strip() == name, "Name must not have leading or trailing whitespaces")
    vassert(os.path.sep not in name, "Name must not contain path delimiters (slash/backslash)")
    vassert(name not in SAMPLE_SIMILARITY_REGISTRY, f'Sample similarity "{name}" is already registered')
    vassert(
        issubclass(cls, SampleSimilarityBase), "Sample similarity class must be subclassed from SampleSimilarityBase"
    )
    SAMPLE_SIMILARITY_REGISTRY[name] = cls


def register_noise_source(name, fn_generate):
    """
    Registers a new noise source, which can generate samples to be used as inputs to generative models.

    Args:

        name (str): Unique name of the noise source.

        fn_generate (callable): Generator of a random samples of specified type and shape. Callable arguments:

            - `rng` (numpy.random.RandomState): random number generator state, initialized with \
                :paramref:`~calculate_metrics.seed`.
            - `shape` (torch.Size): shape of the tensor of random samples.
    """
    vassert(type(name) is str, "Noise source must be given a name")
    vassert(name.strip() == name, "Name must not have leading or trailing whitespaces")
    vassert(os.path.sep not in name, "Name must not contain path delimiters (slash/backslash)")
    vassert(name not in NOISE_SOURCE_REGISTRY, f'Noise source "{name}" is already registered')
    vassert(
        callable(fn_generate),
        "Noise source must be provided as a callable (function, lambda) with 2 arguments: rng, shape",
    )
    NOISE_SOURCE_REGISTRY[name] = fn_generate


def register_interpolation(name, fn_interpolate):
    """
    Registers a new sample interpolation method.

    Args:

        name (str): Unique name of the interpolation method.

        fn_interpolate (callable): Sample interpolation function. Callable arguments:

            - `a` (torch.Tensor): batch of the first endpoint samples.
            - `b` (torch.Tensor): batch of the second endpoint samples.
            - `t` (float): interpolation coefficient in the range [0,1].
    """
    vassert(type(name) is str, "Interpolation must be given a name")
    vassert(name.strip() == name, "Name must not have leading or trailing whitespaces")
    vassert(os.path.sep not in name, "Name must not contain path delimiters (slash/backslash)")
    vassert(name not in INTERPOLATION_REGISTRY, f'Interpolation "{name}" is already registered')
    vassert(
        callable(fn_interpolate),
        "Interpolation must be provided as a callable (function, lambda) with 3 arguments: a, b, t",
    )
    INTERPOLATION_REGISTRY[name] = fn_interpolate


register_dataset(
    "cifar10-train",
    lambda root, download: Cifar10_RGB(root, train=True, transform=TransformPILtoRGBTensor(), download=download),
)
register_dataset(
    "cifar10-val",
    lambda root, download: Cifar10_RGB(root, train=False, transform=TransformPILtoRGBTensor(), download=download),
)
register_dataset(
    "cifar100-train",
    lambda root, download: Cifar100_RGB(root, train=True, transform=TransformPILtoRGBTensor(), download=download),
)
register_dataset(
    "cifar100-val",
    lambda root, download: Cifar100_RGB(root, train=False, transform=TransformPILtoRGBTensor(), download=download),
)
register_dataset(
    "stl10-train",
    lambda root, download: STL10_RGB(root, split="train", transform=TransformPILtoRGBTensor(), download=download),
)
register_dataset(
    "stl10-test",
    lambda root, download: STL10_RGB(root, split="test", transform=TransformPILtoRGBTensor(), download=download),
)
register_dataset(
    "stl10-unlabeled",
    lambda root, download: STL10_RGB(root, split="unlabeled", transform=TransformPILtoRGBTensor(), download=download),
)

register_feature_extractor("inception-v3-compat", FeatureExtractorInceptionV3)

register_feature_extractor("vgg16", FeatureExtractorVGG16)

register_feature_extractor("clip-rn50", FeatureExtractorCLIP)
register_feature_extractor("clip-rn101", FeatureExtractorCLIP)
register_feature_extractor("clip-rn50x4", FeatureExtractorCLIP)
register_feature_extractor("clip-rn50x16", FeatureExtractorCLIP)
register_feature_extractor("clip-rn50x64", FeatureExtractorCLIP)
register_feature_extractor("clip-vit-b-32", FeatureExtractorCLIP)
register_feature_extractor("clip-vit-b-16", FeatureExtractorCLIP)
register_feature_extractor("clip-vit-l-14", FeatureExtractorCLIP)
register_feature_extractor("clip-vit-l-14-336px", FeatureExtractorCLIP)

register_feature_extractor("dinov2-vit-s-14", FeatureExtractorDinoV2)
register_feature_extractor("dinov2-vit-b-14", FeatureExtractorDinoV2)
register_feature_extractor("dinov2-vit-l-14", FeatureExtractorDinoV2)
register_feature_extractor("dinov2-vit-g-14", FeatureExtractorDinoV2)

register_sample_similarity("lpips-vgg16", SampleSimilarityLPIPS)

register_noise_source("normal", random_normal)
register_noise_source("unit", random_unit)
register_noise_source("uniform_0_1", random_uniform_0_1)

register_interpolation("lerp", batch_lerp)
register_interpolation("slerp_any", batch_slerp_any)
register_interpolation("slerp_unit", batch_slerp_unit)
