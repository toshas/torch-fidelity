import sys
import warnings

import torch
import torchvision

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert, text_to_dtype, CleanStderr

from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x


MODEL_METADATA = {
    "dinov2-vit-s-14": "dinov2_vits14",  # dim=384
    "dinov2-vit-b-14": "dinov2_vitb14",  # dim=768
    "dinov2-vit-l-14": "dinov2_vitl14",  # dim=1024
    "dinov2-vit-g-14": "dinov2_vitg14",  # dim=1536
}


class FeatureExtractorDinoV2(FeatureExtractorBase):
    INPUT_IMAGE_SIZE = 224

    def __init__(
        self,
        name,
        features_list,
        feature_extractor_weights_path=None,
        feature_extractor_internal_dtype=None,
        **kwargs,
    ):
        """
        DinoV2 feature extractor for 2D RGB 24bit images.

        Args:

            name (str): Unique name of the feature extractor, must be the same as used in
                :func:`register_feature_extractor`.

            features_list (list): A list of the requested feature names, which will be produced for each input. This
                feature extractor provides the following features:

                - 'dinov2'

            feature_extractor_weights_path (str): Path to the pretrained InceptionV3 model weights in PyTorch format.
                Refer to `util_convert_inception_weights` for making your own. Downloads from internet if `None`.

            feature_extractor_internal_dtype (str): dtype to use inside the feature extractor. Specifying it may improve
                numerical precision in some cases. Supported values are 'float32' (default), and 'float64'.
        """
        super(FeatureExtractorDinoV2, self).__init__(name, features_list)
        vassert(
            feature_extractor_internal_dtype in ("float32", "float64", None),
            "Only 32-bit floats are supported for internal dtype of this feature extractor",
        )

        vassert(name in MODEL_METADATA, f"Model {name} not found; available models = {list(MODEL_METADATA.keys())}")
        self.feature_extractor_internal_dtype = text_to_dtype(feature_extractor_internal_dtype, "float32")

        with CleanStderr(["xFormers not available", "Using cache found in"], sys.stderr), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available")
            if feature_extractor_weights_path is None:
                self.model = torch.hub.load("facebookresearch/dinov2", MODEL_METADATA[name])
            else:
                raise NotImplementedError

        self.to(self.feature_extractor_internal_dtype)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        vassert(torch.is_tensor(x) and x.dtype == torch.uint8, "Expecting image as torch.Tensor with dtype=torch.uint8")
        vassert(x.dim() == 4 and x.shape[1] == 3, f"Input is not Bx3xHxW: {x.shape}")

        x = x.to(self.feature_extractor_internal_dtype)
        # N x 3 x ? x ?

        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
        # N x 3 x 224 x 224

        x = torchvision.transforms.functional.normalize(
            x,
            (255 * 0.485, 255 * 0.456, 255 * 0.406),
            (255 * 0.229, 255 * 0.224, 255 * 0.225),
            inplace=False,
        )
        # N x 3 x 224 x 224

        x = self.model(x)

        out = {
            "dinov2": x.to(torch.float32),
        }

        return tuple(out[a] for a in self.features_list)

    @staticmethod
    def get_provided_features_list():
        return ("dinov2",)

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            "isc": "dinov2",
            "fid": "dinov2",
            "kid": "dinov2",
            "prc": "dinov2",
        }[metric]

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)
