import torch
import torch.nn.functional as F
import torchvision

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert, text_to_dtype

from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x
from torch_fidelity.utils_torchvision import torchvision_load_pretrained_vgg16


class FeatureExtractorVGG16(FeatureExtractorBase):
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
        VGG16 feature extractor for 2D RGB 24bit images.

        Args:

            name (str): Unique name of the feature extractor, must be the same as used in
                :func:`register_feature_extractor`.

            features_list (list): A list of the requested feature names, which will be produced for each input. This
                feature extractor provides the following features:

                - 'fc2'
                - 'fc2_relu'

            feature_extractor_weights_path (str): Path to the pretrained InceptionV3 model weights in PyTorch format.
                Refer to `util_convert_inception_weights` for making your own. Downloads from internet if `None`.

            feature_extractor_internal_dtype (str): dtype to use inside the feature extractor. Specifying it may improve
                numerical precision in some cases. Supported values are 'float32' (default), and 'float64'.
        """
        super(FeatureExtractorVGG16, self).__init__(name, features_list)
        vassert(
            feature_extractor_internal_dtype in ("float32", "float64", None),
            "Only 32-bit floats are supported for internal dtype of this feature extractor",
        )
        self.feature_extractor_internal_dtype = text_to_dtype(feature_extractor_internal_dtype, "float32")

        if feature_extractor_weights_path is None:
            self.model = torchvision_load_pretrained_vgg16(**kwargs)
        else:
            state_dict = torch.load(feature_extractor_weights_path)
            self.model = torchvision.models.vgg16()
            self.model.load_state_dict(state_dict)
        for cls_tail_id in (6, 5, 4):
            del self.model.classifier[cls_tail_id]

        self.to(self.feature_extractor_internal_dtype)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        vassert(torch.is_tensor(x) and x.dtype == torch.uint8, "Expecting image as torch.Tensor with dtype=torch.uint8")
        vassert(x.dim() == 4 and x.shape[1] == 3, f"Input is not Bx3xHxW: {x.shape}")
        features = {}
        remaining_features = self.features_list.copy()

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

        if "fc2" in remaining_features:
            features["fc2"] = x.to(torch.float32)
            remaining_features.remove("fc2")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        features["fc2_relu"] = F.relu(x).to(torch.float32)

        return tuple(features[a] for a in self.features_list)

    @staticmethod
    def get_provided_features_list():
        return "fc2", "fc2_relu"

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            "isc": "fc2_relu",
            "fid": "fc2_relu",
            "kid": "fc2_relu",
            "prc": "fc2_relu",
        }[metric]

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)
