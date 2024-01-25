# Portions of source code adapted from the following sources:
#   https://github.com/openai/CLIP/blob/main/clip/model.py (d50d76d on Jul 27, 2022)
#   https://github.com/openai/CLIP/blob/main/clip/clip.py  (c5478aa on Jul 27, 2022)
#   Distributed under MIT License: https://github.com/openai/CLIP/blob/main/LICENSE
import sys
import time
import warnings
from collections import OrderedDict
from contextlib import redirect_stdout
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.hub import load_state_dict_from_url

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert, vprint, text_to_dtype, get_kwarg
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x

MODEL_BASE_URL = "https://openaipublic.azureedge.net/clip/models"
MODEL_METADATA = {
    "clip-rn50": {
        "hash": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
        "filename": "RN50",
    },
    "clip-rn101": {
        "hash": "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599",
        "filename": "RN101",
    },
    "clip-rn50x4": {
        "hash": "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd",
        "filename": "RN50x4",
    },
    "clip-rn50x16": {
        "hash": "52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa",
        "filename": "RN50x16",
    },
    "clip-rn50x64": {
        "hash": "be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c",
        "filename": "RN50x64",
    },
    "clip-vit-b-32": {
        "hash": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        "filename": "ViT-B-32",
    },
    "clip-vit-b-16": {
        "hash": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
        "filename": "ViT-B-16",
    },
    "clip-vit-l-14": {
        "hash": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        "filename": "ViT-L-14",
    },
    "clip-vit-l-14-336px": {
        "hash": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
        "filename": "ViT-L-14-336px",
    },
}
MODEL_URLS = {k: f'{MODEL_BASE_URL}/{v["hash"]}/{v["filename"]}.pt' for k, v in MODEL_METADATA.items()}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            out = F.layer_norm(
                x.to(torch.float32),
                self.normalized_shape,
                self.weight.to(torch.float32),
                self.bias.to(torch.float32),
                self.eps,
            ).to(orig_dtype)
        else:
            out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return out


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIPVisual(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
    ):
        super().__init__()
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict, feature_extractor_internal_dtype):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        vassert(output_width**2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0], "Bad checkpoint")
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]

    model = CLIPVisual(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
    )

    for key in {
        "input_resolution",
        "context_length",
        "vocab_size",
        "positional_embedding",
        "text_projection",
        "logit_scale",
        "token_embedding.weight",
    }:
        if key in state_dict:
            del state_dict[key]
    state_dict = {
        k: v for k, v in state_dict.items() if not (k.startswith("transformer.") or k.startswith("ln_final."))
    }

    convert_weights(model)
    model.load_state_dict(state_dict)
    model.to(feature_extractor_internal_dtype)
    model.requires_grad_(False)
    model.eval()
    return model


class FeatureExtractorCLIP(FeatureExtractorBase):
    def __init__(
        self,
        name,
        features_list,
        feature_extractor_weights_path=None,
        feature_extractor_internal_dtype=None,
        **kwargs,
    ):
        """
        CLIP feature extractor for 2D RGB 24bit images.

        Args:

            name (str): Unique name of the feature extractor, must be the same as used in
                :func:`register_feature_extractor`.

            features_list (list): A list of the requested feature names, which will be produced for each input. This
                feature extractor provides the following features:

                - 'clip'

            feature_extractor_weights_path (str): Path to the pretrained CLIP model weights in PyTorch format.
                Downloads from internet if `None`.

            feature_extractor_internal_dtype (str): dtype to use inside the feature extractor. Specifying it may improve
                numerical precision in some cases. Supported values are 'float32' (default), and 'float64'.
        """
        super(FeatureExtractorCLIP, self).__init__(name, features_list)
        vassert(name in MODEL_URLS, f"Model {name} not found; available models = {list(MODEL_URLS.keys())}")
        vassert(
            feature_extractor_internal_dtype in ("float32", "float64", None),
            "Only 32-bit floats are supported for internal dtype of this feature extractor",
        )
        self.feature_extractor_internal_dtype = text_to_dtype(feature_extractor_internal_dtype, "float32")

        verbose = get_kwarg("verbose", kwargs)

        model_jit = None
        if feature_extractor_weights_path is None:
            with redirect_stdout(sys.stderr), warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="'torch.load' received a zip file that looks like a TorchScript archive"
                )
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                for attempt in range(10):
                    try:
                        model_jit = load_state_dict_from_url(
                            MODEL_URLS[name],
                            map_location="cpu",
                            progress=verbose,
                            check_hash=True,
                            file_name=f'{name}-{MODEL_METADATA[name]["hash"]}.pt',
                        )
                        break
                    except RuntimeError as e:
                        if "invalid hash value" not in str(e) or attempt == 9:
                            raise e
                        else:
                            vprint(verbose, "Download failed, retrying in 1 second")
                            time.sleep(1)
        else:
            model_jit = torch.jit.load(feature_extractor_weights_path, map_location="cpu")

        self.model = build_model(model_jit.state_dict(), self.feature_extractor_internal_dtype)
        self.resolution = self.model.visual.input_resolution
        self.requires_grad_(False)

    def forward(self, x):
        vassert(torch.is_tensor(x) and x.dtype == torch.uint8, "Expecting image as torch.Tensor with dtype=torch.uint8")
        vassert(x.dim() == 4 and x.shape[1] == 3, f"Input is not Bx3xHxW: {x.shape}")
        features = {}

        x = x.to(self.feature_extractor_internal_dtype)
        # N x 3 x ? x ?

        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(self.resolution, self.resolution),
            align_corners=False,
        )
        # N x 3 x R x R

        x = torchvision.transforms.functional.normalize(
            x,
            (255 * 0.48145466, 255 * 0.4578275, 255 * 0.40821073),
            (255 * 0.26862954, 255 * 0.26130258, 255 * 0.27577711),
            inplace=False,
        )
        # N x 3 x R x R

        x = self.model.visual(x)
        features["clip"] = x.to(torch.float32)

        return tuple(features[a] for a in self.features_list)

    @staticmethod
    def get_provided_features_list():
        return ("clip",)

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            "isc": "clip",
            "fid": "clip",
            "kid": "clip",
            "prc": "clip",
        }[metric]

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)
