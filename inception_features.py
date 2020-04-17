import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from interpolate_as_tensorflow import interpolate_bilinear_2d_like_tensorflow1x

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

TF_INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
PT_INCEPTION_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3Features(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(
            self,
            output_blocks=[DEFAULT_BLOCK_INDEX],
            resize_input=True,
            normalize_input=True,
            normalize_mean=128,
            normalize_stddev=128,
            requires_grad=False,
            tensorflow_compatibility=True,
            inception_weights_path=None,
    ):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input range to the range the
            pretrained Inception network expects, namely (-1, 1)
        normalize_mean : int or float or Tensor
            Mean of the input data (default: 128 for 24 bit RGB)
        normalize_stddev : int or float or Tensor
            Standard deviation of the input data (default: 128 for 24 bit RGB)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        tensorflow_compatibility : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results
        inception_weights_path: str
            Path to the pretrained Inception model weights in PyTorch format.
            Refer to inception_features.py:__main__ for making your own.
            By default downloads the checkpoint from internet.
        """
        super(InceptionV3Features, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.normalize_mean = normalize_mean
        self.normalize_stddev = normalize_stddev
        self.tensorflow_compatibility = tensorflow_compatibility
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if tensorflow_compatibility:
            inception = fid_inception_v3(inception_weights_path)
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            if self.tensorflow_compatibility:
                x = interpolate_bilinear_2d_like_tensorflow1x(x, size=(299, 299), align_corners=False)
            else:
                x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.normalize_input:
            x = (x - self.normalize_mean) / self.normalize_stddev

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return tuple(outp)


def fid_inception_v3(path):
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = models.inception_v3(num_classes=1008,
                                    aux_logits=False,
                                    pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    if path is None:
        state_dict = load_state_dict_from_url(PT_INCEPTION_URL, progress=True)
    else:
        state_dict = torch.load(path)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


def check_or_download_inception_graphdef():
    import os
    import pathlib
    import tarfile
    from urllib import request
    import tempfile
    tempfile.gettempdir()
    model_file = pathlib.Path(os.path.join(tempfile.gettempdir(), 'classify_image_graph_def.pb'))
    if not model_file.exists():
        print("Downloading Inception model")
        fn, _ = request.urlretrieve(TF_INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def load_weights_from_graphdef():
    import tensorflow as tf
    from tensorflow.python.framework import tensor_util
    path = check_or_download_inception_graphdef()
    with tf.io.gfile.GFile(path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        weights = {
            w.name: tensor_util.MakeNdarray(w.attr['value'].tensor)
            for w in graph_def.node
            if w.op == 'Const'
        }
    return weights


def convert_tensorflow_graphdef_to_pytorch_checkpoint(path_out=None):
    map_tf_to_pt = {
        # obtained through semi-automatic matching by the shapes of tensors
        'conv/batchnorm': 'Conv2d_1a_3x3.bn',
        'conv/conv2d_params': 'Conv2d_1a_3x3.conv',
        'conv_1/batchnorm': 'Conv2d_2a_3x3.bn',
        'conv_1/conv2d_params': 'Conv2d_2a_3x3.conv',
        'conv_2/batchnorm': 'Conv2d_2b_3x3.bn',
        'conv_2/conv2d_params': 'Conv2d_2b_3x3.conv',
        'conv_3/batchnorm': 'Conv2d_3b_1x1.bn',
        'conv_3/conv2d_params': 'Conv2d_3b_1x1.conv',
        'conv_4/batchnorm': 'Conv2d_4a_3x3.bn',
        'conv_4/conv2d_params': 'Conv2d_4a_3x3.conv',
        'mixed/conv/batchnorm': 'Mixed_5b.branch1x1.bn',
        'mixed/conv/conv2d_params': 'Mixed_5b.branch1x1.conv',
        'mixed/tower_1/conv/batchnorm': 'Mixed_5b.branch3x3dbl_1.bn',
        'mixed/tower_1/conv/conv2d_params': 'Mixed_5b.branch3x3dbl_1.conv',
        'mixed/tower_1/conv_1/batchnorm': 'Mixed_5b.branch3x3dbl_2.bn',
        'mixed/tower_1/conv_1/conv2d_params': 'Mixed_5b.branch3x3dbl_2.conv',
        'mixed/tower_1/conv_2/batchnorm': 'Mixed_5b.branch3x3dbl_3.bn',
        'mixed/tower_1/conv_2/conv2d_params': 'Mixed_5b.branch3x3dbl_3.conv',
        'mixed/tower/conv/batchnorm': 'Mixed_5b.branch5x5_1.bn',
        'mixed/tower/conv/conv2d_params': 'Mixed_5b.branch5x5_1.conv',
        'mixed/tower/conv_1/batchnorm': 'Mixed_5b.branch5x5_2.bn',
        'mixed/tower/conv_1/conv2d_params': 'Mixed_5b.branch5x5_2.conv',
        'mixed/tower_2/conv/batchnorm': 'Mixed_5b.branch_pool.bn',
        'mixed/tower_2/conv/conv2d_params': 'Mixed_5b.branch_pool.conv',
        'mixed_1/conv/batchnorm': 'Mixed_5c.branch1x1.bn',
        'mixed_1/conv/conv2d_params': 'Mixed_5c.branch1x1.conv',
        'mixed_1/tower_1/conv/batchnorm': 'Mixed_5c.branch3x3dbl_1.bn',
        'mixed_1/tower_1/conv/conv2d_params': 'Mixed_5c.branch3x3dbl_1.conv',
        'mixed_1/tower_1/conv_1/batchnorm': 'Mixed_5c.branch3x3dbl_2.bn',
        'mixed_1/tower_1/conv_1/conv2d_params': 'Mixed_5c.branch3x3dbl_2.conv',
        'mixed_1/tower_1/conv_2/batchnorm': 'Mixed_5c.branch3x3dbl_3.bn',
        'mixed_1/tower_1/conv_2/conv2d_params': 'Mixed_5c.branch3x3dbl_3.conv',
        'mixed_1/tower/conv/batchnorm': 'Mixed_5c.branch5x5_1.bn',
        'mixed_1/tower/conv/conv2d_params': 'Mixed_5c.branch5x5_1.conv',
        'mixed_1/tower/conv_1/batchnorm': 'Mixed_5c.branch5x5_2.bn',
        'mixed_1/tower/conv_1/conv2d_params': 'Mixed_5c.branch5x5_2.conv',
        'mixed_1/tower_2/conv/batchnorm': 'Mixed_5c.branch_pool.bn',
        'mixed_1/tower_2/conv/conv2d_params': 'Mixed_5c.branch_pool.conv',
        'mixed_2/conv/batchnorm': 'Mixed_5d.branch1x1.bn',
        'mixed_2/conv/conv2d_params': 'Mixed_5d.branch1x1.conv',
        'mixed_2/tower_1/conv/batchnorm': 'Mixed_5d.branch3x3dbl_1.bn',
        'mixed_2/tower_1/conv/conv2d_params': 'Mixed_5d.branch3x3dbl_1.conv',
        'mixed_2/tower_1/conv_1/batchnorm': 'Mixed_5d.branch3x3dbl_2.bn',
        'mixed_2/tower_1/conv_1/conv2d_params': 'Mixed_5d.branch3x3dbl_2.conv',
        'mixed_2/tower_1/conv_2/batchnorm': 'Mixed_5d.branch3x3dbl_3.bn',
        'mixed_2/tower_1/conv_2/conv2d_params': 'Mixed_5d.branch3x3dbl_3.conv',
        'mixed_2/tower/conv/batchnorm': 'Mixed_5d.branch5x5_1.bn',
        'mixed_2/tower/conv/conv2d_params': 'Mixed_5d.branch5x5_1.conv',
        'mixed_2/tower/conv_1/batchnorm': 'Mixed_5d.branch5x5_2.bn',
        'mixed_2/tower/conv_1/conv2d_params': 'Mixed_5d.branch5x5_2.conv',
        'mixed_2/tower_2/conv/batchnorm': 'Mixed_5d.branch_pool.bn',
        'mixed_2/tower_2/conv/conv2d_params': 'Mixed_5d.branch_pool.conv',
        'mixed_3/conv/batchnorm': 'Mixed_6a.branch3x3.bn',
        'mixed_3/conv/conv2d_params': 'Mixed_6a.branch3x3.conv',
        'mixed_3/tower/conv/batchnorm': 'Mixed_6a.branch3x3dbl_1.bn',
        'mixed_3/tower/conv/conv2d_params': 'Mixed_6a.branch3x3dbl_1.conv',
        'mixed_3/tower/conv_1/batchnorm': 'Mixed_6a.branch3x3dbl_2.bn',
        'mixed_3/tower/conv_1/conv2d_params': 'Mixed_6a.branch3x3dbl_2.conv',
        'mixed_3/tower/conv_2/batchnorm': 'Mixed_6a.branch3x3dbl_3.bn',
        'mixed_3/tower/conv_2/conv2d_params': 'Mixed_6a.branch3x3dbl_3.conv',
        'mixed_4/conv/batchnorm': 'Mixed_6b.branch1x1.bn',
        'mixed_4/conv/conv2d_params': 'Mixed_6b.branch1x1.conv',
        'mixed_4/tower/conv/batchnorm': 'Mixed_6b.branch7x7_1.bn',
        'mixed_4/tower/conv/conv2d_params': 'Mixed_6b.branch7x7_1.conv',
        'mixed_4/tower/conv_1/batchnorm': 'Mixed_6b.branch7x7_2.bn',
        'mixed_4/tower/conv_1/conv2d_params': 'Mixed_6b.branch7x7_2.conv',
        'mixed_4/tower/conv_2/batchnorm': 'Mixed_6b.branch7x7_3.bn',
        'mixed_4/tower/conv_2/conv2d_params': 'Mixed_6b.branch7x7_3.conv',
        'mixed_4/tower_1/conv/batchnorm': 'Mixed_6b.branch7x7dbl_1.bn',
        'mixed_4/tower_1/conv/conv2d_params': 'Mixed_6b.branch7x7dbl_1.conv',
        'mixed_4/tower_1/conv_1/batchnorm': 'Mixed_6b.branch7x7dbl_2.bn',
        'mixed_4/tower_1/conv_1/conv2d_params': 'Mixed_6b.branch7x7dbl_2.conv',
        'mixed_4/tower_1/conv_2/batchnorm': 'Mixed_6b.branch7x7dbl_3.bn',
        'mixed_4/tower_1/conv_2/conv2d_params': 'Mixed_6b.branch7x7dbl_3.conv',
        'mixed_4/tower_1/conv_3/batchnorm': 'Mixed_6b.branch7x7dbl_4.bn',
        'mixed_4/tower_1/conv_3/conv2d_params': 'Mixed_6b.branch7x7dbl_4.conv',
        'mixed_4/tower_1/conv_4/batchnorm': 'Mixed_6b.branch7x7dbl_5.bn',
        'mixed_4/tower_1/conv_4/conv2d_params': 'Mixed_6b.branch7x7dbl_5.conv',
        'mixed_4/tower_2/conv/batchnorm': 'Mixed_6b.branch_pool.bn',
        'mixed_4/tower_2/conv/conv2d_params': 'Mixed_6b.branch_pool.conv',
        'mixed_5/conv/batchnorm': 'Mixed_6c.branch1x1.bn',
        'mixed_5/conv/conv2d_params': 'Mixed_6c.branch1x1.conv',
        'mixed_5/tower/conv/batchnorm': 'Mixed_6c.branch7x7_1.bn',
        'mixed_5/tower/conv/conv2d_params': 'Mixed_6c.branch7x7_1.conv',
        'mixed_5/tower/conv_1/batchnorm': 'Mixed_6c.branch7x7_2.bn',
        'mixed_5/tower/conv_1/conv2d_params': 'Mixed_6c.branch7x7_2.conv',
        'mixed_5/tower/conv_2/batchnorm': 'Mixed_6c.branch7x7_3.bn',
        'mixed_5/tower/conv_2/conv2d_params': 'Mixed_6c.branch7x7_3.conv',
        'mixed_5/tower_1/conv/batchnorm': 'Mixed_6c.branch7x7dbl_1.bn',
        'mixed_5/tower_1/conv/conv2d_params': 'Mixed_6c.branch7x7dbl_1.conv',
        'mixed_5/tower_1/conv_1/batchnorm': 'Mixed_6c.branch7x7dbl_2.bn',
        'mixed_5/tower_1/conv_1/conv2d_params': 'Mixed_6c.branch7x7dbl_2.conv',
        'mixed_5/tower_1/conv_2/batchnorm': 'Mixed_6c.branch7x7dbl_3.bn',
        'mixed_5/tower_1/conv_2/conv2d_params': 'Mixed_6c.branch7x7dbl_3.conv',
        'mixed_5/tower_1/conv_3/batchnorm': 'Mixed_6c.branch7x7dbl_4.bn',
        'mixed_5/tower_1/conv_3/conv2d_params': 'Mixed_6c.branch7x7dbl_4.conv',
        'mixed_5/tower_1/conv_4/batchnorm': 'Mixed_6c.branch7x7dbl_5.bn',
        'mixed_5/tower_1/conv_4/conv2d_params': 'Mixed_6c.branch7x7dbl_5.conv',
        'mixed_5/tower_2/conv/batchnorm': 'Mixed_6c.branch_pool.bn',
        'mixed_5/tower_2/conv/conv2d_params': 'Mixed_6c.branch_pool.conv',
        'mixed_6/conv/batchnorm': 'Mixed_6d.branch1x1.bn',
        'mixed_6/conv/conv2d_params': 'Mixed_6d.branch1x1.conv',
        'mixed_6/tower/conv/batchnorm': 'Mixed_6d.branch7x7_1.bn',
        'mixed_6/tower/conv/conv2d_params': 'Mixed_6d.branch7x7_1.conv',
        'mixed_6/tower/conv_1/batchnorm': 'Mixed_6d.branch7x7_2.bn',
        'mixed_6/tower/conv_1/conv2d_params': 'Mixed_6d.branch7x7_2.conv',
        'mixed_6/tower/conv_2/batchnorm': 'Mixed_6d.branch7x7_3.bn',
        'mixed_6/tower/conv_2/conv2d_params': 'Mixed_6d.branch7x7_3.conv',
        'mixed_6/tower_1/conv/batchnorm': 'Mixed_6d.branch7x7dbl_1.bn',
        'mixed_6/tower_1/conv/conv2d_params': 'Mixed_6d.branch7x7dbl_1.conv',
        'mixed_6/tower_1/conv_1/batchnorm': 'Mixed_6d.branch7x7dbl_2.bn',
        'mixed_6/tower_1/conv_1/conv2d_params': 'Mixed_6d.branch7x7dbl_2.conv',
        'mixed_6/tower_1/conv_2/batchnorm': 'Mixed_6d.branch7x7dbl_3.bn',
        'mixed_6/tower_1/conv_2/conv2d_params': 'Mixed_6d.branch7x7dbl_3.conv',
        'mixed_6/tower_1/conv_3/batchnorm': 'Mixed_6d.branch7x7dbl_4.bn',
        'mixed_6/tower_1/conv_3/conv2d_params': 'Mixed_6d.branch7x7dbl_4.conv',
        'mixed_6/tower_1/conv_4/batchnorm': 'Mixed_6d.branch7x7dbl_5.bn',
        'mixed_6/tower_1/conv_4/conv2d_params': 'Mixed_6d.branch7x7dbl_5.conv',
        'mixed_6/tower_2/conv/batchnorm': 'Mixed_6d.branch_pool.bn',
        'mixed_6/tower_2/conv/conv2d_params': 'Mixed_6d.branch_pool.conv',
        'mixed_7/conv/batchnorm': 'Mixed_6e.branch1x1.bn',
        'mixed_7/conv/conv2d_params': 'Mixed_6e.branch1x1.conv',
        'mixed_7/tower/conv/batchnorm': 'Mixed_6e.branch7x7_1.bn',
        'mixed_7/tower/conv/conv2d_params': 'Mixed_6e.branch7x7_1.conv',
        'mixed_7/tower/conv_1/batchnorm': 'Mixed_6e.branch7x7_2.bn',
        'mixed_7/tower/conv_1/conv2d_params': 'Mixed_6e.branch7x7_2.conv',
        'mixed_7/tower/conv_2/batchnorm': 'Mixed_6e.branch7x7_3.bn',
        'mixed_7/tower/conv_2/conv2d_params': 'Mixed_6e.branch7x7_3.conv',
        'mixed_7/tower_1/conv/batchnorm': 'Mixed_6e.branch7x7dbl_1.bn',
        'mixed_7/tower_1/conv/conv2d_params': 'Mixed_6e.branch7x7dbl_1.conv',
        'mixed_7/tower_1/conv_1/batchnorm': 'Mixed_6e.branch7x7dbl_2.bn',
        'mixed_7/tower_1/conv_1/conv2d_params': 'Mixed_6e.branch7x7dbl_2.conv',
        'mixed_7/tower_1/conv_2/batchnorm': 'Mixed_6e.branch7x7dbl_3.bn',
        'mixed_7/tower_1/conv_2/conv2d_params': 'Mixed_6e.branch7x7dbl_3.conv',
        'mixed_7/tower_1/conv_3/batchnorm': 'Mixed_6e.branch7x7dbl_4.bn',
        'mixed_7/tower_1/conv_3/conv2d_params': 'Mixed_6e.branch7x7dbl_4.conv',
        'mixed_7/tower_1/conv_4/batchnorm': 'Mixed_6e.branch7x7dbl_5.bn',
        'mixed_7/tower_1/conv_4/conv2d_params': 'Mixed_6e.branch7x7dbl_5.conv',
        'mixed_7/tower_2/conv/batchnorm': 'Mixed_6e.branch_pool.bn',
        'mixed_7/tower_2/conv/conv2d_params': 'Mixed_6e.branch_pool.conv',
        'mixed_8/tower/conv/batchnorm': 'Mixed_7a.branch3x3_1.bn',
        'mixed_8/tower/conv/conv2d_params': 'Mixed_7a.branch3x3_1.conv',
        'mixed_8/tower/conv_1/batchnorm': 'Mixed_7a.branch3x3_2.bn',
        'mixed_8/tower/conv_1/conv2d_params': 'Mixed_7a.branch3x3_2.conv',
        'mixed_8/tower_1/conv/batchnorm': 'Mixed_7a.branch7x7x3_1.bn',
        'mixed_8/tower_1/conv/conv2d_params': 'Mixed_7a.branch7x7x3_1.conv',
        'mixed_8/tower_1/conv_1/batchnorm': 'Mixed_7a.branch7x7x3_2.bn',
        'mixed_8/tower_1/conv_1/conv2d_params': 'Mixed_7a.branch7x7x3_2.conv',
        'mixed_8/tower_1/conv_2/batchnorm': 'Mixed_7a.branch7x7x3_3.bn',
        'mixed_8/tower_1/conv_2/conv2d_params': 'Mixed_7a.branch7x7x3_3.conv',
        'mixed_8/tower_1/conv_3/batchnorm': 'Mixed_7a.branch7x7x3_4.bn',
        'mixed_8/tower_1/conv_3/conv2d_params': 'Mixed_7a.branch7x7x3_4.conv',
        'mixed_9/conv/batchnorm': 'Mixed_7b.branch1x1.bn',
        'mixed_9/conv/conv2d_params': 'Mixed_7b.branch1x1.conv',
        'mixed_9/tower/conv/batchnorm': 'Mixed_7b.branch3x3_1.bn',
        'mixed_9/tower/conv/conv2d_params': 'Mixed_7b.branch3x3_1.conv',
        'mixed_9/tower/mixed/conv/batchnorm': 'Mixed_7b.branch3x3_2a.bn',
        'mixed_9/tower/mixed/conv/conv2d_params': 'Mixed_7b.branch3x3_2a.conv',
        'mixed_9/tower/mixed/conv_1/batchnorm': 'Mixed_7b.branch3x3_2b.bn',
        'mixed_9/tower/mixed/conv_1/conv2d_params': 'Mixed_7b.branch3x3_2b.conv',
        'mixed_9/tower_1/conv/batchnorm': 'Mixed_7b.branch3x3dbl_1.bn',
        'mixed_9/tower_1/conv/conv2d_params': 'Mixed_7b.branch3x3dbl_1.conv',
        'mixed_9/tower_1/conv_1/batchnorm': 'Mixed_7b.branch3x3dbl_2.bn',
        'mixed_9/tower_1/conv_1/conv2d_params': 'Mixed_7b.branch3x3dbl_2.conv',
        'mixed_9/tower_1/mixed/conv/batchnorm': 'Mixed_7b.branch3x3dbl_3a.bn',
        'mixed_9/tower_1/mixed/conv/conv2d_params': 'Mixed_7b.branch3x3dbl_3a.conv',
        'mixed_9/tower_1/mixed/conv_1/batchnorm': 'Mixed_7b.branch3x3dbl_3b.bn',
        'mixed_9/tower_1/mixed/conv_1/conv2d_params': 'Mixed_7b.branch3x3dbl_3b.conv',
        'mixed_9/tower_2/conv/batchnorm': 'Mixed_7b.branch_pool.bn',
        'mixed_9/tower_2/conv/conv2d_params': 'Mixed_7b.branch_pool.conv',
        'mixed_10/conv/batchnorm': 'Mixed_7c.branch1x1.bn',
        'mixed_10/conv/conv2d_params': 'Mixed_7c.branch1x1.conv',
        'mixed_10/tower/conv/batchnorm': 'Mixed_7c.branch3x3_1.bn',
        'mixed_10/tower/conv/conv2d_params': 'Mixed_7c.branch3x3_1.conv',
        'mixed_10/tower/mixed/conv/batchnorm': 'Mixed_7c.branch3x3_2a.bn',
        'mixed_10/tower/mixed/conv/conv2d_params': 'Mixed_7c.branch3x3_2a.conv',
        'mixed_10/tower/mixed/conv_1/batchnorm': 'Mixed_7c.branch3x3_2b.bn',
        'mixed_10/tower/mixed/conv_1/conv2d_params': 'Mixed_7c.branch3x3_2b.conv',
        'mixed_10/tower_1/conv/batchnorm': 'Mixed_7c.branch3x3dbl_1.bn',
        'mixed_10/tower_1/conv/conv2d_params': 'Mixed_7c.branch3x3dbl_1.conv',
        'mixed_10/tower_1/conv_1/batchnorm': 'Mixed_7c.branch3x3dbl_2.bn',
        'mixed_10/tower_1/conv_1/conv2d_params': 'Mixed_7c.branch3x3dbl_2.conv',
        'mixed_10/tower_1/mixed/conv/batchnorm': 'Mixed_7c.branch3x3dbl_3a.bn',
        'mixed_10/tower_1/mixed/conv/conv2d_params': 'Mixed_7c.branch3x3dbl_3a.conv',
        'mixed_10/tower_1/mixed/conv_1/batchnorm': 'Mixed_7c.branch3x3dbl_3b.bn',
        'mixed_10/tower_1/mixed/conv_1/conv2d_params': 'Mixed_7c.branch3x3dbl_3b.conv',
        'mixed_10/tower_2/conv/batchnorm': 'Mixed_7c.branch_pool.bn',
        'mixed_10/tower_2/conv/conv2d_params': 'Mixed_7c.branch_pool.conv',
        'softmax/weights': 'fc.weight',
        'softmax/biases': 'fc.bias',
    }

    weights_tf = load_weights_from_graphdef()

    weights_pt = {}
    for k_tf, k_pt in map_tf_to_pt.items():
        if k_tf.endswith('conv2d_params'):
            weight_tf = torch.from_numpy(weights_tf.pop(k_tf))
            weight_pt = weight_tf.permute(3, 2, 0, 1).contiguous()
            weights_pt[k_pt + '.weight'] = weight_pt
        elif k_tf.endswith('batchnorm'):
            weights_pt[k_pt + '.weight'] = torch.from_numpy(weights_tf.pop(k_tf + '/gamma'))
            weights_pt[k_pt + '.bias'] = torch.from_numpy(weights_tf.pop(k_tf + '/beta'))
            weights_pt[k_pt + '.running_mean'] = torch.from_numpy(weights_tf.pop(k_tf + '/moving_mean'))
            weights_pt[k_pt + '.running_var'] = torch.from_numpy(weights_tf.pop(k_tf + '/moving_variance'))
        elif k_tf == 'softmax/weights':
            weight_tf = torch.from_numpy(weights_tf.pop(k_tf))
            weight_pt = weight_tf.permute(1, 0).contiguous()
            weights_pt[k_pt] = weight_pt
        elif k_tf == 'softmax/biases':
            weights_pt[k_pt] = torch.from_numpy(weights_tf.pop(k_tf))
        else:
            raise NotImplementedError(f'Cannot handle TensorFlow GraphDef Const item {k_tf}')

    if path_out is None:
        path_out = 'pt-inception-2015-12-05.pth'

    torch.save(weights_pt, path_out)


if __name__ == '__main__':
    print('Converting TensorFlow InceptionV3 pretrained weights to a PyTorch checkpoint...')
    convert_tensorflow_graphdef_to_pytorch_checkpoint()
    print('Done')