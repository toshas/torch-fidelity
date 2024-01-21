import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from PIL import Image
from tfdeterminism import patch as patch_tensorflow_for_determinism

from tests import TimeTrackingTestCase
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x
from torch_fidelity.utils import prepare_input_from_id, create_feature_extractor


class TestConvolution(TimeTrackingTestCase):
    @staticmethod
    def save(img, name):
        if torch.is_tensor(img):
            if img.dim() == 4:
                img = img.sum(dim=(0, 1))
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            img = 255 * (img - np.min(img)) / max(1e-12, (np.max(img) - np.min(img)))
            img = img.astype(np.uint8)
        Image.fromarray(img).save(name)

    @staticmethod
    def calc_output_dims(conv, x):
        y_height = (x.shape[2] + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // \
                   conv.stride[0] + 1  # fmt: skip
        y_width = (x.shape[3] + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // \
                  conv.stride[1] + 1  # fmt: skip
        return y_height, y_width

    @staticmethod
    def forward_pt_manualchw(conv, x):
        y_height, y_width = TestConvolution.calc_output_dims(conv, x)
        x = F.unfold(x, conv.kernel_size, conv.dilation, conv.padding, conv.stride)
        w = conv.weight.view(1, conv.weight.shape[0], -1)
        y = w.bmm(x)
        y = y.view(y.shape[0], y.shape[1], y_height, y_width)
        return y.cpu()

    @staticmethod
    def forward_pt_manualhwc(conv, x):
        y_height, y_width = TestConvolution.calc_output_dims(conv, x)
        x = F.unfold(x, conv.kernel_size, conv.dilation, conv.padding, conv.stride)
        x = (
            x.view(x.shape[0], conv.in_channels, *conv.kernel_size, x.shape[2])
            .permute(0, 2, 3, 1, 4)
            .reshape(x.shape[0], -1, x.shape[2])
        )
        w = conv.weight.permute(0, 2, 3, 1).reshape(1, conv.weight.shape[0], -1)
        y = w.bmm(x)
        y = y.view(y.shape[0], y.shape[1], y_height, y_width)
        return y.cpu()

    @staticmethod
    def forward_pt(conv_pt, x_pt):
        return F.conv2d(
            x_pt, conv_pt.weight, None, conv_pt.stride, conv_pt.padding, conv_pt.dilation, conv_pt.groups
        ).cpu()

    @staticmethod
    def forward_tf(conv_pt, x_pt):
        x_tf = x_pt.permute(0, 2, 3, 1).cpu().numpy()  # B x H x W x C
        weight_tf = conv_pt.weight.permute(2, 3, 1, 0).cpu().numpy()  # K x K x C_in x C_out
        with tf.Session() as sess:
            with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(
                    initializer=tf.constant_initializer(weight_tf), shape=weight_tf.shape, name="kernel"
                )
            x = tf.placeholder(tf.float32, shape=x_tf.shape, name="x")
            op = tf.nn.conv2d(
                x,
                kernel,
                strides=conv_pt.stride,
                padding="VALID" if conv_pt.padding[0] == 0 else "SAME",
                dilations=conv_pt.dilation,
            )
            sess.run(tf.global_variables_initializer())
            out = sess.run(op, feed_dict={x: x_tf})
        out = torch.from_numpy(out).permute(0, 3, 1, 2)  # B x C x H x W
        return out

    def estimate_implementation_exactness(self, cuda):
        model_pt = create_feature_extractor("inception-v3-compat", ["2048"], cuda=cuda)
        conv_pt = model_pt.Conv2d_1a_3x3.conv

        batch_size = 1
        # keep_filters = 16  # anything less makes the backends diverge and causes much different results
        # conv_pt.weight.data = conv_pt.weight[0:keep_filters]
        # conv_pt.out_channels = keep_filters

        ds = prepare_input_from_id("cifar10-train", datasets_root=tempfile.gettempdir())
        rng = np.random.RandomState(2020)
        x_pt = torch.cat([ds[i].unsqueeze(0) for i in rng.choice(len(ds), batch_size, replace=False)], dim=0)
        if cuda:
            x_pt = x_pt.cuda()
        x_pt = x_pt.float()
        x_pt = interpolate_bilinear_2d_like_tensorflow1x(x_pt, size=(299, 299), align_corners=False)
        x_pt = (x_pt - 128) / 128

        out_tf = self.forward_tf(conv_pt, x_pt)
        out_pt_builtin = self.forward_pt(conv_pt, x_pt)
        out_pt_manualchw = self.forward_pt_manualchw(conv_pt, x_pt)
        out_pt_manualhwc = self.forward_pt_manualhwc(conv_pt, x_pt)

        err_abs_tf_pt_builtin = (out_tf - out_pt_builtin).abs()
        err_abs_tf_pt_manualchw = (out_tf - out_pt_manualchw).abs()
        err_abs_tf_pt_manualhwc = (out_tf - out_pt_manualhwc).abs()
        err_abs_pt_builtin_manualchw = (out_pt_builtin - out_pt_manualchw).abs()
        err_abs_pt_builtin_manualhwc = (out_pt_builtin - out_pt_manualhwc).abs()

        suffix = f'convolution_{"gpu" if cuda else "cpu"}'

        self.save(out_tf, f"{suffix}_conv_tf.png")
        self.save(out_pt_builtin, f"{suffix}_conv_pt_builtin.png")
        self.save(err_abs_tf_pt_builtin, f"{suffix}_err_abs_tf_pt_builtin.png")
        self.save(err_abs_tf_pt_manualchw, f"{suffix}_err_abs_tf_pt_manualchw.png")
        self.save(err_abs_tf_pt_manualhwc, f"{suffix}_err_abs_tf_pt_manualhwc.png")
        self.save(err_abs_pt_builtin_manualchw, f"{suffix}_err_abs_pt_builtin_manualchw.png")
        self.save(err_abs_pt_builtin_manualhwc, f"{suffix}_err_abs_pt_builtin_manualhwc.png")

        flipping_pixel_err_abs = err_abs_tf_pt_builtin[0, 0, -1, -1].item()
        print(f"{suffix}_bottom_right_flipping_pixel_err_abs={flipping_pixel_err_abs}", file=sys.stderr)

        err_abs = err_abs_tf_pt_builtin.max().item()
        print(f"{suffix}_max_pixelwise_err_abs={err_abs}", file=sys.stderr)

        err_rel = err_abs / out_tf.abs().max().clamp_min(1e-9).item()
        print(f"{suffix}_max_pixelwise_err_rel={err_rel}", file=sys.stderr)

        return err_rel

    def test_convolution(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""

        err_rel = self.estimate_implementation_exactness(cuda)
        if cuda:
            self.assertLess(err_rel, 1e-6)
        else:
            self.assertGreaterEqual(err_rel, 0)
            self.assertLess(err_rel, 1e-7)

        if cuda:
            print("ENABLING TENSORFLOW DETERMINISM", file=sys.stderr)
            with redirect_stdout(sys.stderr):
                patch_tensorflow_for_determinism()

            err_rel = self.estimate_implementation_exactness(cuda)
            self.assertLess(err_rel, 1e-6)


if __name__ == "__main__":
    unittest.main()
