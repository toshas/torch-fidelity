import os
import sys
import unittest

import numpy as np
import tensorflow as tf
import torch
from PIL import Image

from tests import TimeTrackingTestCase
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x


class TestInterpolation(TimeTrackingTestCase):
    @staticmethod
    def checkerboard(side):
        side_e = side
        if side % 2 == 0:
            side_e += 1
        n = ((side_e**2 + 1) // 2) * 2
        out = np.zeros((n, 2), dtype=np.uint8)
        out[:, 1] = 255
        out = np.ravel(out)[: side_e**2]
        out = np.reshape(out, (side_e, side_e))
        out = out[:side, :side]
        return out

    @staticmethod
    def save(img, name):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        Image.fromarray(img).save(name)

    @staticmethod
    def resize_pt(img, size, cuda, cast_back_uint=True, method=None):
        assert type(img) is np.ndarray and img.dtype == np.uint8
        img_in = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        if cuda:
            img_in = img_in.cuda()
        img_out = interpolate_bilinear_2d_like_tensorflow1x(img_in, size, align_corners=False, method=method)
        img_out = img_out.squeeze().cpu().numpy()
        if cast_back_uint:
            img_out = img_out.astype(np.uint8)
        return img_out

    @staticmethod
    def resize_tf(img, size, cast_back_uint=True):
        assert type(img) is np.ndarray and img.dtype == np.uint8
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, -1)
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [1, img.shape[1], img.shape[2], 1])
            y = sess.run(tf.image.resize_bilinear(x, size=size, align_corners=False), feed_dict={x: img})
        y = np.squeeze(y)
        if cast_back_uint:
            y = y.astype(np.uint8)
        return y

    def _max_resize_residual(self, cuda, method, visualize=None, suffix=""):
        img_032 = self.checkerboard(32)
        img_299_pt = self.resize_pt(img_032, (299, 299), cuda, cast_back_uint=False, method=method)
        img_299_tf = self.resize_tf(img_032, (299, 299), cast_back_uint=False)
        residual = np.abs(img_299_pt - img_299_tf)
        if visualize:
            residual_img = 255 * residual / max(np.max(residual), 1e-6)
            self.save(img_032, suffix + "032.png")
            self.save(img_299_pt, suffix + "299_pt.png")
            self.save(img_299_tf, suffix + "299_tf.png")
            self.save(residual_img, suffix + "residual.png")
        return np.max(residual)

    def test_resize_eps(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        suffix = f'interpolate_eps_{"gpu" if cuda else "cpu"}_'
        L_inf = self._max_resize_residual(cuda, "fast", visualize=True, suffix=suffix)
        print(f"{suffix}err_abs={L_inf}", file=sys.stderr)
        err_rel = L_inf / 255
        print(f"{suffix}err_rel={err_rel}", file=sys.stderr)
        self.assertLessEqual(err_rel, 1e-5)

    def test_resize_bit(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        suffix = f'interpolate_bit_{"gpu" if cuda else "cpu"}_'
        L_inf = self._max_resize_residual(cuda, "slow", visualize=True, suffix=suffix)
        print(f"{suffix}err_abs={L_inf}", file=sys.stderr)
        err_rel = L_inf / 255
        print(f"{suffix}err_rel={err_rel}", file=sys.stderr)
        if cuda:
            self.assertLessEqual(err_rel, 1e-5)
        else:
            self.assertLessEqual(err_rel, 0)


if __name__ == "__main__":
    unittest.main()
