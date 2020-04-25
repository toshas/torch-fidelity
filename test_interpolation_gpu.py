import os
import unittest

import numpy as np
import tensorflow as tf
import torch
from PIL import Image

from interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x


class TestInterpolation(unittest.TestCase):
    @staticmethod
    def checkerboard(side):
        side_e = side
        if side % 2 == 0:
            side_e += 1
        n = ((side_e ** 2 + 1) // 2) * 2
        out = np.zeros((n, 2), dtype=np.uint8)
        out[:, 1] = 255
        out = np.ravel(out)[:side_e**2]
        out = np.reshape(out, (side_e, side_e))
        out = out[:side, :side]
        return out

    @staticmethod
    def save(img, name):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        Image.fromarray(img).save(name)

    @staticmethod
    def resize_pt(img, size, cast_back_uint=True, how_exact=None):
        assert type(img) is np.ndarray and img.dtype == np.uint8
        img_in = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        img_out = interpolate_bilinear_2d_like_tensorflow1x(img_in, size, align_corners=False, how_exact=how_exact)
        img_out = img_out.squeeze().numpy()
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
            y = sess.run(
                tf.image.resize_bilinear(x, size=size, align_corners=False),
                feed_dict={x: img}
            )
        y = np.squeeze(y)
        if cast_back_uint:
            y = y.astype(np.uint8)
        return y

    def _test_resize(self, how_exact, threshold, visualize=None):
        img_032 = self.checkerboard(32)
        img_299_pt = self.resize_pt(img_032, (299, 299), cast_back_uint=False, how_exact=how_exact)
        img_299_tf = self.resize_tf(img_032, (299, 299), cast_back_uint=False)
        residual = np.abs(img_299_pt - img_299_tf)
        if visualize:
            print(np.histogram(residual))
            residual_img = 255 * residual / max(np.max(residual), 1e-6)
            self.save(img_032, 'img_032.png')
            self.save(img_299_pt, 'img_299_pt.png')
            self.save(img_299_tf, 'img_299_tf.png')
            self.save(residual_img, 'img_residual.png')
        L_inf = np.max(residual)
        self.assertLessEqual(L_inf, threshold)

    def test_resize_eps(self):
        self._test_resize('eps', 0.001, visualize=False)

    def test_resize_bit(self):
        self._test_resize('bit', 0, visualize=False)


if __name__ == '__main__':
    assert os.environ['CUDA_VISIBLE_DEVICES'] != ''
    unittest.main()
