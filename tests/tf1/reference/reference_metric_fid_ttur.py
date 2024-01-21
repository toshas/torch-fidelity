# !/usr/bin/env python3

# Adaptation of the following sources:
#   https://github.com/bioinf-jku/TTUR/blob/master/fid.py commit id d4baae8
#   Distributed under Apache License 2.0: https://github.com/bioinf-jku/TTUR/blob/master/LICENSE

import json
import os
import pathlib
import sys
import tempfile
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
from imageio import imread
from scipy import linalg
from tfdeterminism import patch as patch_tensorflow_for_determinism
from tqdm import tqdm

# InceptionV3 pretrained weights from TensorFlow models library
#   Distributed under Apache License 2.0: https://github.com/tensorflow/models/blob/master/LICENSE
URL_INCEPTION_V3 = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

KEY_FID = "frechet_inception_distance"


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    with tf.io.gfile.GFile(pth, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="FID_Inception_Net")


def get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer."""
    layername = "FID_Inception_Net/pool_3:0"
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                # shape = [s.value for s in shape] TF 1.x
                shape = [s for s in shape]  # TF 2.x
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = get_inception_layer(sess)
    n_images = images.shape[0]
    if batch_size > n_images:
        if verbose:
            print("WARNING: batch size is bigger than the data size. setting batch size to data size", sys.stderr)
        batch_size = n_images
    n_batches = n_images // batch_size
    pred_arr = np.empty((n_images, 2048))
    for i in tqdm(range(n_batches), disable=not verbose):
        start = i * batch_size
        if start + batch_size < n_images:
            end = start + batch_size
        else:
            end = n_images
        batch = images[start:end]
        pred = sess.run(inception_layer, {"FID_Inception_Net/ExpandDims:0": batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size=batch_size, verbose=verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def load_image_batch(files):
    """Convenience method for batch-loading images
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    Returns:
    -- A numpy array of dimensions (num_images,hi, wi, 3) representing the image pixel values.
    """
    return np.array([imread(str(fn)).astype(np.float32) for fn in files])


def get_activations_from_files(files, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = get_inception_layer(sess)
    n_imgs = len(files)
    if batch_size > n_imgs:
        if verbose:
            print("WARNING: batch size is bigger than the data size. setting batch size to data size", sys.stderr)
        batch_size = n_imgs
    n_batches = n_imgs // batch_size + 1
    pred_arr = np.empty((n_imgs, 2048))
    for i in tqdm(range(n_batches), disable=not verbose):
        start = i * batch_size
        if start + batch_size < n_imgs:
            end = start + batch_size
        else:
            end = n_imgs
        batch = load_image_batch(files[start:end])
        pred = sess.run(inception_layer, {"FID_Inception_Net/ExpandDims:0": batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
        del batch  # clean up memory
    return pred_arr


def calculate_activation_statistics_from_files(files, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations_from_files(files, sess, batch_size=batch_size, verbose=verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def check_or_download_inception(verbose=False):
    """Checks if the path to the inception file is valid, or downloads the file if it is not present."""
    inception_path = tempfile.gettempdir()
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / "classify_image_graph_def.pb"
    if not model_file.exists():
        if verbose:
            print("Downloading Inception model", sys.stderr)
        from urllib import request
        import tarfile

        fn, _ = request.urlretrieve(URL_INCEPTION_V3)
        with tarfile.open(fn, mode="r") as f:
            f.extract("classify_image_graph_def.pb", str(model_file.parent))
    return str(model_file)


def handle_path(path, sess, low_profile=False, verbose=False):
    if path.endswith(".npz"):
        f = np.load(path)
        m, s = f["mu"][:], f["sigma"][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
        if low_profile:
            m, s = calculate_activation_statistics_from_files(files, sess, verbose=verbose)
        else:
            x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
            m, s = calculate_activation_statistics(x, sess, verbose=verbose)
            del x
    return m, s


def calculate_fid_given_paths(paths, low_profile=False, verbose=False):
    """Calculates the FID of two paths."""
    inception_path = check_or_download_inception(verbose=verbose)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = handle_path(paths[0], sess, low_profile=low_profile, verbose=verbose)
        m2, s2 = handle_path(paths[1], sess, low_profile=low_profile, verbose=verbose)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return {KEY_FID: float(fid_value)}


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2, help="Path to the generated images or to .npz statistic files")
    parser.add_argument("--gpu", default="", type=str, help="GPU to use (leave blank for CPU only)")
    parser.add_argument("--json", action="store_true", help="Print scores in JSON")
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="Enforce determinism in TensorFlow to remove variance when running with the same inputs. "
        "Without it inception score varies between different runs on the same data (e.g. 18.11 +/- "
        "0.02). More information: https://github.com/NVIDIA/tensorflow-determinism",
    )
    parser.add_argument(
        "--lowprofile",
        action="store_true",
        help="Keep only one batch of images in memory at a time. "
        "This reduces memory footprint, but may decrease speed slightly.",
    )
    parser.add_argument("-s", "--silent", action="store_true", help="Verbose or silent progress bar and messages")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.determinism:
        with redirect_stdout(sys.stderr):
            patch_tensorflow_for_determinism()

    metrics = calculate_fid_given_paths(args.path, low_profile=args.lowprofile, verbose=not args.silent)

    if args.json:
        print(json.dumps(metrics, indent=4))
    else:
        print(", ".join((f"{k}: {v}" for k, v in metrics.items())))
