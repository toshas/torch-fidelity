# !/usr/bin/env python3

# Adaptation of the following sources:
#   https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py commit id 4738ea6
#   Distributed under BSD 3-Clause: https://github.com/mbinkowski/MMD-GAN/blob/master/LICENSE

import json
import os.path
import sys
import tarfile
import tempfile
from contextlib import redirect_stdout
from urllib import request

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from tfdeterminism import patch as patch_tensorflow_for_determinism
from tqdm import tqdm

# InceptionV3 pretrained weights from TensorFlow models library
#   Distributed under Apache License 2.0: https://github.com/tensorflow/models/blob/master/LICENSE
URL_INCEPTION_V3 = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"


def glob_images_path(path, glob_recursively, verbose=False):
    have_lossy = False
    files = []
    for r, d, ff in os.walk(path):
        if not glob_recursively and os.path.realpath(r) != os.path.realpath(path):
            continue
        for f in ff:
            ext = os.path.splitext(f)[1].lower()
            if ext not in (".png", ".jpg", ".jpeg"):
                continue
            if ext in (".jpg", ".jpeg"):
                have_lossy = True
            files.append(os.path.realpath(os.path.join(r, f)))
    files = sorted(files)
    if verbose:
        print(
            f'Found {len(files)} images in "{path}"'
            f'{". Some images are lossy-compressed - this may affect metrics!" if have_lossy else ""}',
            file=sys.stderr,
        )
    return files


class Inception(object):
    def __init__(self):
        MODEL_DIR = tempfile.gettempdir()
        self.softmax_dim = 1008
        self.coder_dim = 2048

        filename = URL_INCEPTION_V3.split("/")[-1]
        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(filepath):
            filepath, _ = request.urlretrieve(URL_INCEPTION_V3)

        tarfile.open(filepath, "r:gz").extractall(MODEL_DIR)
        with tf.gfile.FastGFile(os.path.join(MODEL_DIR, "classify_image_graph_def.pb"), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        # Works with an arbitrary minibatch size.
        self.sess = sess = tf.Session()
        pool3 = sess.graph.get_tensor_by_name("pool_3:0")
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

        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        self.coder = tf.squeeze(tf.squeeze(pool3, 2), 1)
        logits = tf.matmul(self.coder, w)
        self.softmax = tf.nn.softmax(logits)

        assert self.coder.get_shape()[1].value == self.coder_dim
        assert self.softmax.get_shape()[1].value == self.softmax_dim

        self.input = "ExpandDims:0"


class LeNet(object):
    def __init__(self):
        MODEL_DIR = "lenet/saved_model"
        self.softmax_dim = 10
        self.coder_dim = 512

        self.sess = sess = tf.Session()

        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], MODEL_DIR)
        g = sess.graph

        self.softmax = g.get_tensor_by_name("Softmax_1:0")
        self.coder = g.get_tensor_by_name("Relu_5:0")

        assert self.coder.get_shape()[1].value == self.coder_dim
        assert self.softmax.get_shape()[1].value == self.softmax_dim
        self.input = "Placeholder_2:0"


def featurize(
    images,
    model,
    batch_size=100,
    transformer=np.asarray,
    get_preds=True,
    get_codes=False,
    out_preds=None,
    out_codes=None,
    verbose=False,
):
    """images: a list of numpy arrays with values in [0, 255]"""
    sub = transformer(images[:10])
    lo, hi = np.min(sub), np.max(sub)
    assert sub.ndim == 4
    if isinstance(model, Inception):
        assert sub.shape[3] == 3
        if verbose and (hi > 255 or lo < 0):
            print(f"WARNING! Inception min/max violated: min={lo}, max={hi}", file=sys.stderr)
    elif isinstance(model, LeNet):
        batch_size = 64
        assert sub.shape[3] == 1
        if verbose and (hi > 0.5 or lo < -0.5):
            print(f"WARNING! LeNet min/max violated: min={lo}, max={hi}", file=sys.stderr)

    n = len(images)

    to_get = ()
    ret = ()
    if get_preds:
        to_get += (model.softmax,)
        if out_preds is not None:
            assert out_preds.shape == (n, model.softmax_dim)
            assert out_preds.dtype == np.float32
            preds = out_preds
        else:
            preds = np.empty((n, model.softmax_dim), dtype=np.float32)
            preds.fill(np.nan)
        ret += (preds,)
    if get_codes:
        to_get += (model.coder,)
        if out_codes is not None:
            assert out_codes.shape == (n, model.coder_dim)
            assert out_codes.dtype == np.float32
            codes = out_codes
        else:
            codes = np.empty((n, model.coder_dim), dtype=np.float32)
            codes.fill(np.nan)
        ret += (codes,)

    for start in tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        inp = transformer(images[start:end])

        if end - start != batch_size:
            pad = batch_size - (end - start)
            extra = np.zeros((pad,) + inp.shape[1:], dtype=inp.dtype)
            inp = np.r_[inp, extra]
            w = slice(0, end - start)
        else:
            w = slice(None)

        out = model.sess.run(to_get, {model.input: inp})
        if get_preds:
            preds[start:end] = out[0][w]
        if get_codes:
            codes[start:end] = out[-1][w]

    return ret


def get_splits(n, splits=10, split_method="openai"):
    if split_method == "openai":
        return [slice(i * n // splits, (i + 1) * n // splits) for i in range(splits)]
    elif split_method == "bootstrap":
        return [np.random.choice(n, n) for _ in range(splits)]
    else:
        raise ValueError("bad split_method {}".format(split_method))


def inception_score(preds, **split_args):
    split_inds = get_splits(preds.shape[0], **split_args)
    scores = np.zeros(len(split_inds))
    for i, inds in enumerate(split_inds):
        part = preds[inds]
        kl = part * (np.log(part) - np.log(np.mean(part, 0, keepdims=True)))
        kl = np.mean(np.sum(kl, 1))
        scores[i] = np.exp(kl)
    return scores


def fid_score(codes_g, codes_r, eps=1e-6, **split_args):
    splits_g = get_splits(codes_g.shape[0], **split_args)
    splits_r = get_splits(codes_r.shape[0], **split_args)
    assert len(splits_g) == len(splits_r)
    d = codes_g.shape[1]
    assert codes_r.shape[1] == d

    scores = np.zeros(len(splits_g))
    with tqdm(splits_g, desc="FID") as bar:
        for i, (w_g, w_r) in enumerate(zip(bar, splits_r)):
            part_g = codes_g[w_g]
            part_r = codes_r[w_r]

            mn_g = part_g.mean(axis=0)
            mn_r = part_r.mean(axis=0)

            cov_g = np.cov(part_g, rowvar=False)
            cov_r = np.cov(part_r, rowvar=False)

            covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
            if not np.isfinite(covmean).all():
                cov_g[range(d), range(d)] += eps
                cov_r[range(d), range(d)] += eps
                covmean = linalg.sqrtm(cov_g.dot(cov_r))

            scores[i] = np.sum((mn_g - mn_r) ** 2) + (np.trace(cov_g) + np.trace(cov_r) - 2 * np.trace(covmean))
            bar.set_postfix({"mean": scores[: i + 1].mean()})
    return scores


def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000, ret_var=True, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    rng = np.random.RandomState(2020)

    with tqdm(range(n_subsets), desc="MMD") as bar:
        for i in bar:
            g = codes_g[rng.choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[rng.choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({"mean": mmds[: i + 1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1, var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY, var_at_m=var_at_m, ret_var=ret_var)


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(
    K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est="unbiased", block_size=1024, var_at_m=None, ret_var=True
):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == "biased":
        mmd2 = (Kt_XX_sum + sum_diag_X) / (m * m) + (Kt_YY_sum + sum_diag_Y) / (m * m) - 2 * K_XY_sum / (m * m)
    else:
        assert mmd_est in {"unbiased", "u-statistic"}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == "unbiased":
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (_sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (_sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = 4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est

    return mmd2, var_est


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    parser.add_argument("reference_feats", nargs="?")
    parser.add_argument("--output", "-o")

    parser.add_argument(
        "--reference-subset", default=slice(None), type=lambda x: slice(*(int(s) if s else None for s in x.split(":")))
    )

    parser.add_argument("--batch-size", type=int, default=50)

    parser.add_argument("--model", choices=["inception", "lenet"], default="inception")

    parser.add_argument("--gpu", default="", type=str, help="GPU to use (leave blank for CPU only)")
    parser.add_argument("--json", action="store_true", help="Print scores in JSON")
    parser.add_argument(
        "--determinism",
        action="store_true",
        help="Enforce determinism in TensorFlow to remove variance when running with the same inputs. "
        "Without it inception score varies between different runs on the same data (e.g. 7.86 +/- "
        "0.05). More information: https://github.com/NVIDIA/tensorflow-determinism",
    )
    parser.add_argument("-s", "--silent", action="store_true", help="Verbose or silent progress bar and messages")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--save-codes")
    g.add_argument("--load-codes")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--save-preds")
    g.add_argument("--load-preds")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--do-inception", action="store_true", default=True)
    g.add_argument("--no-inception", action="store_false", dest="do_inception")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--do-fid", action="store_true", default=False)
    g.add_argument("--no-fid", action="store_false", dest="do_fid")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--do-mmd", action="store_true", default=False)
    g.add_argument("--no-mmd", action="store_false", dest="do_mmd")
    parser.add_argument("--mmd-degree", type=int, default=3)
    parser.add_argument("--mmd-gamma", type=float, default=None)
    parser.add_argument("--mmd-coef0", type=float, default=1)

    parser.add_argument("--mmd-subsets", type=int, default=100)
    parser.add_argument("--mmd-subset-size", type=int, default=1000)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--mmd-var", action="store_true", default=False)
    g.add_argument("--no-mmd-var", action="store_false", dest="mmd_var")

    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--split-method", choices=["openai", "bootstrap"], default="openai")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.determinism:
        with redirect_stdout(sys.stderr):
            patch_tensorflow_for_determinism()

    if (args.do_fid or args.do_mmd) and args.reference_feats is None:
        parser.error("Need REFERENCE_FEATS if you're doing FID or MMD")

    def check_path(pth):
        if os.path.exists(pth):
            parser.error("Path {} already exists".format(pth))
        d = os.path.dirname(pth)
        if d and not os.path.exists(d):
            os.makedirs(d)

    if args.output:
        check_path(args.output)

    if os.path.isdir(args.samples):
        files = glob_images_path(args.samples, glob_recursively=False, verbose=not args.silent)
        samples = [np.expand_dims(np.array(Image.open(f).convert("RGB")), 0) for f in files]
        samples = np.concatenate(samples)
    else:
        samples = np.load(args.samples, mmap_mode="r")

    if args.model == "inception":
        model = Inception()
        if samples.dtype == np.uint8:
            transformer = np.asarray
        elif samples.dtype == np.float32:
            m = samples[:10].max()
            assert 0.5 <= m <= 1
            transformer = lambda x: x * 255
        else:
            raise TypeError("don't know how to handle {}".format(samples.dtype))
    elif args.model == "lenet":
        model = LeNet()
        if samples.dtype == np.uint8:

            def transformer(x):
                return (np.asarray(x, dtype=np.float32) - (255 / 2.0)) / 255

        elif samples.dtype == np.float32:
            assert 0.8 <= samples[:10].max() <= 1
            assert 0 <= samples[:10].min() <= 0.3
            transformer = lambda x: x - 0.5
        else:
            raise TypeError("don't know how to handle {}".format(samples.dtype))
    else:
        raise ValueError("bad model {}".format(args.model))

    if args.reference_feats:
        ref_feats = np.load(args.reference_feats, mmap_mode="r")[args.reference_subset]

    out_kw = {}
    if args.save_codes:
        check_path(args.save_codes)
        out_kw["out_codes"] = np.lib.format.open_memmap(
            args.save_codes, mode="w+", dtype=np.float32, shape=(samples.shape[0], model.coder_dim)
        )
    if args.save_preds:
        check_path(args.save_preds)
        out_kw["out_preds"] = np.lib.format.open_memmap(
            args.save_preds, mode="w+", dtype=np.float32, shape=(samples.shape[0], model.softmax_dim)
        )

    need_preds = args.do_inception or args.save_preds
    need_codes = args.do_fid or args.do_mmd or args.save_codes

    if not args.silent:
        print(
            "Transformer test: transformer([-1, 0, 10.]) = " + repr(transformer(np.array([-1, 0, 10.0]))),
            file=sys.stderr,
        )

    if args.load_codes or args.load_preds:
        if args.load_codes:
            codes = np.load(args.load_codes, mmap_mode="r")
            assert codes.ndim == 2
            assert codes.shape[0] == samples.shape[0]
            assert codes.shape[1] == model.coder_dim

        if args.load_preds:
            preds = np.load(args.load_preds, mmap_mode="r")
            assert preds.ndim == 2
            assert preds.shape[0] == samples.shape[0]
            assert preds.shape[1] == model.softmax_dim
        elif need_preds:
            raise NotImplementedError()
    else:
        out = featurize(
            samples,
            model,
            batch_size=args.batch_size,
            transformer=transformer,
            get_preds=need_preds,
            get_codes=need_codes,
            verbose=not args.silent,
            **out_kw,
        )
        if need_preds:
            preds = out[0]
        if need_codes:
            codes = out[-1]

    split_args = {"splits": args.splits, "split_method": args.split_method}

    output = {"args": args}

    metrics_json = {}

    if args.do_inception:
        output["inception"] = scores = inception_score(preds, **split_args)
        if args.json:
            metrics_json["inception_score_mean"] = float(np.mean(scores))
            metrics_json["inception_score_std"] = float(np.std(scores))
        else:
            print("Inception mean:", np.mean(scores))
            print("Inception std:", np.std(scores))
            print("Inception scores:", scores, sep="\n")

    if args.do_fid:
        output["fid"] = scores = fid_score(codes, ref_feats, **split_args)
        if args.json:
            metrics_json["frechet_inception_distance"] = float(np.mean(scores))
        else:
            print("FID mean:", np.mean(scores))
            print("FID std:", np.std(scores))
            print("FID scores:", scores, sep="\n")
            print()

    if args.do_mmd:
        ret = polynomial_mmd_averages(
            codes,
            ref_feats,
            degree=args.mmd_degree,
            gamma=args.mmd_gamma,
            coef0=args.mmd_coef0,
            ret_var=args.mmd_var,
            n_subsets=args.mmd_subsets,
            subset_size=args.mmd_subset_size,
        )
        if args.mmd_var:
            output["mmd2"], output["mmd2_var"] = mmd2s, vars = ret
        else:
            output["mmd2"] = mmd2s = ret
        if args.json:
            metrics_json["kernel_inception_distance_mean"] = float(mmd2s.mean())
            metrics_json["kernel_inception_distance_std"] = float(mmd2s.std())
        else:
            print("mean MMD^2 estimate:", mmd2s.mean())
            print("std MMD^2 estimate:", mmd2s.std())
            print("MMD^2 estimates:", mmd2s, sep="\n")
            print()
            if args.mmd_var:
                print("mean Var[MMD^2] estimate:", vars.mean())
                print("std Var[MMD^2] estimate:", vars.std())
                print("Var[MMD^2] estimates:", vars, sep="\n")
                print()

    if args.output:
        np.savez(args.output, **output)

    if args.json:
        print(json.dumps(metrics_json, indent=4))


if __name__ == "__main__":
    main()
