# !/usr/bin/env python3

# Adaptation of the following sources:
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py commit id 0b7ed92
#   https://github.com/bioinf-jku/TTUR/blob/master/fid.py                       commit id d4baae8
#   https://github.com/bioinf-jku/TTUR/blob/master/FIDvsINC/fidutils.py         commit id a5c0140
#   Distributed under Apache License 2.0: https://github.com/bioinf-jku/TTUR/blob/master/LICENSE

import argparse
import json
import math
import os.path
import sys
import tarfile
import tempfile
from contextlib import redirect_stdout
from urllib import request

import numpy as np
import tensorflow as tf
from PIL import Image
from tfdeterminism import patch as patch_tensorflow_for_determinism
from tqdm import tqdm

# InceptionV3 pretrained weights from TensorFlow models library
#   Distributed under Apache License 2.0: https://github.com/tensorflow/models/blob/master/LICENSE
URL_INCEPTION_V3 = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

KEY_IS_MEAN = "inception_score_mean"
KEY_IS_STD = "inception_score_std"


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


def get_inception_score(model, images, splits=10, verbose=False):
    assert type(images) == list
    assert type(images[0]) == np.ndarray
    assert len(images[0].shape) == 3
    assert np.max(images[0]) > 10
    assert np.min(images[0]) >= 0.0
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 50
    with tf.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in tqdm(range(n_batches), disable=not verbose):
            inp = inps[(i * bs) : min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(model, {"ExpandDims:0": inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits) : ((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    return {
        KEY_IS_MEAN: float(np.mean(scores)),
        KEY_IS_STD: float(np.std(scores)),
    }


def init_inception():
    model_dir = tempfile.gettempdir()
    filename = URL_INCEPTION_V3.split("/")[-1]
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stderr.write(
                "\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) * 100.0)
            )
            sys.stderr.flush()

        filepath, _ = request.urlretrieve(URL_INCEPTION_V3, filepath, _progress)
        statinfo = os.stat(filepath)
        print(f"Succesfully downloaded {filename} {statinfo.st_size} bytes.", file=sys.stderr)
    tarfile.open(filepath, "r:gz").extractall(model_dir)
    with tf.gfile.FastGFile(os.path.join(model_dir, "classify_image_graph_def.pb"), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")
    # Works with an arbitrary minibatch size.
    with tf.Session() as sess:
        layername = "pool_3:0"
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
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        model = tf.nn.softmax(logits)
    return model


def get_inception_score_of_path(path, verbose=False):
    files = glob_images_path(path, False, verbose=verbose)
    images = [np.array(Image.open(f).convert("RGB")) for f in files]
    model = init_inception()
    return get_inception_score(model, images, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=1, help="Path to the generated images")
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
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.determinism:
        with redirect_stdout(sys.stderr):
            patch_tensorflow_for_determinism()

    metrics = get_inception_score_of_path(args.path[0])

    if args.json:
        print(json.dumps(metrics, indent=4))
    else:
        print(", ".join((f"{k}: {v}" for k, v in metrics.items())))
