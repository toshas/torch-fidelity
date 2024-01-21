import os
import pathlib
import tarfile
import tempfile
from urllib import request

import tensorflow as tf
import torch
from tensorflow.python.framework import tensor_util

TF_INCEPTION_URL = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"


def check_or_download_inception_graphdef():
    tempfile.gettempdir()
    model_file = pathlib.Path(os.path.join(tempfile.gettempdir(), "classify_image_graph_def.pb"))
    if not model_file.exists():
        print("Downloading Inception model")
        fn, _ = request.urlretrieve(TF_INCEPTION_URL)
        with tarfile.open(fn, mode="r") as f:
            f.extract("classify_image_graph_def.pb", str(model_file.parent))
    return str(model_file)


def load_weights_from_graphdef():
    path = check_or_download_inception_graphdef()
    with tf.io.gfile.GFile(path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        weights = {w.name: tensor_util.MakeNdarray(w.attr["value"].tensor) for w in graph_def.node if w.op == "Const"}
    return weights


def convert_tensorflow_graphdef_to_pytorch_checkpoint(path_out=None):
    map_tf_to_pt = {
        # obtained through semi-automatic matching by the shapes of tensors
        "conv/batchnorm": "Conv2d_1a_3x3.bn",
        "conv/conv2d_params": "Conv2d_1a_3x3.conv",
        "conv_1/batchnorm": "Conv2d_2a_3x3.bn",
        "conv_1/conv2d_params": "Conv2d_2a_3x3.conv",
        "conv_2/batchnorm": "Conv2d_2b_3x3.bn",
        "conv_2/conv2d_params": "Conv2d_2b_3x3.conv",
        "conv_3/batchnorm": "Conv2d_3b_1x1.bn",
        "conv_3/conv2d_params": "Conv2d_3b_1x1.conv",
        "conv_4/batchnorm": "Conv2d_4a_3x3.bn",
        "conv_4/conv2d_params": "Conv2d_4a_3x3.conv",
        "mixed/conv/batchnorm": "Mixed_5b.branch1x1.bn",
        "mixed/conv/conv2d_params": "Mixed_5b.branch1x1.conv",
        "mixed/tower_1/conv/batchnorm": "Mixed_5b.branch3x3dbl_1.bn",
        "mixed/tower_1/conv/conv2d_params": "Mixed_5b.branch3x3dbl_1.conv",
        "mixed/tower_1/conv_1/batchnorm": "Mixed_5b.branch3x3dbl_2.bn",
        "mixed/tower_1/conv_1/conv2d_params": "Mixed_5b.branch3x3dbl_2.conv",
        "mixed/tower_1/conv_2/batchnorm": "Mixed_5b.branch3x3dbl_3.bn",
        "mixed/tower_1/conv_2/conv2d_params": "Mixed_5b.branch3x3dbl_3.conv",
        "mixed/tower/conv/batchnorm": "Mixed_5b.branch5x5_1.bn",
        "mixed/tower/conv/conv2d_params": "Mixed_5b.branch5x5_1.conv",
        "mixed/tower/conv_1/batchnorm": "Mixed_5b.branch5x5_2.bn",
        "mixed/tower/conv_1/conv2d_params": "Mixed_5b.branch5x5_2.conv",
        "mixed/tower_2/conv/batchnorm": "Mixed_5b.branch_pool.bn",
        "mixed/tower_2/conv/conv2d_params": "Mixed_5b.branch_pool.conv",
        "mixed_1/conv/batchnorm": "Mixed_5c.branch1x1.bn",
        "mixed_1/conv/conv2d_params": "Mixed_5c.branch1x1.conv",
        "mixed_1/tower_1/conv/batchnorm": "Mixed_5c.branch3x3dbl_1.bn",
        "mixed_1/tower_1/conv/conv2d_params": "Mixed_5c.branch3x3dbl_1.conv",
        "mixed_1/tower_1/conv_1/batchnorm": "Mixed_5c.branch3x3dbl_2.bn",
        "mixed_1/tower_1/conv_1/conv2d_params": "Mixed_5c.branch3x3dbl_2.conv",
        "mixed_1/tower_1/conv_2/batchnorm": "Mixed_5c.branch3x3dbl_3.bn",
        "mixed_1/tower_1/conv_2/conv2d_params": "Mixed_5c.branch3x3dbl_3.conv",
        "mixed_1/tower/conv/batchnorm": "Mixed_5c.branch5x5_1.bn",
        "mixed_1/tower/conv/conv2d_params": "Mixed_5c.branch5x5_1.conv",
        "mixed_1/tower/conv_1/batchnorm": "Mixed_5c.branch5x5_2.bn",
        "mixed_1/tower/conv_1/conv2d_params": "Mixed_5c.branch5x5_2.conv",
        "mixed_1/tower_2/conv/batchnorm": "Mixed_5c.branch_pool.bn",
        "mixed_1/tower_2/conv/conv2d_params": "Mixed_5c.branch_pool.conv",
        "mixed_2/conv/batchnorm": "Mixed_5d.branch1x1.bn",
        "mixed_2/conv/conv2d_params": "Mixed_5d.branch1x1.conv",
        "mixed_2/tower_1/conv/batchnorm": "Mixed_5d.branch3x3dbl_1.bn",
        "mixed_2/tower_1/conv/conv2d_params": "Mixed_5d.branch3x3dbl_1.conv",
        "mixed_2/tower_1/conv_1/batchnorm": "Mixed_5d.branch3x3dbl_2.bn",
        "mixed_2/tower_1/conv_1/conv2d_params": "Mixed_5d.branch3x3dbl_2.conv",
        "mixed_2/tower_1/conv_2/batchnorm": "Mixed_5d.branch3x3dbl_3.bn",
        "mixed_2/tower_1/conv_2/conv2d_params": "Mixed_5d.branch3x3dbl_3.conv",
        "mixed_2/tower/conv/batchnorm": "Mixed_5d.branch5x5_1.bn",
        "mixed_2/tower/conv/conv2d_params": "Mixed_5d.branch5x5_1.conv",
        "mixed_2/tower/conv_1/batchnorm": "Mixed_5d.branch5x5_2.bn",
        "mixed_2/tower/conv_1/conv2d_params": "Mixed_5d.branch5x5_2.conv",
        "mixed_2/tower_2/conv/batchnorm": "Mixed_5d.branch_pool.bn",
        "mixed_2/tower_2/conv/conv2d_params": "Mixed_5d.branch_pool.conv",
        "mixed_3/conv/batchnorm": "Mixed_6a.branch3x3.bn",
        "mixed_3/conv/conv2d_params": "Mixed_6a.branch3x3.conv",
        "mixed_3/tower/conv/batchnorm": "Mixed_6a.branch3x3dbl_1.bn",
        "mixed_3/tower/conv/conv2d_params": "Mixed_6a.branch3x3dbl_1.conv",
        "mixed_3/tower/conv_1/batchnorm": "Mixed_6a.branch3x3dbl_2.bn",
        "mixed_3/tower/conv_1/conv2d_params": "Mixed_6a.branch3x3dbl_2.conv",
        "mixed_3/tower/conv_2/batchnorm": "Mixed_6a.branch3x3dbl_3.bn",
        "mixed_3/tower/conv_2/conv2d_params": "Mixed_6a.branch3x3dbl_3.conv",
        "mixed_4/conv/batchnorm": "Mixed_6b.branch1x1.bn",
        "mixed_4/conv/conv2d_params": "Mixed_6b.branch1x1.conv",
        "mixed_4/tower/conv/batchnorm": "Mixed_6b.branch7x7_1.bn",
        "mixed_4/tower/conv/conv2d_params": "Mixed_6b.branch7x7_1.conv",
        "mixed_4/tower/conv_1/batchnorm": "Mixed_6b.branch7x7_2.bn",
        "mixed_4/tower/conv_1/conv2d_params": "Mixed_6b.branch7x7_2.conv",
        "mixed_4/tower/conv_2/batchnorm": "Mixed_6b.branch7x7_3.bn",
        "mixed_4/tower/conv_2/conv2d_params": "Mixed_6b.branch7x7_3.conv",
        "mixed_4/tower_1/conv/batchnorm": "Mixed_6b.branch7x7dbl_1.bn",
        "mixed_4/tower_1/conv/conv2d_params": "Mixed_6b.branch7x7dbl_1.conv",
        "mixed_4/tower_1/conv_1/batchnorm": "Mixed_6b.branch7x7dbl_2.bn",
        "mixed_4/tower_1/conv_1/conv2d_params": "Mixed_6b.branch7x7dbl_2.conv",
        "mixed_4/tower_1/conv_2/batchnorm": "Mixed_6b.branch7x7dbl_3.bn",
        "mixed_4/tower_1/conv_2/conv2d_params": "Mixed_6b.branch7x7dbl_3.conv",
        "mixed_4/tower_1/conv_3/batchnorm": "Mixed_6b.branch7x7dbl_4.bn",
        "mixed_4/tower_1/conv_3/conv2d_params": "Mixed_6b.branch7x7dbl_4.conv",
        "mixed_4/tower_1/conv_4/batchnorm": "Mixed_6b.branch7x7dbl_5.bn",
        "mixed_4/tower_1/conv_4/conv2d_params": "Mixed_6b.branch7x7dbl_5.conv",
        "mixed_4/tower_2/conv/batchnorm": "Mixed_6b.branch_pool.bn",
        "mixed_4/tower_2/conv/conv2d_params": "Mixed_6b.branch_pool.conv",
        "mixed_5/conv/batchnorm": "Mixed_6c.branch1x1.bn",
        "mixed_5/conv/conv2d_params": "Mixed_6c.branch1x1.conv",
        "mixed_5/tower/conv/batchnorm": "Mixed_6c.branch7x7_1.bn",
        "mixed_5/tower/conv/conv2d_params": "Mixed_6c.branch7x7_1.conv",
        "mixed_5/tower/conv_1/batchnorm": "Mixed_6c.branch7x7_2.bn",
        "mixed_5/tower/conv_1/conv2d_params": "Mixed_6c.branch7x7_2.conv",
        "mixed_5/tower/conv_2/batchnorm": "Mixed_6c.branch7x7_3.bn",
        "mixed_5/tower/conv_2/conv2d_params": "Mixed_6c.branch7x7_3.conv",
        "mixed_5/tower_1/conv/batchnorm": "Mixed_6c.branch7x7dbl_1.bn",
        "mixed_5/tower_1/conv/conv2d_params": "Mixed_6c.branch7x7dbl_1.conv",
        "mixed_5/tower_1/conv_1/batchnorm": "Mixed_6c.branch7x7dbl_2.bn",
        "mixed_5/tower_1/conv_1/conv2d_params": "Mixed_6c.branch7x7dbl_2.conv",
        "mixed_5/tower_1/conv_2/batchnorm": "Mixed_6c.branch7x7dbl_3.bn",
        "mixed_5/tower_1/conv_2/conv2d_params": "Mixed_6c.branch7x7dbl_3.conv",
        "mixed_5/tower_1/conv_3/batchnorm": "Mixed_6c.branch7x7dbl_4.bn",
        "mixed_5/tower_1/conv_3/conv2d_params": "Mixed_6c.branch7x7dbl_4.conv",
        "mixed_5/tower_1/conv_4/batchnorm": "Mixed_6c.branch7x7dbl_5.bn",
        "mixed_5/tower_1/conv_4/conv2d_params": "Mixed_6c.branch7x7dbl_5.conv",
        "mixed_5/tower_2/conv/batchnorm": "Mixed_6c.branch_pool.bn",
        "mixed_5/tower_2/conv/conv2d_params": "Mixed_6c.branch_pool.conv",
        "mixed_6/conv/batchnorm": "Mixed_6d.branch1x1.bn",
        "mixed_6/conv/conv2d_params": "Mixed_6d.branch1x1.conv",
        "mixed_6/tower/conv/batchnorm": "Mixed_6d.branch7x7_1.bn",
        "mixed_6/tower/conv/conv2d_params": "Mixed_6d.branch7x7_1.conv",
        "mixed_6/tower/conv_1/batchnorm": "Mixed_6d.branch7x7_2.bn",
        "mixed_6/tower/conv_1/conv2d_params": "Mixed_6d.branch7x7_2.conv",
        "mixed_6/tower/conv_2/batchnorm": "Mixed_6d.branch7x7_3.bn",
        "mixed_6/tower/conv_2/conv2d_params": "Mixed_6d.branch7x7_3.conv",
        "mixed_6/tower_1/conv/batchnorm": "Mixed_6d.branch7x7dbl_1.bn",
        "mixed_6/tower_1/conv/conv2d_params": "Mixed_6d.branch7x7dbl_1.conv",
        "mixed_6/tower_1/conv_1/batchnorm": "Mixed_6d.branch7x7dbl_2.bn",
        "mixed_6/tower_1/conv_1/conv2d_params": "Mixed_6d.branch7x7dbl_2.conv",
        "mixed_6/tower_1/conv_2/batchnorm": "Mixed_6d.branch7x7dbl_3.bn",
        "mixed_6/tower_1/conv_2/conv2d_params": "Mixed_6d.branch7x7dbl_3.conv",
        "mixed_6/tower_1/conv_3/batchnorm": "Mixed_6d.branch7x7dbl_4.bn",
        "mixed_6/tower_1/conv_3/conv2d_params": "Mixed_6d.branch7x7dbl_4.conv",
        "mixed_6/tower_1/conv_4/batchnorm": "Mixed_6d.branch7x7dbl_5.bn",
        "mixed_6/tower_1/conv_4/conv2d_params": "Mixed_6d.branch7x7dbl_5.conv",
        "mixed_6/tower_2/conv/batchnorm": "Mixed_6d.branch_pool.bn",
        "mixed_6/tower_2/conv/conv2d_params": "Mixed_6d.branch_pool.conv",
        "mixed_7/conv/batchnorm": "Mixed_6e.branch1x1.bn",
        "mixed_7/conv/conv2d_params": "Mixed_6e.branch1x1.conv",
        "mixed_7/tower/conv/batchnorm": "Mixed_6e.branch7x7_1.bn",
        "mixed_7/tower/conv/conv2d_params": "Mixed_6e.branch7x7_1.conv",
        "mixed_7/tower/conv_1/batchnorm": "Mixed_6e.branch7x7_2.bn",
        "mixed_7/tower/conv_1/conv2d_params": "Mixed_6e.branch7x7_2.conv",
        "mixed_7/tower/conv_2/batchnorm": "Mixed_6e.branch7x7_3.bn",
        "mixed_7/tower/conv_2/conv2d_params": "Mixed_6e.branch7x7_3.conv",
        "mixed_7/tower_1/conv/batchnorm": "Mixed_6e.branch7x7dbl_1.bn",
        "mixed_7/tower_1/conv/conv2d_params": "Mixed_6e.branch7x7dbl_1.conv",
        "mixed_7/tower_1/conv_1/batchnorm": "Mixed_6e.branch7x7dbl_2.bn",
        "mixed_7/tower_1/conv_1/conv2d_params": "Mixed_6e.branch7x7dbl_2.conv",
        "mixed_7/tower_1/conv_2/batchnorm": "Mixed_6e.branch7x7dbl_3.bn",
        "mixed_7/tower_1/conv_2/conv2d_params": "Mixed_6e.branch7x7dbl_3.conv",
        "mixed_7/tower_1/conv_3/batchnorm": "Mixed_6e.branch7x7dbl_4.bn",
        "mixed_7/tower_1/conv_3/conv2d_params": "Mixed_6e.branch7x7dbl_4.conv",
        "mixed_7/tower_1/conv_4/batchnorm": "Mixed_6e.branch7x7dbl_5.bn",
        "mixed_7/tower_1/conv_4/conv2d_params": "Mixed_6e.branch7x7dbl_5.conv",
        "mixed_7/tower_2/conv/batchnorm": "Mixed_6e.branch_pool.bn",
        "mixed_7/tower_2/conv/conv2d_params": "Mixed_6e.branch_pool.conv",
        "mixed_8/tower/conv/batchnorm": "Mixed_7a.branch3x3_1.bn",
        "mixed_8/tower/conv/conv2d_params": "Mixed_7a.branch3x3_1.conv",
        "mixed_8/tower/conv_1/batchnorm": "Mixed_7a.branch3x3_2.bn",
        "mixed_8/tower/conv_1/conv2d_params": "Mixed_7a.branch3x3_2.conv",
        "mixed_8/tower_1/conv/batchnorm": "Mixed_7a.branch7x7x3_1.bn",
        "mixed_8/tower_1/conv/conv2d_params": "Mixed_7a.branch7x7x3_1.conv",
        "mixed_8/tower_1/conv_1/batchnorm": "Mixed_7a.branch7x7x3_2.bn",
        "mixed_8/tower_1/conv_1/conv2d_params": "Mixed_7a.branch7x7x3_2.conv",
        "mixed_8/tower_1/conv_2/batchnorm": "Mixed_7a.branch7x7x3_3.bn",
        "mixed_8/tower_1/conv_2/conv2d_params": "Mixed_7a.branch7x7x3_3.conv",
        "mixed_8/tower_1/conv_3/batchnorm": "Mixed_7a.branch7x7x3_4.bn",
        "mixed_8/tower_1/conv_3/conv2d_params": "Mixed_7a.branch7x7x3_4.conv",
        "mixed_9/conv/batchnorm": "Mixed_7b.branch1x1.bn",
        "mixed_9/conv/conv2d_params": "Mixed_7b.branch1x1.conv",
        "mixed_9/tower/conv/batchnorm": "Mixed_7b.branch3x3_1.bn",
        "mixed_9/tower/conv/conv2d_params": "Mixed_7b.branch3x3_1.conv",
        "mixed_9/tower/mixed/conv/batchnorm": "Mixed_7b.branch3x3_2a.bn",
        "mixed_9/tower/mixed/conv/conv2d_params": "Mixed_7b.branch3x3_2a.conv",
        "mixed_9/tower/mixed/conv_1/batchnorm": "Mixed_7b.branch3x3_2b.bn",
        "mixed_9/tower/mixed/conv_1/conv2d_params": "Mixed_7b.branch3x3_2b.conv",
        "mixed_9/tower_1/conv/batchnorm": "Mixed_7b.branch3x3dbl_1.bn",
        "mixed_9/tower_1/conv/conv2d_params": "Mixed_7b.branch3x3dbl_1.conv",
        "mixed_9/tower_1/conv_1/batchnorm": "Mixed_7b.branch3x3dbl_2.bn",
        "mixed_9/tower_1/conv_1/conv2d_params": "Mixed_7b.branch3x3dbl_2.conv",
        "mixed_9/tower_1/mixed/conv/batchnorm": "Mixed_7b.branch3x3dbl_3a.bn",
        "mixed_9/tower_1/mixed/conv/conv2d_params": "Mixed_7b.branch3x3dbl_3a.conv",
        "mixed_9/tower_1/mixed/conv_1/batchnorm": "Mixed_7b.branch3x3dbl_3b.bn",
        "mixed_9/tower_1/mixed/conv_1/conv2d_params": "Mixed_7b.branch3x3dbl_3b.conv",
        "mixed_9/tower_2/conv/batchnorm": "Mixed_7b.branch_pool.bn",
        "mixed_9/tower_2/conv/conv2d_params": "Mixed_7b.branch_pool.conv",
        "mixed_10/conv/batchnorm": "Mixed_7c.branch1x1.bn",
        "mixed_10/conv/conv2d_params": "Mixed_7c.branch1x1.conv",
        "mixed_10/tower/conv/batchnorm": "Mixed_7c.branch3x3_1.bn",
        "mixed_10/tower/conv/conv2d_params": "Mixed_7c.branch3x3_1.conv",
        "mixed_10/tower/mixed/conv/batchnorm": "Mixed_7c.branch3x3_2a.bn",
        "mixed_10/tower/mixed/conv/conv2d_params": "Mixed_7c.branch3x3_2a.conv",
        "mixed_10/tower/mixed/conv_1/batchnorm": "Mixed_7c.branch3x3_2b.bn",
        "mixed_10/tower/mixed/conv_1/conv2d_params": "Mixed_7c.branch3x3_2b.conv",
        "mixed_10/tower_1/conv/batchnorm": "Mixed_7c.branch3x3dbl_1.bn",
        "mixed_10/tower_1/conv/conv2d_params": "Mixed_7c.branch3x3dbl_1.conv",
        "mixed_10/tower_1/conv_1/batchnorm": "Mixed_7c.branch3x3dbl_2.bn",
        "mixed_10/tower_1/conv_1/conv2d_params": "Mixed_7c.branch3x3dbl_2.conv",
        "mixed_10/tower_1/mixed/conv/batchnorm": "Mixed_7c.branch3x3dbl_3a.bn",
        "mixed_10/tower_1/mixed/conv/conv2d_params": "Mixed_7c.branch3x3dbl_3a.conv",
        "mixed_10/tower_1/mixed/conv_1/batchnorm": "Mixed_7c.branch3x3dbl_3b.bn",
        "mixed_10/tower_1/mixed/conv_1/conv2d_params": "Mixed_7c.branch3x3dbl_3b.conv",
        "mixed_10/tower_2/conv/batchnorm": "Mixed_7c.branch_pool.bn",
        "mixed_10/tower_2/conv/conv2d_params": "Mixed_7c.branch_pool.conv",
        "softmax/weights": "fc.weight",
        "softmax/biases": "fc.bias",
    }

    weights_tf = load_weights_from_graphdef()

    weights_pt = {}
    for k_tf, k_pt in map_tf_to_pt.items():
        if k_tf.endswith("conv2d_params"):
            weight_tf = torch.from_numpy(weights_tf.pop(k_tf))
            weight_pt = weight_tf.permute(3, 2, 0, 1).contiguous()
            weights_pt[k_pt + ".weight"] = weight_pt
        elif k_tf.endswith("batchnorm"):
            weights_pt[k_pt + ".weight"] = torch.from_numpy(weights_tf.pop(k_tf + "/gamma"))
            weights_pt[k_pt + ".bias"] = torch.from_numpy(weights_tf.pop(k_tf + "/beta"))
            weights_pt[k_pt + ".running_mean"] = torch.from_numpy(weights_tf.pop(k_tf + "/moving_mean"))
            weights_pt[k_pt + ".running_var"] = torch.from_numpy(weights_tf.pop(k_tf + "/moving_variance"))
        elif k_tf == "softmax/weights":
            weight_tf = torch.from_numpy(weights_tf.pop(k_tf))
            weight_pt = weight_tf.permute(1, 0).contiguous()
            weights_pt[k_pt] = weight_pt
        elif k_tf == "softmax/biases":
            weights_pt[k_pt] = torch.from_numpy(weights_tf.pop(k_tf))
        else:
            raise NotImplementedError(f"Cannot handle TensorFlow GraphDef Const item {k_tf}")

    if path_out is None:
        path_out = "pt-inception-2015-12-05.pth"

    torch.save(weights_pt, path_out)


if __name__ == "__main__":
    print("Converting TensorFlow InceptionV3 pretrained weights to a PyTorch checkpoint...")
    convert_tensorflow_graphdef_to_pytorch_checkpoint()
    print("Done")
