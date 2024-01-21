import os
import sys
import tarfile
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stdout

import numpy as np
import tensorflow as tf
import torch
from tfdeterminism import patch as patch_tensorflow_for_determinism

from tests import TimeTrackingTestCase
from torch_fidelity.utils import prepare_input_from_id, create_feature_extractor

DATA_URL = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"


class TestInception(TimeTrackingTestCase):
    @staticmethod
    def get_inception_tf():
        model_dir = tempfile.gettempdir()
        filename = DATA_URL.split("/")[-1]
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath)
            statinfo = os.stat(filepath)
            print(f"Succesfully downloaded {filename} {statinfo.st_size} bytes.", file=sys.stderr)
        tarfile.open(filepath, "r:gz").extractall(model_dir)
        with tf.gfile.FastGFile(os.path.join(model_dir, "classify_image_graph_def.pb"), "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")
        with tf.Session() as sess:
            pool3 = sess.graph.get_tensor_by_name("pool_3:0")
            for op_idx, op in enumerate(pool3.graph.get_operations()):
                for op_out in op.outputs:
                    shape = op_out.get_shape()
                    if shape._dims is not None:
                        shape = [s for s in shape]
                        new_shape = []
                        for j, s in enumerate(shape):
                            if s == 1 and j == 0:
                                new_shape.append(None)
                            else:
                                new_shape.append(s)
                        op_out.__dict__["_shape_val"] = tf.TensorShape(new_shape)
            pool3 = tf.squeeze(pool3, [1, 2])
            w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
            logits_unbiased = tf.matmul(pool3, w)
        return pool3, logits_unbiased

    @staticmethod
    def forward_tf(model, x_pt_bchw):
        x_np_bhwc = x_pt_bchw.permute(0, 2, 3, 1).numpy()
        assert (
            type(x_np_bhwc) is np.ndarray
            and x_np_bhwc.dtype == np.uint8
            and len(x_np_bhwc.shape) == 4
            and x_np_bhwc.shape[3] == 3
        )
        x = x_np_bhwc
        with tf.Session() as sess:
            pool3, logits_unbiased = sess.run(model, {"ExpandDims:0": x})
        return pool3, logits_unbiased

    @staticmethod
    def forward_pt(model, x_pt_bchw, cuda):
        assert (
            torch.is_tensor(x_pt_bchw)
            and x_pt_bchw.dtype == torch.uint8
            and len(x_pt_bchw.shape) == 4
            and x_pt_bchw.shape[1] == 3
        )
        x = x_pt_bchw
        if cuda:
            x = x.cuda()
        x = model(x)
        featuresdict = model.convert_features_tuple_to_dict(x)
        return featuresdict["2048"].cpu().numpy(), featuresdict["logits_unbiased"].cpu().numpy()

    @staticmethod
    def estimate_implementation_exactness(cuda, batch_size=8, rng_seed=2020):
        model_tf = TestInception.get_inception_tf()
        model_pt = create_feature_extractor(
            "inception-v3-compat",
            ["2048", "logits_unbiased"],
            cuda=cuda,
        )
        ds = prepare_input_from_id("cifar10-train", datasets_root=tempfile.gettempdir())
        rng = np.random.RandomState(rng_seed)
        batch_pt = torch.cat([ds[i].unsqueeze(0) for i in rng.choice(len(ds), batch_size, replace=False)], dim=0)
        f1_pt, f2_pt = TestInception.forward_pt(model_pt, batch_pt, cuda)
        f1_tf, f2_tf = TestInception.forward_tf(model_tf, batch_pt)
        return {
            "2048": {"pt": [f1_pt], "tf": [f1_tf]},
            "logits_unbiased": {"pt": [f2_pt], "tf": [f2_tf]},
        }

    def test_inception_tolerances_and_determinism(self):
        cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
        suffix = f'inception_{"gpu" if cuda else "cpu"}_'

        def process_feature_pair(featvec_pt, featvec_tf):
            f_pt = np.concatenate([np.expand_dims(a, 0) for a in featvec_pt], 0)  # repeats x batch x feat_dims ..
            f_tf = np.concatenate([np.expand_dims(a, 0) for a in featvec_tf], 0)
            return {
                "max_pixelwise_difference_across_runs_pt": np.max(np.max(f_pt, axis=0) - np.min(f_pt, axis=0)),
                "max_pixelwise_difference_across_runs_tf": np.max(np.max(f_tf, axis=0) - np.min(f_tf, axis=0)),
                "max_pixelwise_err_abs_pt_tf": np.max(np.abs(f_pt - f_tf)),
                "max_pixelwise_err_rel_pt_tf": np.max(np.abs(f_pt - f_tf) / max(np.max(np.abs(f_tf)), 1e-9)),
            }

        def get_statistics(repeat=8):
            features_acc = None
            for _ in range(repeat):
                fs = self.estimate_implementation_exactness(cuda)
                if features_acc is None:
                    features_acc = fs
                else:
                    for fname in features_acc.keys():
                        f_acc_dict = features_acc[fname]
                        f_cur_dict = fs[fname]
                        f_acc_dict["pt"] += f_cur_dict["pt"]
                        f_acc_dict["tf"] += f_cur_dict["tf"]
            stats = {}
            for fname, fvec_pt_tf in features_acc.items():
                f_pt = fvec_pt_tf["pt"]
                f_tf = fvec_pt_tf["tf"]
                stats[fname] = process_feature_pair(f_pt, f_tf)
            return stats

        fstats = get_statistics()

        for fname, fstat in fstats.items():
            fsuffix = suffix + ("nondet_" if cuda else "") + fname + "_"
            for sname, sval in fstat.items():
                name = fsuffix + sname
                print(f"{name}: {sval}", file=sys.stderr)
                if sname == "max_pixelwise_difference_across_runs_pt":
                    self.assertEqual(sval, 0, name)
                elif sname == "max_pixelwise_difference_across_runs_tf":
                    if cuda:
                        self.assertGreaterEqual(sval, 0, name)
                        self.assertLessEqual(sval, 1e-4, name)
                    else:
                        self.assertEqual(sval, 0, name)
                elif sname == "max_pixelwise_err_abs_pt_tf":
                    if cuda:
                        self.assertLessEqual(sval, 1e-3, name)
                    else:
                        self.assertLessEqual(sval, 1e-4, name)
                elif sname == "max_pixelwise_err_rel_pt_tf":
                    if cuda:
                        self.assertLessEqual(sval, 1e-4, name)
                    else:
                        self.assertLessEqual(sval, 1e-5, name)
                else:
                    raise ValueError(f"Unknown statistic name {sname}")

        if cuda:
            print("ENABLING TENSORFLOW DETERMINISM", file=sys.stderr)
            with redirect_stdout(sys.stderr):
                patch_tensorflow_for_determinism()

            fstats = get_statistics()

            for fname, fstat in fstats.items():
                fsuffix = suffix + "determ_" + fname + "_"
                for sname, sval in fstat.items():
                    name = fsuffix + sname
                    print(f"{name}: {sval}", file=sys.stderr)
                    if sname in ("max_pixelwise_difference_across_runs_pt", "max_pixelwise_difference_across_runs_tf"):
                        self.assertEqual(sval, 0, name)
                    elif sname == "max_pixelwise_err_abs_pt_tf":
                        self.assertLessEqual(sval, 1e-3, name)
                    elif sname == "max_pixelwise_err_rel_pt_tf":
                        self.assertLessEqual(sval, 1e-4, name)
                    else:
                        raise ValueError(f"Unknown statistic name {sname}")


if __name__ == "__main__":
    unittest.main()
