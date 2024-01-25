import numpy as np
import torch
from tqdm import tqdm

from torch_fidelity.generative_model_base import GenerativeModelBase
from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.utils import (
    sample_random,
    batch_interp,
    create_sample_similarity,
    prepare_input_descriptor_from_input_id,
    prepare_input_from_descriptor,
)

KEY_METRIC_PPL_RAW = "perceptual_path_length_raw"
KEY_METRIC_PPL_MEAN = "perceptual_path_length_mean"
KEY_METRIC_PPL_STD = "perceptual_path_length_std"


def calculate_ppl(input_id, **kwargs):
    """
    Inspired by https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
    """
    kwargs["ppl"] = True
    batch_size = get_kwarg("batch_size", kwargs)
    cuda = get_kwarg("cuda", kwargs)
    verbose = get_kwarg("verbose", kwargs)
    epsilon = get_kwarg("ppl_epsilon", kwargs)
    interp = get_kwarg("ppl_z_interp_mode", kwargs)
    reduction = get_kwarg("ppl_reduction", kwargs)
    similarity_name = get_kwarg("ppl_sample_similarity", kwargs)
    sample_similarity_resize = get_kwarg("ppl_sample_similarity_resize", kwargs)
    sample_similarity_dtype = get_kwarg("ppl_sample_similarity_dtype", kwargs)
    discard_percentile_lower = get_kwarg("ppl_discard_percentile_lower", kwargs)
    discard_percentile_higher = get_kwarg("ppl_discard_percentile_higher", kwargs)

    input_desc = prepare_input_descriptor_from_input_id(input_id, **kwargs)
    model = prepare_input_from_descriptor(input_desc, **kwargs)
    vassert(
        isinstance(model, GenerativeModelBase),
        "Input needs to be an instance of GenerativeModelBase, which can be either passed programmatically by wrapping "
        "a model with GenerativeModelModuleWrapper, or via command line by specifying a path to a ONNX or PTH (JIT) "
        "model and a set of input1_model_* arguments",
    )

    if cuda:
        model.cuda()

    input_model_num_samples = input_desc["input_model_num_samples"]
    input_model_num_classes = model.num_classes
    input_model_z_size = model.z_size
    input_model_z_type = model.z_type

    vassert(input_model_num_classes >= 0, "Model can be unconditional (0 classes) or conditional (positive)")
    vassert(
        type(input_model_z_size) is int and input_model_z_size > 0,
        'Dimensionality of generator noise not specified ("input1_model_z_size" argument)',
    )
    vassert(type(epsilon) is float and epsilon > 0, "Epsilon must be a small positive floating point number")
    vassert(type(input_model_num_samples) is int and input_model_num_samples > 0, "Number of samples must be positive")
    vassert(reduction in ("none", "mean"), "Reduction must be one of [none, mean]")
    vassert(discard_percentile_lower is None or 0 < discard_percentile_lower < 100, "Invalid percentile")
    vassert(discard_percentile_higher is None or 0 < discard_percentile_higher < 100, "Invalid percentile")
    if discard_percentile_lower is not None and discard_percentile_higher is not None:
        vassert(0 < discard_percentile_lower < discard_percentile_higher < 100, "Invalid percentiles")

    sample_similarity = create_sample_similarity(
        similarity_name,
        sample_similarity_resize=sample_similarity_resize,
        sample_similarity_dtype=sample_similarity_dtype,
        **kwargs,
    )

    is_cond = input_desc["input_model_num_classes"] > 0

    rng = np.random.RandomState(get_kwarg("rng_seed", kwargs))

    lat_e0 = sample_random(rng, (input_model_num_samples, input_model_z_size), input_model_z_type)
    lat_e1 = sample_random(rng, (input_model_num_samples, input_model_z_size), input_model_z_type)
    lat_e1 = batch_interp(lat_e0, lat_e1, epsilon, interp)

    labels = None
    if is_cond:
        labels = torch.from_numpy(rng.randint(0, input_model_num_classes, (input_model_num_samples,)))

    distances = []

    with tqdm(
        disable=not verbose, leave=False, unit="samples", total=input_model_num_samples, desc="Perceptual Path Length"
    ) as t, torch.no_grad():
        for begin_id in range(0, input_model_num_samples, batch_size):
            end_id = min(begin_id + batch_size, input_model_num_samples)
            batch_sz = end_id - begin_id

            batch_lat_e0 = lat_e0[begin_id:end_id]
            batch_lat_e1 = lat_e1[begin_id:end_id]
            if is_cond:
                batch_labels = labels[begin_id:end_id]

            if cuda:
                batch_lat_e0 = batch_lat_e0.cuda(non_blocking=True)
                batch_lat_e1 = batch_lat_e1.cuda(non_blocking=True)
                if is_cond:
                    batch_labels = batch_labels.cuda(non_blocking=True)

            if is_cond:
                rgb_e01 = model.forward(
                    torch.cat((batch_lat_e0, batch_lat_e1), dim=0),
                    torch.cat((batch_labels, batch_labels), dim=0),
                )
            else:
                rgb_e01 = model.forward(torch.cat((batch_lat_e0, batch_lat_e1), dim=0))
            rgb_e0, rgb_e1 = rgb_e01.chunk(2)

            sim = sample_similarity(rgb_e0, rgb_e1)
            dist_lat_e01 = sim / (epsilon**2)
            distances.append(dist_lat_e01.cpu().numpy())

            t.update(batch_sz)

    distances = np.concatenate(distances, axis=0)

    cond, lo, hi = None, None, None
    if discard_percentile_lower is not None:
        lo = np.percentile(distances, discard_percentile_lower, interpolation="lower")
        cond = lo <= distances
    if discard_percentile_higher is not None:
        hi = np.percentile(distances, discard_percentile_higher, interpolation="higher")
        cond = np.logical_and(cond, distances <= hi)
    if cond is not None:
        distances = np.extract(cond, distances)

    out = {
        KEY_METRIC_PPL_MEAN: float(np.mean(distances)),
        KEY_METRIC_PPL_STD: float(np.std(distances)),
    }
    if reduction == "none":
        out[KEY_METRIC_PPL_RAW] = distances

    vprint(verbose, f"Perceptual Path Length: {out[KEY_METRIC_PPL_MEAN]:.7g} ± {out[KEY_METRIC_PPL_STD]:.7g}")

    return out
