import numpy as np
import torch
from tqdm import tqdm

from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.lpips import LPIPS_VGG16
from torch_fidelity.utils import OnnxModel, sample_random, batch_interp

KEY_METRIC_PPL = 'perceptual_path_length'


def ppl_model_to_metric(**kwargs):
    """
    Inspired by https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
    """
    model = get_kwarg('model', kwargs)
    is_cuda = get_kwarg('cuda', kwargs)
    verbose = get_kwarg('verbose', kwargs)
    num_samples = get_kwarg('ppl_num_samples', kwargs)
    model_z_type = get_kwarg('model_z_type', kwargs)
    model_z_size = get_kwarg('model_z_size', kwargs)
    model_conditioning_num_classes = get_kwarg('model_conditioning_num_classes', kwargs)
    epsilon = get_kwarg('ppl_epsilon', kwargs)
    interp = get_kwarg('ppl_z_interp_mode', kwargs)
    batch_size = get_kwarg('batch_size', kwargs)

    vprint(verbose, 'Computing Perceptual Path Length')

    vassert(model_z_size is not None, 'Dimensionality of generator noise not specified ("model_z_size" argument)')
    vassert(model_conditioning_num_classes >= 0, 'Model can be unconditional (0 classes) or conditional (positive)')

    is_cond = model_conditioning_num_classes > 0

    if type(model) is str:
        model = OnnxModel(model)
    else:
        vassert(
            isinstance(model, torch.nn.Module),
            'Model can be either a path to ONNX model, or an instance of torch.nn.Module'
        )
        if is_cuda:
            model.cuda()
        model.eval()

    lpips = LPIPS_VGG16()
    if is_cuda:
        lpips.cuda()

    rng = np.random.RandomState(get_kwarg('rng_seed', kwargs))

    lat_e0 = sample_random(rng, (num_samples, model_z_size), model_z_type)
    lat_e1 = sample_random(rng, (num_samples, model_z_size), model_z_type)
    lat_e1 = batch_interp(lat_e0, lat_e1, epsilon, interp)

    labels = None
    if is_cond:
        labels = torch.from_numpy(rng.randint(0, model_conditioning_num_classes, (num_samples,)))

    distances = []

    with tqdm(disable=not verbose, leave=False, unit='samples', total=num_samples, desc='Processing samples') as t, \
            torch.no_grad():
        for begin_id in range(0, num_samples, batch_size):
            end_id = min(begin_id + batch_size, num_samples)
            batch_sz = end_id - begin_id

            batch_lat_e0 = lat_e0[begin_id:end_id]
            batch_lat_e1 = lat_e1[begin_id:end_id]
            if is_cond:
                batch_labels = labels[begin_id:end_id]

            if is_cuda:
                batch_lat_e0 = batch_lat_e0.cuda(non_blocking=True)
                batch_lat_e1 = batch_lat_e1.cuda(non_blocking=True)
                if is_cond:
                    batch_labels = batch_labels.cuda(non_blocking=True)

            if is_cond:
                rgb_e0 = model.forward(batch_lat_e0, batch_labels)
                rgb_e1 = model.forward(batch_lat_e1, batch_labels)
            else:
                rgb_e0 = model.forward(batch_lat_e0)
                rgb_e1 = model.forward(batch_lat_e1)

            rgb_e01 = torch.cat((rgb_e0, rgb_e1), dim=0)

            if rgb_e01.shape[-1] > 256:
                rgb_e01 = torch.nn.functional.interpolate(rgb_e01, (256, 256), mode='area')
            else:
                rgb_e01 = torch.nn.functional.interpolate(rgb_e01, (256, 256), mode='bilinear', align_corners=False)

            rgb_e01 = ((rgb_e01 + 1) * (255. / 2)).to(dtype=torch.uint8)

            rgb_e0, rgb_e1 = rgb_e01[0:batch_sz], rgb_e01[batch_sz:]

            dist_lat_e01 = lpips.forward(rgb_e0, rgb_e1) / (epsilon ** 2)
            distances.append(dist_lat_e01.cpu().numpy())

            t.update(batch_sz)

    distances = np.concatenate(distances, axis=0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_distances = np.extract(np.logical_and(lo <= distances, distances <= hi), distances)
    metric = float(np.mean(filtered_distances))

    return {
        KEY_METRIC_PPL: metric,
    }
