import numpy as np
import torch
from tqdm import tqdm

from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.lpips import LPIPS_VGG16
from torch_fidelity.utils import OnnxModel

KEY_METRIC_PPL = 'perceptual_path_length'


def batch_normalize_last_dim(v):
    return v / (v ** 2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-7)


def batch_slerp(a, b, t):
    a = batch_normalize_last_dim(a)
    b = batch_normalize_last_dim(b)
    d = (a * b).sum(dim=-1, keepdim=True)
    p = t * d.acos()
    c = batch_normalize_last_dim(b - d * a)
    d = a * p.cos() + c * p.sin()
    return batch_normalize_last_dim(d)


def batch_lerp(a, b, t):
    return a + (b - a) * t


def batch_interp(a, b, t, method):
    vassert(method in ('lerp', 'slerp'), f'Unknown interpolation method "{method}"')
    return {
        'lerp': batch_lerp,
        'slerp': batch_slerp,
    }[method](a, b, t)


def sample_real(rng, shape, z_type):
    if z_type == 'normal':
        return torch.from_numpy(rng.randn(*shape)).float()
    elif z_type == 'uniform_0_1':
        return torch.from_numpy(rng.rand(*shape)).float()
    else:
        vassert(False, f'Sampling from "{z_type}" is not implemented')


def ppl_model_to_metric(**kwargs):
    model = get_kwarg('model', kwargs)
    is_cuda = get_kwarg('cuda', kwargs)
    verbose = get_kwarg('verbose', kwargs)
    num_samples = get_kwarg('ppl_num_samples', kwargs)
    model_z_type = get_kwarg('model_z_type', kwargs)
    model_z_size = get_kwarg('model_z_size', kwargs)
    epsilon = get_kwarg('ppl_epsilon', kwargs)
    interp = get_kwarg('ppl_z_interp_mode', kwargs)
    batch_size = get_kwarg('batch_size', kwargs)

    vprint(verbose, 'Computing Perceptual Path Length')

    vassert(model_z_size is not None, 'Dimensionality of generator noise not specified ("model_z_size" argument)')

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

    lat_t0 = sample_real(rng, (num_samples, model_z_size), model_z_type)
    lat_t1 = sample_real(rng, (num_samples, model_z_size), model_z_type)
    t = sample_real(rng, (num_samples, 1), 'uniform_0_1') * (1.0 - epsilon)
    lat_e0 = batch_interp(lat_t0, lat_t1, t, interp)
    lat_e1 = batch_interp(lat_t0, lat_t1, t + epsilon, interp)

    distances = []

    with tqdm(disable=not verbose, leave=False, unit='samples', total=num_samples, desc='Processing samples') as t, \
            torch.no_grad():
        for begin_id in range(0, num_samples, batch_size):
            end_id = min(begin_id + batch_size, num_samples)
            nz = end_id - begin_id

            batch_lat_e0 = lat_e0[begin_id:end_id]
            batch_lat_e1 = lat_e1[begin_id:end_id]

            if is_cuda:
                batch_lat_e0 = batch_lat_e0.cuda(non_blocking=True)
                batch_lat_e1 = batch_lat_e1.cuda(non_blocking=True)

            rgb_e0 = model.forward(batch_lat_e0)
            rgb_e1 = model.forward(batch_lat_e1)
            rgb_e01 = torch.cat((rgb_e0, rgb_e1), dim=0)

            if rgb_e01.shape[-1] > 256:
                rgb_e01 = torch.nn.functional.interpolate(rgb_e01, (256, 256), mode='area')
            else:
                rgb_e01 = torch.nn.functional.interpolate(rgb_e01, (256, 256), mode='bilinear', align_corners=False)

            rgb_e01 = ((rgb_e01 + 1) * (255. / 2)).to(dtype=torch.uint8)

            rgb_e0, rgb_e1 = rgb_e01[0:nz], rgb_e01[nz:]

            dist_lat_e01 = lpips.forward(rgb_e0, rgb_e1) / (epsilon ** 2)
            distances.append(dist_lat_e01.cpu().numpy())

            t.update(nz)

    distances = np.concatenate(distances, axis=0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_distances = np.extract(np.logical_and(lo <= distances, distances <= hi), distances)
    metric = float(np.mean(filtered_distances))

    return {
        KEY_METRIC_PPL: metric,
    }
