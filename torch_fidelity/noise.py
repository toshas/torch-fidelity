import torch


def batch_normalize_last_dim(v, eps=1e-7):
    return v / (v ** 2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)


def random_normal(rng, shape):
    return torch.from_numpy(rng.randn(*shape)).float()


def random_unit(rng, shape):
    return batch_normalize_last_dim(torch.from_numpy(rng.rand(*shape)).float())


def random_uniform_0_1(rng, shape):
    return torch.from_numpy(rng.rand(*shape)).float()


def batch_lerp(a, b, t):
    return a + (b - a) * t


def batch_slerp_any(a, b, t, eps=1e-7):
    assert torch.is_tensor(a) and torch.is_tensor(b) and a.dim() >= 2 and a.shape == b.shape
    ndims, N = a.dim() - 1, a.shape[-1]
    a_1 = batch_normalize_last_dim(a, eps)
    b_1 = batch_normalize_last_dim(b, eps)
    d = (a_1 * b_1).sum(dim=-1, keepdim=True)
    mask_zero = (a_1.norm(dim=-1, keepdim=True) < eps) | (b_1.norm(dim=-1, keepdim=True) < eps)
    mask_collinear = (d > 1 - eps) | (d < -1 + eps)
    mask_lerp = (mask_zero | mask_collinear).repeat([1 for _ in range(ndims)] + [N])
    omega = d.acos()
    denom = omega.sin().clamp_min(eps)
    coef_a = ((1 - t) * omega).sin() / denom
    coef_b = (t * omega).sin() / denom
    out = coef_a * a + coef_b * b
    out[mask_lerp] = batch_lerp(a, b, t)[mask_lerp]
    return out


def batch_slerp_unit(a, b, t, eps=1e-7):
    out = batch_slerp_any(a, b, t, eps)
    out = batch_normalize_last_dim(out, eps)
    return out
