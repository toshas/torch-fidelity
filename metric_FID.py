import os
import sys

import numpy as np
import scipy.linalg
import torch

from image_features_pipeline import glob_images_path, get_features

KEY_FID = 'frechet_inception_distance'


def calculate_statistics_of_features(features):
    """
    Calculation of the statistics used by the FID.
    Params:
    -- features    : Classifier features
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    assert torch.is_tensor(features) and features.dim() == 2
    features = features.numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def load_statistics(path):
    assert path.endswith('.npz')
    with np.load(path) as f:
        mu, sigma = f['mu'][:], f['sigma'][:]
    return mu, sigma


def store_statistics(path, stats):
    with open(path, 'wb') as f:
        np.savez_compressed(f, {
            'mu': stats[0],
            'sigma': stats[1],
        })


def calculate_metric_of_statistics(stat_1, stat_2, verbose=True):
    """
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- stat_1   : Statistics of the 1st distribution
    -- stat_2   : Statistics of the 2nd distribution

    Returns:
    --   : The Frechet Distance.
    """
    eps = 1e-6

    mu1, sigma1 = stat_1
    mu2, sigma2 = stat_2

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        if verbose:
            print(
                f'WARNING: fid calculation produces singular product; '
                f'adding {eps} to diagonal of cov estimates',
                sys.stderr
            )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=verbose)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return {
        KEY_FID: float(fid),
    }




from feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

INCEPTION_FEATURES_FOR_FID = '2048'


def calculate_statistics_of_images(files, model, batch_size=50, cuda=False, verbose=True):
    """
    Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    features = get_features(files, model, batch_size, cuda, verbose)

    features = features[INCEPTION_FEATURES_FOR_FID]

    return calculate_statistics_of_features(features)


def calculate_statistics_of_path(path, recurse, model, batch_size, cuda, verbose):
    if path.endswith('.npz'):
        stat = load_statistics(path)
    else:
        files = glob_images_path(path, recurse, verbose)
        stat = calculate_statistics_of_images(files, model, batch_size, cuda, verbose)
    return stat


def calculate_metric_of_paths(paths, recurse, batch_size, cuda, model_path, verbose):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    model = FeatureExtractorInceptionV3(
        [INCEPTION_FEATURES_FOR_FID],
        normalize_input=False,
        inception_weights_path=model_path
    )
    model.eval()
    if cuda:
        model.cuda()

    stat_1 = calculate_statistics_of_path(paths[0], recurse, model, batch_size, cuda, verbose)
    stat_2 = calculate_statistics_of_path(paths[1], recurse, model, batch_size, cuda, verbose)
    metric = calculate_metric_of_statistics(stat_1, stat_2, verbose)

    return metric
