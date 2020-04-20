import os
import numpy as np
import torch

from feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from image_features_pipeline import glob_images_path, get_features

INCEPTION_FEATURES_FOR_IS = 'logits_unbiased'
#INCEPTION_FEATURES_FOR_IS = 'logits'

KEY_IS_MEAN = 'inception_score_mean'
KEY_IS_STD = 'inception_score_std'


def calculate_statistics_of_features(features, splits, shuffle, shuffle_seed):
    assert torch.is_tensor(features) and features.dim() == 2
    N, C = features.shape
    if shuffle:
        rng = np.random.RandomState(shuffle_seed)
        features = features[rng.permutation(N), :]
    features = features.double()

    p = features.softmax(dim=1)
    log_p = features.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * N // splits) : ((i + 1) * N // splits), :]
        log_p_chunk = log_p[(i * N // splits) : ((i + 1) * N // splits), :]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = p_chunk * (log_p_chunk - q_chunk.log())
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)

    return {
        KEY_IS_MEAN: float(np.mean(scores)),
        KEY_IS_STD: float(np.std(scores)),
    }


def calculate_metric_of_path(
        path, recurse, batch_size, cuda, splits=10,
        shuffle=True, shuffle_seed=2020,
        model_path=None, verbose=True
):
    """Calculates the IS of a path"""
    if not os.path.exists(path):
        raise RuntimeError(f'Invalid path: {path}')

    model = FeatureExtractorInceptionV3(
        [INCEPTION_FEATURES_FOR_IS],
        normalize_input=False,
        inception_weights_path=model_path
    )
    model.eval()
    if cuda:
        model.cuda()

    files = glob_images_path(path, recurse, verbose)
    features = get_features(files, model, batch_size, cuda, verbose)
    features = features[INCEPTION_FEATURES_FOR_IS]

    return calculate_statistics_of_features(features, splits, shuffle, shuffle_seed)
