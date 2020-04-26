import json
import multiprocessing
import os
import sys
from json.decoder import JSONDecodeError

import torch
import torch.hub
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torch_fidelity.datasets import ImagesPathDataset
from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.registry import DATASETS_REGISTRY, FEATURE_EXTRACTORS_REGISTRY


def json_decode_string(s):
    try:
        out = json.loads(s)
    except JSONDecodeError as e:
        print(f'Failed to decode JSON string: {s}', file=sys.stderr)
        raise
    return out


def glob_image_paths(path, glob_recursively, verbose):
    have_lossy = False
    files = []
    for r, d, ff in os.walk(path):
        if not glob_recursively and os.path.realpath(r) != os.path.realpath(path):
            continue
        for f in ff:
            ext = os.path.splitext(f)[1].lower()
            if ext not in ('.png', '.jpg', '.jpeg'):
                continue
            if ext in ('.jpg', '.jpeg'):
                have_lossy = True
            files.append(os.path.realpath(os.path.join(r, f)))
    files = sorted(files)
    if verbose:
        print(f'Found {len(files)} images in "{path}"'
              f'{". Some images are lossy-compressed - this may affect metrics!" if have_lossy else ""}',
              file=sys.stderr)
    return files


def create_feature_extractor(name, list_features, cuda=True, **kwargs):
    assert name in FEATURE_EXTRACTORS_REGISTRY, f'Feature extractor "{name}" not registered'
    cls = FEATURE_EXTRACTORS_REGISTRY[name]
    feat_extractor = cls(name, list_features, **kwargs)
    feat_extractor.eval()
    if cuda:
        feat_extractor.cuda()
    return feat_extractor


def get_featuresdict_from_dataset(input, feat_extractor, batch_size, cuda, verbose):
    assert isinstance(input, Dataset), 'Input can only be a Dataset instance'
    assert isinstance(feat_extractor, FeatureExtractorBase), \
        'Feature extractor is not a subclass of FeatureExtractorBase'

    if batch_size > len(input):
        batch_size = len(input)

    dataloader = DataLoader(
        input,
        batch_size=batch_size,
        drop_last=False,
        num_workers=min(4, 2 * multiprocessing.cpu_count()),
        pin_memory=cuda,
    )

    out = None

    for batch in tqdm(dataloader, disable=not verbose):
        if cuda:
            batch = batch.cuda(non_blocking=True)

        with torch.no_grad():
            features = feat_extractor(batch)
        featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
        featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

        if out is None:
            out = featuresdict
        else:
            out = {k: out[k] + featuresdict[k] for k in out.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}

    return out


def check_input(input):
    assert type(input) is str or isinstance(input, Dataset), \
        f'Input can be either a Dataset instance, or a string (path to directory with images, or one of the ' \
        f'registered datasets: {", ".join(DATASETS_REGISTRY.keys())}'


def get_input_cacheable_name(input):
    check_input(input)
    if type(input) is str:
        if input in DATASETS_REGISTRY:
            return input
        elif os.path.isdir(input):
            return None
        else:
            raise ValueError(f'Unknown format of input string "{input}"')
    elif isinstance(input, Dataset):
        assert hasattr(input, 'name'), \
            'Please use "name" attribute on your Dataset to enable statistics cachine (str) or disable it (None)'
        return getattr(input, 'name')


def prepare_inputs_as_datasets(
        input, glob_recursively=False, datasets_root=None, datasets_download_on=True, verbose=True
):
    check_input(input)
    if type(input) is str:
        if input in DATASETS_REGISTRY:
            fn_instantiate = DATASETS_REGISTRY[input]
            if datasets_root is None:
                datasets_root = os.path.join(torch.hub._get_torch_home(), 'fidelity_datasets')
            os.makedirs(datasets_root, exist_ok=True)
            input = fn_instantiate(datasets_root, datasets_download_on)
        elif os.path.isdir(input):
            input = glob_image_paths(input, glob_recursively, verbose)
            assert len(input) > 0, f'No images found in {input} with glob_recursively={glob_recursively}'
            input = ImagesPathDataset(input)
        else:
            raise ValueError(f'Unknown format of input string "{input}"')
    return input


def cache_lookup_one_recompute_on_miss(cached_filename, fn_recompute, **kwargs):
    if kwargs['cache_off']:
        return fn_recompute()
    cache_root = kwargs['cache_root']
    if cache_root is None:
        cache_root = os.path.join(torch.hub._get_torch_home(), 'fidelity_cache')
    os.makedirs(cache_root, exist_ok=True)
    item_path = os.path.join(cache_root, cached_filename + '.pt')
    if os.path.exists(item_path):
        if kwargs['verbose']:
            print(f'Loading cached {item_path}', file=sys.stderr)
        return torch.load(item_path, map_location='cpu')
    item = fn_recompute()
    if kwargs['verbose']:
        print(f'Caching {item_path}', file=sys.stderr)
    torch.save(item, item_path)
    return item


def cache_lookup_group_recompute_all_on_any_miss(cached_filename_prefix, item_names, fn_recompute, **kwargs):
    if kwargs['cache_off']:
        return fn_recompute()
    cache_root = kwargs['cache_root']
    if cache_root is None:
        cache_root = os.path.join(torch.hub._get_torch_home(), 'fidelity_cache')
    os.makedirs(cache_root, exist_ok=True)
    cached_paths = [os.path.join(cache_root, cached_filename_prefix + a + '.pt') for a in item_names]
    if all([os.path.exists(a) for a in cached_paths]):
        out = {}
        for n, p in zip(item_names, cached_paths):
            if kwargs['verbose']:
                print(f'Loading cached {p}', file=sys.stderr)
            out[n] = torch.load(p, map_location='cpu')
        return out
    items = fn_recompute()
    for n, p in zip(item_names, cached_paths):
        if kwargs['verbose']:
            print(f'Caching {p}', file=sys.stderr)
        torch.save(items[n], p)
    return items


def extract_featuresdict_from_input(input, feat_extractor, **kwargs):
    input_ds = prepare_inputs_as_datasets(
        input,
        glob_recursively=kwargs['glob_recursively'],
        datasets_root=kwargs['datasets_root'],
        datasets_download_on=kwargs['datasets_download_on'],
        verbose=kwargs['verbose'],
    )
    featuresdict = get_featuresdict_from_dataset(
        input_ds,
        feat_extractor,
        kwargs['batch_size'],
        kwargs['cuda'],
        kwargs['verbose'],
    )
    return featuresdict


def extract_featuresdict_from_input_cached(input, feat_extractor, **kwargs):

    def fn_recompute():
        return extract_featuresdict_from_input(input, feat_extractor, **kwargs)

    input_name = get_input_cacheable_name(input)
    if input_name is not None:
        feat_extractor_name = feat_extractor.get_name()
        cached_filename_prefix = f'{input_name}-{feat_extractor_name}-features-'
        featuresdict = cache_lookup_group_recompute_all_on_any_miss(
            cached_filename_prefix,
            feat_extractor.get_requested_features_list(),
            fn_recompute,
            **kwargs,
        )
    else:
        featuresdict = fn_recompute()
    return featuresdict
