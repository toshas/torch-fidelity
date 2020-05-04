import multiprocessing
import os
import sys

import torch
import torch.hub
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torch_fidelity.datasets import ImagesPathDataset
from torch_fidelity.defaults import DEFAULTS
from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.registry import DATASETS_REGISTRY, FEATURE_EXTRACTORS_REGISTRY


def glob_samples_paths(path, samples_find_deep, samples_find_ext, samples_ext_lossy=None, verbose=True):
    vassert(type(samples_find_ext) is str and samples_find_ext != '', 'Sample extensions not specified')
    vassert(
        samples_ext_lossy is None or type(samples_ext_lossy) is str, 'Lossy sample extensions can be None or string'
    )
    vprint(verbose, f'Looking for samples {"recursively" if samples_find_deep else "non-recursivelty"} in "{path}" '
                    f'with extensions {samples_find_ext}')
    samples_find_ext = [a.strip() for a in samples_find_ext.split(',') if a.strip() != '']
    if samples_ext_lossy is not None:
        samples_ext_lossy = [a.strip() for a in samples_ext_lossy.split(',') if a.strip() != '']
    have_lossy = False
    files = []
    for r, d, ff in os.walk(path):
        if not samples_find_deep and os.path.realpath(r) != os.path.realpath(path):
            continue
        for f in ff:
            ext = os.path.splitext(f)[1].lower()
            if len(ext) > 0 and ext[0] == '.':
                ext = ext[1:]
            if ext not in samples_find_ext:
                continue
            if samples_ext_lossy is not None and ext in samples_ext_lossy:
                have_lossy = True
            files.append(os.path.realpath(os.path.join(r, f)))
    files = sorted(files)
    vprint(verbose, f'Found {len(files)} samples'
                    f'{", some are lossy-compressed - this may affect metrics" if have_lossy else ""}')
    return files


def create_feature_extractor(name, list_features, cuda=True, **kwargs):
    vassert(name in FEATURE_EXTRACTORS_REGISTRY, f'Feature extractor "{name}" not registered')
    vprint(get_kwarg('verbose', kwargs), f'Creating feature extractor "{name}" with features {list_features}')
    cls = FEATURE_EXTRACTORS_REGISTRY[name]
    feat_extractor = cls(name, list_features, **kwargs)
    feat_extractor.eval()
    if cuda:
        feat_extractor.cuda()
    return feat_extractor


def get_featuresdict_from_dataset(input, feat_extractor, batch_size, cuda, save_cpu_ram, verbose):
    vassert(isinstance(input, Dataset), 'Input can only be a Dataset instance')
    vassert(
        isinstance(feat_extractor, FeatureExtractorBase), 'Feature extractor is not a subclass of FeatureExtractorBase'
    )

    if batch_size > len(input):
        batch_size = len(input)

    num_workers = 0 if save_cpu_ram else min(4, 2 * multiprocessing.cpu_count())

    dataloader = DataLoader(
        input,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=cuda,
    )

    out = None

    with tqdm(disable=not verbose, leave=False, unit='samples', total=len(input), desc='Processing samples') as t:
        for bid, batch in enumerate(dataloader):
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
            t.update(batch_size)

    vprint(verbose, 'Processing samples')

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}

    return out


def check_input(input):
    vassert(
        type(input) is str or isinstance(input, Dataset),
        f'Input can be either a Dataset instance, or a string (path to directory with samples, or one of the '
        f'registered datasets: {", ".join(DATASETS_REGISTRY.keys())}'
    )


def get_input_cacheable_name(input, cache_input_name=None):
    check_input(input)
    if type(input) is str:
        if input in DATASETS_REGISTRY:
            return input
        elif os.path.isdir(input):
            return cache_input_name
        else:
            raise ValueError(f'Unknown format of input string "{input}"')
    elif isinstance(input, Dataset):
        return cache_input_name


def prepare_inputs_as_datasets(
        input, samples_find_deep=False, samples_find_ext=DEFAULTS['samples_find_ext'],
        samples_ext_lossy=DEFAULTS['samples_ext_lossy'], datasets_root=None, datasets_download=True, verbose=True
):
    check_input(input)
    if type(input) is str:
        if input in DATASETS_REGISTRY:
            fn_instantiate = DATASETS_REGISTRY[input]
            if datasets_root is None:
                datasets_root = os.path.join(torch.hub._get_torch_home(), 'fidelity_datasets')
            os.makedirs(datasets_root, exist_ok=True)
            input = fn_instantiate(datasets_root, datasets_download)
        elif os.path.isdir(input):
            input = glob_samples_paths(input, samples_find_deep, samples_find_ext, samples_ext_lossy, verbose)
            vassert(len(input) > 0, f'No samples found in {input} with samples_find_deep={samples_find_deep}')
            input = ImagesPathDataset(input)
        else:
            raise ValueError(f'Unknown format of input string "{input}"')
    return input


def cache_lookup_one_recompute_on_miss(cached_filename, fn_recompute, **kwargs):
    if not get_kwarg('cache', kwargs):
        return fn_recompute()
    cache_root = get_kwarg('cache_root', kwargs)
    if cache_root is None:
        cache_root = os.path.join(torch.hub._get_torch_home(), 'fidelity_cache')
    os.makedirs(cache_root, exist_ok=True)
    item_path = os.path.join(cache_root, cached_filename + '.pt')
    if os.path.exists(item_path):
        vprint(get_kwarg('verbose', kwargs), f'Loading cached {item_path}')
        return torch.load(item_path, map_location='cpu')
    item = fn_recompute()
    if get_kwarg('verbose', kwargs):
        print(f'Caching {item_path}', file=sys.stderr)
    torch.save(item, item_path)
    return item


def cache_lookup_group_recompute_all_on_any_miss(cached_filename_prefix, item_names, fn_recompute, **kwargs):
    verbose = get_kwarg('verbose', kwargs)
    if not get_kwarg('cache', kwargs):
        return fn_recompute()
    cache_root = get_kwarg('cache_root', kwargs)
    if cache_root is None:
        cache_root = os.path.join(torch.hub._get_torch_home(), 'fidelity_cache')
    os.makedirs(cache_root, exist_ok=True)
    cached_paths = [os.path.join(cache_root, cached_filename_prefix + a + '.pt') for a in item_names]
    if all([os.path.exists(a) for a in cached_paths]):
        out = {}
        for n, p in zip(item_names, cached_paths):
            vprint(verbose, f'Loading cached {p}')
            out[n] = torch.load(p, map_location='cpu')
        return out
    items = fn_recompute()
    for n, p in zip(item_names, cached_paths):
        vprint(verbose, f'Caching {p}')
        torch.save(items[n], p)
    return items


def extract_featuresdict_from_input(input, feat_extractor, **kwargs):
    input_ds = prepare_inputs_as_datasets(
        input,
        samples_find_deep=get_kwarg('samples_find_deep', kwargs),
        samples_find_ext=get_kwarg('samples_find_ext', kwargs),
        samples_ext_lossy=get_kwarg('samples_ext_lossy', kwargs),
        datasets_root=get_kwarg('datasets_root', kwargs),
        datasets_download=get_kwarg('datasets_download', kwargs),
        verbose=get_kwarg('verbose', kwargs),
    )
    featuresdict = get_featuresdict_from_dataset(
        input_ds,
        feat_extractor,
        get_kwarg('batch_size', kwargs),
        get_kwarg('cuda', kwargs),
        get_kwarg('save_cpu_ram', kwargs),
        get_kwarg('verbose', kwargs),
    )
    return featuresdict


def extract_featuresdict_from_input_cached(input, cacheable_input_name, feat_extractor, **kwargs):

    def fn_recompute():
        return extract_featuresdict_from_input(input, feat_extractor, **kwargs)

    if cacheable_input_name is not None:
        feat_extractor_name = feat_extractor.get_name()
        cached_filename_prefix = f'{cacheable_input_name}-{feat_extractor_name}-features-'
        featuresdict = cache_lookup_group_recompute_all_on_any_miss(
            cached_filename_prefix,
            feat_extractor.get_requested_features_list(),
            fn_recompute,
            **kwargs,
        )
    else:
        featuresdict = fn_recompute()
    return featuresdict
