import multiprocessing
import os
import sys
import tempfile

import numpy as np
import torch
import torch.hub
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torch_fidelity import GenerativeModelModuleWrapper
from torch_fidelity.datasets import ImagesPathDataset
from torch_fidelity.defaults import DEFAULTS
from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.generative_model_base import GenerativeModelBase
from torch_fidelity.generative_model_onnx import GenerativeModelONNX
from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.registry import DATASETS_REGISTRY, FEATURE_EXTRACTORS_REGISTRY, SAMPLE_SIMILARITY_REGISTRY, \
    INTERPOLATION_REGISTRY, NOISE_SOURCE_REGISTRY


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


def create_sample_similarity(name, cuda=True, **kwargs):
    vassert(name in SAMPLE_SIMILARITY_REGISTRY, f'Sample similarity "{name}" not registered')
    vprint(get_kwarg('verbose', kwargs), f'Creating sample similarity "{name}"')
    cls = SAMPLE_SIMILARITY_REGISTRY[name]
    sample_similarity = cls(name, **kwargs)
    sample_similarity.eval()
    if cuda:
        sample_similarity.cuda()
    return sample_similarity


def sample_random(rng, shape, z_type):
    fn_noise_src = NOISE_SOURCE_REGISTRY.get(z_type, None)
    vassert(fn_noise_src is not None, f'Noise source "{z_type}" not registered')
    return fn_noise_src(rng, shape)


def batch_interp(a, b, t, method):
    fn_interpolate = INTERPOLATION_REGISTRY.get(method, None)
    vassert(method is not None, f'Interpolation method "{method}" not registered')
    return fn_interpolate(a, b, t)


def get_featuresdict_from_dataset(input, feat_extractor, batch_size, cuda, save_cpu_ram, verbose):
    vassert(isinstance(input, Dataset), 'Input can only be a Dataset instance')
    vassert(torch.is_tensor(input[0]), 'Input Dataset should return torch.Tensor')
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

    with tqdm(disable=not verbose, leave=False, unit='samples', total=len(input), desc='Processing samples') as t, \
            torch.no_grad():
        for bid, batch in enumerate(dataloader):
            if cuda:
                batch = batch.cuda(non_blocking=True)

            features = feat_extractor(batch)
            featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
            featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

            if out is None:
                out = featuresdict
            else:
                out = {k: out[k] + featuresdict[k] for k in out.keys()}
            t.update(batch.shape[0])

    vprint(verbose, 'Processing samples')

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}

    return out


def get_featuresdict_from_generative_model(gen_model, feat_extractor, num_samples, batch_size, cuda, rng_seed, verbose):
    vassert(isinstance(gen_model, GenerativeModelBase), 'Input can only be a GenerativeModel instance')
    vassert(
        isinstance(feat_extractor, FeatureExtractorBase), 'Feature extractor is not a subclass of FeatureExtractorBase'
    )

    if batch_size > num_samples:
        batch_size = num_samples

    out = None

    rng = np.random.RandomState(rng_seed)

    if cuda:
        gen_model.cuda()

    with tqdm(disable=not verbose, leave=False, unit='samples', total=num_samples, desc='Processing samples') as t, \
            torch.no_grad():
        for sample_start in range(0, num_samples, batch_size):
            sample_end = min(sample_start + batch_size, num_samples)
            sz = sample_end - sample_start

            noise = NOISE_SOURCE_REGISTRY[gen_model.z_type](rng, (sz, gen_model.z_size))
            if cuda:
                noise = noise.cuda(non_blocking=True)
            gen_args = [noise]
            if gen_model.num_classes > 0:
                cond_labels = torch.from_numpy(rng.randint(low=0, high=gen_model.num_classes, size=(sz,), dtype=np.int))
                if cuda:
                    cond_labels = cond_labels.cuda(non_blocking=True)
                gen_args.append(cond_labels)

            fakes = gen_model(*gen_args)
            features = feat_extractor(fakes)
            featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
            featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

            if out is None:
                out = featuresdict
            else:
                out = {k: out[k] + featuresdict[k] for k in out.keys()}
            t.update(sz)

    vprint(verbose, 'Processing samples')

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}

    return out


def make_input_descriptor_from_int(input_int, **kwargs):
    vassert(input_int in (1, 2), 'Supported input slots: 1, 2')
    inputX = f'input{input_int}'
    input = get_kwarg(inputX, kwargs)
    input_desc = {
        'input': input,
        'input_cache_name': get_kwarg(f'{inputX}_cache_name', kwargs),
        'input_model_z_type': get_kwarg(f'{inputX}_model_z_type', kwargs),
        'input_model_z_size': get_kwarg(f'{inputX}_model_z_size', kwargs),
        'input_model_num_classes': get_kwarg(f'{inputX}_model_num_classes', kwargs),
        'input_model_num_samples': get_kwarg(f'{inputX}_model_num_samples', kwargs),
    }
    if type(input) is str and input in DATASETS_REGISTRY:
        input_desc['input_cache_name'] = input
    return input_desc


def make_input_descriptor_from_str(input_str):
    vassert(type(input_str) is str and input_str in DATASETS_REGISTRY,
            f'Supported input str: {list(DATASETS_REGISTRY.keys())}')
    return {
        'input': input_str,
        'input_cache_name': input_str,
        'input_model_z_type': DEFAULTS['input1_model_z_type'],
        'input_model_z_size': DEFAULTS['input1_model_z_size'],
        'input_model_num_classes': DEFAULTS['input1_model_num_classes'],
        'input_model_num_samples': DEFAULTS['input1_model_num_samples'],
    }


def prepare_input_from_descriptor(input_desc, **kwargs):
    bad_input = False
    input = input_desc['input']
    if type(input) is str:
        if input in DATASETS_REGISTRY:
            datasets_root = get_kwarg('datasets_root', kwargs)
            datasets_download = get_kwarg('datasets_download', kwargs)
            fn_instantiate = DATASETS_REGISTRY[input]
            if datasets_root is None:
                datasets_root = os.path.join(torch.hub._get_torch_home(), 'fidelity_datasets')
            os.makedirs(datasets_root, exist_ok=True)
            input = fn_instantiate(datasets_root, datasets_download)
        elif os.path.isdir(input):
            samples_find_deep = get_kwarg('samples_find_deep', kwargs)
            samples_find_ext = get_kwarg('samples_find_ext', kwargs)
            samples_ext_lossy = get_kwarg('samples_ext_lossy', kwargs)
            verbose = get_kwarg('verbose', kwargs)
            input = glob_samples_paths(input, samples_find_deep, samples_find_ext, samples_ext_lossy, verbose)
            vassert(len(input) > 0, f'No samples found in {input} with samples_find_deep={samples_find_deep}')
            input = ImagesPathDataset(input)
        elif os.path.isfile(input) and input.endswith('.onnx'):
            input = GenerativeModelONNX(
                input,
                input_desc['input_model_z_size'],
                input_desc['input_model_z_type'],
                input_desc['input_model_num_classes']
            )
        elif os.path.isfile(input) and input.endswith('.pth'):
            input = torch.jit.load(input, map_location='cpu')
            input = GenerativeModelModuleWrapper(
                input,
                input_desc['input_model_z_size'],
                input_desc['input_model_z_type'],
                input_desc['input_model_num_classes']
            )
        else:
            bad_input = True
    elif isinstance(input, Dataset) or isinstance(input, GenerativeModelBase):
        pass
    else:
        bad_input = True
    vassert(
        not bad_input,
        f'Input descriptor "input" field can be either an instance of Dataset, GenerativeModelBase class, or a string, '
        f'such as a path to a name of a registered dataset ({", ".join(DATASETS_REGISTRY.keys())}), a directory with '
        f'file samples, or a path to an ONNX or PTH (JIT) module'
    )
    return input


def prepare_input_descriptor_from_input_id(input_id, **kwargs):
    vassert(type(input_id) is int or type(input_id) is str and input_id in DATASETS_REGISTRY,
            'Input can be either integer (1 or 2) specifying the first or the second set of kwargs, or a string as a '
            'shortcut for registered datasets')
    if type(input_id) is int:
        input_desc = make_input_descriptor_from_int(input_id, **kwargs)
    else:
        input_desc = make_input_descriptor_from_str(input_id)
    return input_desc


def prepare_input_from_id(input_id, **kwargs):
    input_desc = prepare_input_descriptor_from_input_id(input_id, **kwargs)
    return prepare_input_from_descriptor(input_desc, **kwargs)


def get_cacheable_input_name(input_id, **kwargs):
    input_desc = prepare_input_descriptor_from_input_id(input_id, **kwargs)
    return input_desc['input_cache_name']


def atomic_torch_save(what, path):
    path = os.path.expanduser(path)
    path_dir = os.path.dirname(path)
    fp = tempfile.NamedTemporaryFile(delete=False, dir=path_dir)
    try:
        torch.save(what, fp)
        fp.close()
        os.rename(fp.name, path)
    finally:
        fp.close()
        if os.path.exists(fp.name):
            os.remove(fp.name)


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
    atomic_torch_save(item, item_path)
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
        atomic_torch_save(items[n], p)
    return items


def extract_featuresdict_from_input_id(input_id, feat_extractor, **kwargs):
    batch_size = get_kwarg('batch_size', kwargs)
    cuda = get_kwarg('cuda', kwargs)
    rng_seed = get_kwarg('rng_seed', kwargs)
    verbose = get_kwarg('verbose', kwargs)
    input = prepare_input_from_id(input_id, **kwargs)
    if isinstance(input, Dataset):
        save_cpu_ram = get_kwarg('save_cpu_ram', kwargs)
        featuresdict = get_featuresdict_from_dataset(input, feat_extractor, batch_size, cuda, save_cpu_ram, verbose)
    else:
        input_desc = prepare_input_descriptor_from_input_id(input_id, **kwargs)
        num_samples = input_desc['input_model_num_samples']
        vassert(type(num_samples) is int and num_samples > 0, 'Number of samples must be positive')
        featuresdict = get_featuresdict_from_generative_model(
            input, feat_extractor, num_samples, batch_size, cuda, rng_seed, verbose
        )
    return featuresdict


def extract_featuresdict_from_input_id_cached(input_id, feat_extractor, **kwargs):
    cacheable_input_name = get_cacheable_input_name(input_id, **kwargs)

    def fn_recompute():
        return extract_featuresdict_from_input_id(input_id, feat_extractor, **kwargs)

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
