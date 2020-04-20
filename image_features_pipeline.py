import multiprocessing
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm


def glob_images_path(path, recurse, verbose):
    have_lossy = False
    files = []
    for r, d, ff in os.walk(path):
        if not recurse and os.path.realpath(r) != os.path.realpath(path):
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


class ImagesPathDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        width, height = img.size
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(height, width, 3)
        img = img.permute(2, 0, 1).float()
        img = (img - 128) / 128
        return img


def get_features(files, model, batch_size=50, cuda=False, verbose=False):
    """
    Extract features from a list of images.

    Params:
    -- files       : List of image files paths
    -- model       : Feature extractor instance, inherited from FeatureExtractorBase
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if batch_size > len(files):
        if verbose:
            print('WARNING: Batch size is bigger than the data size. Setting batch size to data size', sys.stderr)
        batch_size = len(files)

    dataset = ImagesPathDataset(files)

    dataloader = DataLoader(
        dataset,
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
            features = model(batch)
        features = model.convert_features_tuple_to_dict(features)
        features = {k: [v.cpu()] for k, v in features.items()}

        if out is None:
            out = features
        else:
            out = {k: out[k] + features[k] for k in out.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}

    return out
