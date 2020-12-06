""" Dataset helper functions """
import random
import numpy as np
import torch
from utils.image import crop_and_pad, normalise_intensity
from utils.image_io import load_nifti


def _to_tensor(data_dict):
    # cast to Pytorch Tensor
    for name, data in data_dict.items():
        data_dict[name] = torch.from_numpy(data).float()
    return data_dict


def _crop_and_pad(data_dict, crop_size):
    # cropping and padding
    for name, data in data_dict.items():
        data_dict[name] = crop_and_pad(data, new_size=crop_size)
    return data_dict


def _normalise_intensity(data_dict, keys, vmin=0., vmax=1.):
    for name in keys:
        data_dict[name] = normalise_intensity(data_dict[name],
                                              min_out=vmin, max_out=vmax,
                                              mode="minmax", clip=True)
    return data_dict


def _load2d(data_path_dict, slice_range=None, random_slice=False):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, N) ->  (N, H, W)
        data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)

    # all slices if not specified
    if slice_range is None:
        slice_range = (0, min([d.shape[0] for _, d in data_dict.items()]))

    if random_slice:
        # randomly choose one slice within range
        z = random.randint(slice_range[0], slice_range[1])
        slicer = slice(z, z + 1)  # use slicer to keep dim
    else:
        # all slices within range
        slicer = slice(slice_range[0], slice_range[1])

    for name, data in data_dict.items():
        data_dict[name] = data[slicer, ...]  # (N, H, W)

    return data_dict


def _load3d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
        data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
    return data_dict
