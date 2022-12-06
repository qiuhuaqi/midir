""" Dataset helper functions """
import random
import numpy as np
from omegaconf.listconfig import ListConfig
import torch
from utils.image import crop_and_pad, normalise_intensity
from utils.image_io import load_nifti


def _to_tensor(data_dict):
    # cast to Pytorch Tensor
    for name, data in data_dict.items():
        data_dict[name] = torch.from_numpy(data)
    return data_dict


def _crop_and_pad(data_dict, crop_size):
    # cropping and padding
    for name, data in data_dict.items():
        data_dict[name] = crop_and_pad(data, new_size=crop_size)
    return data_dict


def _normalise_intensity(data_dict, keys=None, vmin=0.0, vmax=1.0):
    """Normalise intensity of data in `data_dict` with `keys`"""
    if keys is None:
        keys = {"target", "source", "target_original"}

    # images in one pairing should be normalised using the same scaling
    vmin_in = np.amin(np.array([data_dict[k] for k in keys]))
    vmax_in = np.amax(np.array([data_dict[k] for k in keys]))

    for k, x in data_dict.items():
        if k in keys:
            data_dict[k] = normalise_intensity(
                x,
                min_in=vmin_in,
                max_in=vmax_in,
                min_out=vmin,
                max_out=vmax,
                mode="minmax",
                clip=True,
            )
    return data_dict


def _shape_checker(data_dict):
    """Check if all data points have the same shape
    if so return the common shape, if not print data type"""
    data_shapes_dict = {n: x.shape for n, x in data_dict.items()}
    shapes = [x for _, x in data_shapes_dict.items()]
    if all([s == shapes[0] for s in shapes]):
        common_shape = shapes[0]
        return common_shape
    else:
        raise AssertionError(
            f"Not all data points have the same shape, {data_shapes_dict}"
        )


def _magic_slicer(data_dict, slice_range=None, slicing=None):
    """Select all slices, one random slice, or some slices
    within `slice_range`, according to `slicing`
    """
    # slice selection
    num_slices = _shape_checker(data_dict)[0]

    # set range for slicing
    if slice_range is None:
        # all slices if not specified
        slice_range = (0, num_slices)
    else:
        # check slice_range
        assert isinstance(slice_range, (tuple, list, ListConfig))
        assert len(slice_range) == 2
        assert all(isinstance(s, int) for s in slice_range)
        assert slice_range[0] < slice_range[1]
        assert all(0 <= s <= num_slices for s in slice_range)

    # select slice(s)
    if slicing is None:
        # all slices within slice_range
        slicer = slice(slice_range[0], slice_range[1])

    elif slicing == "random":
        # randomly choose one slice within range
        z = random.randint(slice_range[0], slice_range[1] - 1)
        slicer = slice(z, z + 1)  # use slicer to keep dim

    elif isinstance(slicing, (list, tuple, ListConfig)):
        # slice several slices specified by slicing
        assert all(
            0 <= i <= 1 for i in slicing
        ), f"Relative slice positions {slicing} need to be within [0, 1]"
        slicer = tuple(
            int(i * (slice_range[1] - slice_range[0])) + slice_range[0] for i in slicing
        )

    else:
        raise ValueError(f"Slicing mode {slicing} not recognised.")

    # slicing
    for name, data in data_dict.items():
        data_dict[name] = data[slicer, ...]  # (N, H, W)

    return data_dict


def _load2d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, N) ->  (N, H, W)
        data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
    return data_dict


def _load3d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
        data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
    return data_dict
