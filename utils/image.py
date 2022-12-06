"""Image/Array utils"""
import torch
import numpy as np
from utils.misc import param_ndim_setup
import torch.nn.functional as F


def crop_and_pad(x, new_size=192, mode="constant", **kwargs):
    """
    Crop and/or pad input to new size.
    (Adapted from DLTK: https://github.com/DLTK/DLTK/blob/master/dltk/io/preprocessing.py)

    Args:
        x: (np.ndarray) input array, shape (N, H, W) or (N, H, W, D)
        new_size: (int or tuple/list) new size excluding the batch size
        mode: (string) padding value filling mode for numpy.pad() (compulsory in Numpy v1.18)
        kwargs: additional arguments to be passed to np.pad

    Returns:
        (np.ndarray) cropped and/or padded input array
    """
    assert isinstance(x, (np.ndarray, np.generic))
    new_size = param_ndim_setup(new_size, ndim=x.ndim - 1)

    dim = x.ndim - 1
    sizes = x.shape[1:]

    # Initialise padding and slicers
    to_padding = [[0, 0] for i in range(x.ndim)]
    slicer = [slice(0, x.shape[i]) for i in range(x.ndim)]

    # For each dimensions except the dim 0, set crop slicers or paddings
    for i in range(dim):
        if sizes[i] < new_size[i]:
            to_padding[i + 1][0] = (new_size[i] - sizes[i]) // 2
            to_padding[i + 1][1] = new_size[i] - sizes[i] - to_padding[i + 1][0]
        else:
            # Create slicer object to crop each dimension
            crop_start = int(np.floor((sizes[i] - new_size[i]) / 2.0))
            crop_end = crop_start + new_size[i]
            slicer[i + 1] = slice(crop_start, crop_end)

    return np.pad(x[tuple(slicer)], to_padding, mode=mode, **kwargs)


def normalise_intensity(
    x,
    mode="minmax",
    min_in=0.0,
    max_in=255.0,
    min_out=0.0,
    max_out=1.0,
    clip=False,
    clip_range_percentile=(0.05, 99.95),
):
    """
    Intensity normalisation (& optional percentile clipping)
    for both Numpy Array and Pytorch Tensor of arbitrary dimensions.

    The "mode" of normalisation indicates different ways to normalise the intensities, including:
    1) "meanstd": normalise to 0 mean 1 std;
    2) "minmax": normalise to specified (min, max) range;
    3) "fixed": normalise with a fixed ratio

    Args:
        x: (ndarray / Tensor, shape (N, *size))
        mode: (str) indicate normalisation mode
        min_in: (float) minimum value of the input (assumed value for fixed mode)
        max_in: (float) maximum value of the input (assumed value for fixed mode)
        min_out: (float) minimum value of the output
        max_out: (float) maximum value of the output
        clip: (boolean) value clipping if True
        clip_range_percentile: (tuple of floats) percentiles (min, max) to determine the thresholds for clipping

    Returns:
        x: (same as input) in-place op on input x
    """

    # determine data dimension
    dim = x.ndim - 1
    image_axes = tuple(range(1, 1 + dim))  # (1,2) for 2D; (1,2,3) for 3D

    # for numpy.ndarray
    if type(x) is np.ndarray:
        # Clipping
        if clip:
            # intensity clipping
            clip_min, clip_max = np.percentile(
                x, clip_range_percentile, axis=image_axes, keepdims=True
            )
            x = np.clip(x, clip_min, clip_max)

        # Normalise meanstd
        if mode == "meanstd":
            mean = np.mean(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            std = np.std(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode == "minmax":
            min_in = np.amin(x, axis=image_axes, keepdims=True)  # (N, *range(dim))
            max_in = np.amax(x, axis=image_axes, keepdims=True)  # (N, *range(dim)))
            x = (
                (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)
            )  # (!) multiple broadcasting)

        # Fixed ratio
        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError(
                "Intensity normalisation mode not understood."
                "Expect either one of: 'meanstd', 'minmax', 'fixed'"
            )

        # cast to float 32
        x = x.astype(np.float32)

    # for torch.Tensor
    elif type(x) is torch.Tensor:
        # todo: clipping not supported at the moment (requires Pytorch version of the np.percentile()

        if mode == "meanstd":
            mean = torch.mean(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            std = torch.std(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        elif mode == "minmax":
            # get min/max across dims by flattening first
            min_in = (
                x.flatten(start_dim=1, end_dim=-1).min(dim=1)[0].view(-1, *(1,) * dim)
            )  # (N, (1,)*dim)
            max_in = (
                x.flatten(start_dim=1, end_dim=-1).max(dim=1)[0].view(-1, *(1,) * dim)
            )  # (N, (1,)*dim)
            x = (
                (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)
            )  # (!) multiple broadcasting)

        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError(
                "Intensity normalisation mode not recognised."
                "Expect: 'meanstd', 'minmax', 'fixed'"
            )

        # cast to float32
        x = x.float()

    else:
        raise TypeError(
            "Input data type not recognised, support numpy.ndarray or torch.Tensor"
        )
    return x


def mask_and_crop(x, roi_mask):
    """
    Mask input by roi_mask and crop by roi_mask bounding box

    Args:
        x: (numpy.nadarry, shape (N, ch, *dims))
        roi_mask: (numpy.nadarry, shape (N, ch, *dims))

    Returns:
        mask and cropped x
    """
    # find brian mask bbox mask
    mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])

    # mask by roi mask(N, dim, *dims) * (N, 1, *dims) = (N, dim, *dims)
    x *= roi_mask

    # crop out DVF within the roi mask bounding box (N, dim, *dims_cropped)
    x = bbox_crop(x, mask_bbox)
    return x


def bbox_crop(x, bbox):
    """
    Crop image by slicing using bounding box indices (2D/3D)

    Args:
        x: (numpy.ndarray, shape (N, ch, *dims))
        bbox: (list of tuples) [*(bbox_min_index, bbox_max_index)]

    Returns:
        x cropped using bounding box
    """
    # slice all of batch and channel
    slicer = [slice(0, x.shape[0]), slice(0, x.shape[1])]

    # slice image dimensions
    for bb in bbox:
        slicer.append(slice(*bb))
    return x[tuple(slicer)]


def bbox_from_mask(mask, pad_ratio=0.2):
    """
    Find a bounding box indices of a mask (with positive > 0)
    The output indices can be directly used for slicing
    - for 2D, find the largest bounding box out of the N masks
    - for 3D, find the bounding box of the volume mask

    Args:
        mask: (numpy.ndarray, shape (N, H, W) or (N, H, W, D)
        pad_ratio: (int or tuple) the ratio of between the mask bounding box to image boundary to pad

    Return:
        bbox: (list of tuples) [*(bbox_min_index, bbox_max_index)]
        bbox_mask: (numpy.ndarray shape (N, mH, mW) or (N, mH, mW, mD)) binary mask of the bounding box
    """
    dim = mask.ndim - 1
    mask_shape = mask.shape[1:]
    pad_ratio = param_ndim_setup(pad_ratio, dim)

    # find non-zero locations in the mask
    nonzero_indices = np.nonzero(mask > 0)
    bbox = [
        (nonzero_indices[i + 1].min(), nonzero_indices[i + 1].max()) for i in range(dim)
    ]

    # pad pad_ratio of the minimum distance
    #  from mask bounding box to the image boundaries (half each side)
    for i in range(dim):
        if pad_ratio[i] > 1:
            print(f"Invalid padding value (>1) on dimension {dim}, set to 1")
            pad_ratio[i] = 1
    bbox_padding = [
        pad_ratio[i] * min(bbox[i][0], mask_shape[i] - bbox[i][1]) for i in range(dim)
    ]
    # "padding" by modifying the bounding box indices
    bbox = [
        (bbox[i][0] - int(bbox_padding[i] / 2), bbox[i][1] + int(bbox_padding[i] / 2))
        for i in range(dim)
    ]

    # bbox mask
    bbox_mask = np.zeros(mask.shape, dtype=np.float32)
    slicer = [slice(0, mask.shape[0])]  # all slices/batch
    for i in range(dim):
        slicer.append(slice(*bbox[i]))
    bbox_mask[tuple(slicer)] = 1.0
    return bbox, bbox_mask


def roi_crop(x, mask, dim):
    """Crop input Tensor by the bounding box of roi mask"""

    if type(x) == torch.Tensor:
        # TODO: Tensor version of bbox_from_mask
        bbox, _ = bbox_from_mask(mask.squeeze(1).cpu().numpy())
    elif type(x) == np.ndarray:
        bbox, _ = bbox_from_mask(mask)

    for i in range(dim):
        x = x.narrow(i + 2, int(bbox[i][0]), int(bbox[i][1] - bbox[i][0]))
    # returns a view instead of a copy?
    return x


def avg_filtering(x, filter_size=7):
    """Applies average filtering to Tensor of size (N, 1, *sizes)"""
    dim = x.ndim - 2
    avg_filter = torch.ones(1, 1, *(filter_size,) * dim).type_as(x)
    pad = filter_size // 2
    conv_fn = getattr(F, f"conv{dim}d")
    return conv_fn(x, avg_filter, padding=pad)
