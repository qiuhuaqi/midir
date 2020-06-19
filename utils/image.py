"""Image/Array utils"""
import torch
import numpy as np
from PIL import Image
from utils.misc import param_dim_setup

def crop_and_pad(x, new_size=192, mode="constant", **kwargs):
    """
    Crop and/or pad input to new size.
    (Adapted from DLTK: https://github.com/DLTK/DLTK/blob/master/dltk/io/preprocessing.py)

    Args:
        x: (np.ndarray) input array, shape (N, H, W) or (N, H, W, D)
        new_size: (int or tuple/list) new size excluding the first dimension
        mode: (string) padding value filling mode for numpy.pad() (default in Numpy v1.18, remove this if upgrade)
        kwargs: additional arguments to be passed to np.pad

    Returns:
        (np.ndarray) cropped and/or padded input array
    """
    assert isinstance(x, (np.ndarray, np.generic))
    new_size = param_dim_setup(new_size, dim=x.ndim-1)

    dim = x.ndim - 1
    sizes = x.shape[1:]

    # Initialise padding and slicers
    to_padding = [[0, 0] for i in range(x.ndim)]
    slicer = [slice(0, x.shape[i]) for i in range(x.ndim)]

    # For each dimensions except the dim 0, set crop slicers or paddings
    for i in range(dim):
        if sizes[i] < new_size[i]:
            to_padding[i+1][0] = (new_size[i] - sizes[i]) // 2
            to_padding[i+1][1] = new_size[i] - sizes[i] - to_padding[i+1][0]
        else:
            # Create slicer object to crop each dimension
            crop_start = int(np.floor((sizes[i] - new_size[i]) / 2.))
            crop_end = crop_start + new_size[i]
            slicer[i+1] = slice(crop_start, crop_end)

    return np.pad(x[tuple(slicer)], to_padding, mode=mode, **kwargs)


def normalise_intensity(x,
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
            clip_min, clip_max = np.percentile(x, clip_range_percentile, axis=image_axes, keepdims=True)
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
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)  # (!) multiple broadcasting)

        # Fixed ratio
        elif mode == "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not understood."
                             "Expect either one of: 'meanstd', 'minmax', 'fixed'")

        # cast to float 32
        x = x.astype(np.float32)


    # for torch.Tensor
    elif type(x) is torch.Tensor:
        # todo: clipping not supported at the moment (requires Pytorch version of the np.percentile()

        # Normalise meanstd
        if mode is "meanstd":
            mean = torch.mean(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            std = torch.std(x, dim=image_axes, keepdim=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode is "minmax":
            # get min/max across dims by flattening first
            min_in = x.flatten(start_dim=1, end_dim=-1).min(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            max_in = x.flatten(start_dim=1, end_dim=-1).max(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)  # (!) multiple broadcasting)

        # Fixed ratio
        elif mode is "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not recognised."
                             "Expect: 'meanstd', 'minmax', 'fixed'")

        # cast to float32
        x = x.float()

    else:
        raise TypeError("Input data type not recognised, support numpy.ndarray or torch.Tensor")


    return x



# todo: (for Pytorch version of the normalisation function) modify the following function from
# ()
# to enable the option `keep_dims` and (maybe) linear interpolation

# def percentile(t: torch.tensor, q: float) -> Union[int, float]:
#     """
#     Return the ``q``-th percentile of the flattened input tensor's data.
#
#     CAUTION:
#      * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
#      * Values are not interpolated, which corresponds to
#        ``numpy.percentile(..., interpolation="nearest")``.
#
#     :param t: Input tensor.
#     :param q: Percentile to compute, which must be between 0 and 100 inclusive.
#     :return: Resulting value (scalar).
#     """
#     # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
#     # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
#     # so that ``round()`` returns an integer, even if q is a np.float32.
#     k = 1 + round(.01 * float(q) * (t.numel() - 1))
#     result = t.view(-1).kthvalue(k).values.item()
#     return result


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
    pad_ratio = param_dim_setup(pad_ratio, dim)

    # find non-zero locations in the mask
    nonzero_indices = np.nonzero(mask > 0)
    bbox = [(nonzero_indices[i+1].min(), nonzero_indices[i+1].max())
                    for i in range(dim)]

    # pad pad_ratio of the minimum distance
    #  from mask bounding box to the image boundaries (half each side)
    for i in range(dim):
        if pad_ratio[i] > 1:
            print(f"Invalid padding value (>1) on dimension {dim}, set to 1")
            pad_ratio[i] = 1
    bbox_padding = [pad_ratio[i] * min(bbox[i][0], mask_shape[i] - bbox[i][1])
                    for i in range(dim)]
    # "padding" by modifying the bounding box indices
    bbox = [(bbox[i][0] - int(bbox_padding[i]/2), bbox[i][1] + int(bbox_padding[i]/2))
                    for i in range(dim)]

    # bbox mask
    bbox_mask = np.zeros(mask.shape, dtype=np.float32)
    slicer = [slice(0, mask.shape[0])]  # all slices/batch
    for i in range(dim):
        slicer.append(slice(*bbox[i]))
    bbox_mask[tuple(slicer)] = 1.0
    return bbox, bbox_mask


def upsample_image(image, size):
    return np.array(Image.fromarray(image).resize((size, size)))


""" Legacy 2D-only version of the functions """
# def bbox_from_mask(mask, pad_ratio=0.2):
#     """
#     Find a bounding box indices of a mask (with positive > 0)
#     The output indices can be directly used for slicing
#     - for 2D, find the largest bounding box out of the N masks
#     - for 3D, find the bounding box of the volume mask
#
#     Args:
#         mask: (ndarray, shape (N, H, W) or (N, H, W, D)
#         pad_ratio: ratio of distance between the edge of mask bounding box to image boundary to pad
#
#     Return:
#         None: if structure in mask is too small
#
#         bbox: list [[dim_i_min, dim_i_max], [dim_j_min, dim_j_max], ...] otherwise
#         bbox_mask: (ndarray, shape (N, H, W)) binary mask which is 1 inside the bbox, 0 outside
#
#     """
#     mask_indices = np.nonzero(mask > 0)
#     bbox_i = (mask_indices[1].min(), mask_indices[1].max()+1)
#     bbox_j = (mask_indices[2].min(), mask_indices[2].max()+1)
#     # bbox_k = (mask_indices[3].min(), mask_indices[3].max()+1)
#
#     # pad 20% of minimum distance to the image boundaries (10% each side)
#     if pad_ratio > 1:
#         print("Invalid padding value (>1), set to 1")
#         pad_ratio = 1
#
#     bbox_pad_i = pad_ratio * min(bbox_i[0], mask.shape[1] - bbox_i[1])
#     bbox_pad_j = pad_ratio * min(bbox_j[0], mask.shape[2] - bbox_j[1])
#     # bbox_pad_k = 0.2 * min(bbox_k[0], image.shape[3] - bbox_k[1])
#
#     bbox_i = (bbox_i[0] - int(bbox_pad_i/2), bbox_i[1] + int(bbox_pad_i/2))
#     bbox_j = (bbox_j[0] - int(bbox_pad_j/2), bbox_j[1] + int(bbox_pad_j/2))
#     # bbox_k = (bbox_k[0] - int(bbox_pad_k/2), bbox_k[1] + int(bbox_pad_k/2))
#     bbox = [bbox_i, bbox_j]
#
#     # bbox mask
#     bbox_mask = np.zeros(mask.shape, dtype=np.float32)
#     bbox_mask[:, bbox_i[0]:bbox_i[1], bbox_j[0]:bbox_j[1]] = 1.0
#
#     return bbox, bbox_mask

#
# def bbox_crop(x, bbox):
#     """
#     Crop image by slicing using bounding box indices
#
#     Args:
#         x: (numpy.ndarray, shape (N, ch, *dims))
#         bbox: (list of tuples) [*(bbox_min_index, bbox_max_index)]
#
#     Returns:
#         input cropped by according to bounding box
#     """
#     return x[:, :, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
#
