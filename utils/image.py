"""Image/Array utils"""
import numpy as np
from PIL import Image
import torch


def normalise_intensity(x,
                        mode="minmax",
                        min_in=0.0, max_in=255.0,
                        min_out=0.0, max_out=1.0,
                        clip=False, clip_range_percentile=(0.05, 99.95),
                        ):
    """
    Intensity normalisation (& optional percentile clipping)
    for both Numpy Array and Pytorch Tensor of arbitrary dimensions.

    The "mode" of normalisation indicates different ways to normalise the intensities, including:
    1) "meanstd": normalise to 0 mean 1 std;
    2) "minmax": normalise to specified (min, max) range;
    3) "fixed": normalise with a fixed ratio

    Args:
        x: (ndarray / Tensor, shape (N, *dims))
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
    dim = len(x.shape) - 1
    image_axis = tuple(range(1, 1 + dim))  # (1,2) for 2D; (1,2,3) for 3D

    # Numpy Array version
    if type(x) is np.ndarray:

        # Clipping
        if clip:
            # intensity clipping
            clip_min, clip_max = np.percentile(x, clip_range_percentile, axis=image_axis, keepdims=True)
            x = np.clip(x, clip_min, clip_max)

        # Normalise meanstd
        if mode is "meanstd":
            mean = np.mean(x, axis=image_axis, keepdims=True)  # (N, *range(dim))
            std = np.std(x, axis=image_axis, keepdims=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode is "minmax":
            min_in = np.amin(x, axis=image_axis, keepdims=True)  # (N, *range(dim))
            max_in = np.amax(x, axis=image_axis, keepdims=True)  # (N, *range(dim)))
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)  # (!) multiple broadcasting)

        # Fixed ratio
        elif mode is "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not understood."
                             "Expect either one of: 'meanstd', 'minmax', 'fixed'")


    # Pytorch Tensor version
    elif type(x) is torch.Tensor:
        # todo: clipping not supported at the moment (requires Pytorch version of the np.percentile()

        # Normalise meanstd
        if mode is "meanstd":
            mean = torch.mean(x, dim=image_axis, keepdim=True)  # (N, *range(dim))
            std = torch.std(x, dim=image_axis, keepdim=True)  # (N, *range(dim))
            x = (x - mean) / std  # axis should match & broadcast

        # Normalise minmax
        elif mode is "minmax":
            min_in = x.flatten(start_dim=1, end_dim=-1).min(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            max_in = x.flatten(start_dim=1, end_dim=-1).max(dim=1)[0].view(-1, *(1,)*dim)  # (N, (1,)*dim)
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)  # (!) multiple broadcasting)

        # Fixed ratio
        elif mode is "fixed":
            x = (x - min_in) * (max_out - min_out) / (max_in - min_in + 1e-12)

        else:
            raise ValueError("Intensity normalisation mode not recognised."
                             "Expect: 'meanstd', 'minmax', 'fixed'")

    else:
        raise RuntimeError("Intensity normalisation: input data type not recognised. "
                           "Expect: numpy.ndarray or torch.Tensor")
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

    """
    # find brian mask bbox mask
    mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])

    # mask by roi mask(N, dim, *dims) * (N, 1, *dims) = (N, dim, *dims)
    x *= roi_mask[:, np.newaxis, ...]

    # crop out DVF within the roi mask bounding box (N, dim, *dims_cropped)
    x = bbox_crop(x, mask_bbox)
    return x


def bbox_crop(x, bbox):
    """
    Crop image by slicing uisng bounding box indices

    Args:
        x: (numpy.ndarray, shape (N, ch, *dims))
        bbox: (tuple of tuples) ((bbox_min_dim1, bbox_max_dim1), (bbox_min_dim2, bbox_max_dim2), ...)

    Returns:
        cropped x
    """
    return x[:, :, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]


def bbox_from_mask(mask, pad_ratio=0.2):
    """
    Find a bounding box indices of a mask (with positive > 0)
    (largest bounding box of N masks)
    The indices follows Python indexing rule and can be directly used for slicing

    Args:
        mask: (ndarray, shape (N, H, W))
        pad_ratio: ratio of distance between the edge of mask bounding box to image boundary to pad

    Return:
        None: if structure in mask is too small

        bbox: list [[dim_i_min, dim_i_max], [dim_j_min, dim_j_max], ...] otherwise
        bbox_mask: (ndarray, shape (N, H, W)) binary mask which is 1 inside the bbox, 0 outside

    """
    mask_indices = np.nonzero(mask > 0)
    bbox_i = (mask_indices[1].min(), mask_indices[1].max()+1)
    bbox_j = (mask_indices[2].min(), mask_indices[2].max()+1)
    # bbox_k = (mask_indices[3].min(), mask_indices[3].max()+1)


    # pad 20% of minimum distance to the image boundaries (10% each side)
    if pad_ratio > 1:
        print("Invalid padding value (>1), set to 1")
        pad_ratio = 1

    bbox_pad_i = pad_ratio * min(bbox_i[0], mask.shape[1] - bbox_i[1])
    bbox_pad_j = pad_ratio * min(bbox_j[0], mask.shape[2] - bbox_j[1])
    # bbox_pad_k = 0.2 * min(bbox_k[0], image.shape[3] - bbox_k[1])

    bbox_i = (bbox_i[0] - int(bbox_pad_i/2), bbox_i[1] + int(bbox_pad_i/2))
    bbox_j = (bbox_j[0] - int(bbox_pad_j/2), bbox_j[1] + int(bbox_pad_j/2))
    # bbox_k = (bbox_k[0] - int(bbox_pad_k/2), bbox_k[1] + int(bbox_pad_k/2))
    bbox = [bbox_i, bbox_j]

    # bbox mask
    bbox_mask = np.zeros(mask.shape, dtype=np.float32)
    bbox_mask[:, bbox_i[0]:bbox_i[1], bbox_j[0]:bbox_j[1]] = 1.0

    return bbox, bbox_mask


def upsample_image(image, size):
    return np.array(Image.fromarray(image).resize((size, size)))



